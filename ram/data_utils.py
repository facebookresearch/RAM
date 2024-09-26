"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import contextlib
import datetime
import gzip
import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from typing import List

import requests
import tqdm
from transformers import AutoTokenizer

try:
    from torch.multiprocessing import Pool
except ImportError:
    from multiprocessing import Pool

from pathlib import Path

from iopath.common.file_io import PathManager as _PathManager


@contextlib.contextmanager
def get_http_session():
    with requests.Session() as session:
        yield session


PathManager = _PathManager()

DUMMY_VERSION = "0.0"

try:
    from ram.internal_constants import cache_paths

    backup_cache_path = None
except ImportError:
    logging.warning(
        f"internal constants not available, cache is saved to user home dir"
    )
    cache_paths = None
    backup_cache_path = os.environ.get("HOME")


# re-use this function for other cache dirnames e.g. huggingface datasets cache
# by changing the cache_dirname argument
def get_ram_task_data_download_cache(
    cache_dirname="cache/ram_tasks_data_downloads",
    backup_cache_path=backup_cache_path,
    create_dirs: bool = True,
):
    if cache_paths is None:
        root_path = backup_cache_path

    elif os.path.exists(cache_paths["AWS_CACHE_ROOT"]):
        root_path = cache_paths["AWS_CACHE_ROOT"]

    elif os.path.exists(cache_paths["H2_CACHE_ROOT"]):
        root_path = cache_paths["H2_CACHE_ROOT"]

    elif os.path.exists(cache_paths["GITHUB_ACTIONS_CACHE_ROOT"]):
        root_path = cache_paths["GITHUB_ACTIONS_CACHE_ROOT"]

    cache_path = f"{root_path}/{cache_dirname}"
    if create_dirs:
        os.makedirs(cache_path, exist_ok=True)

    return cache_path


class DownloadableFile:
    """
    A class used to abstract any file that has to be downloaded online.

    Any task that needs to download a file needs to have a list RESOURCES
    that have objects of this class as elements.

    This class provides the following functionality:

    - Download a file from a URL / Google Drive
    - Untar the file if zipped
    - Checksum for the downloaded file
    - Send HEAD request to validate URL or Google Drive link

    An object of this class needs to be created with:

    - url <string> : URL or Google Drive id to download from
    - file_name <string> : File name that the file should be named
    - hashcode <string> : SHA256 hashcode of the downloaded file
    - zipped <boolean> : False if the file is not compressed
    """

    def __init__(self, url, file_name, hashcode, zipped=True):
        self.url = url
        self.file_name = file_name
        self.hashcode = hashcode
        self.zipped = zipped

    def checksum(self, dpath):
        """
        Checksum on a given file.

        :param dpath: path to the downloaded file.
        """
        sha256_hash = hashlib.sha256()
        with PathManager.open(os.path.join(dpath, self.file_name), "rb") as f:
            for byte_block in iter(lambda: f.read(65536), b""):
                sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != self.hashcode:
                # remove_dir(dpath)
                raise AssertionError(
                    f"Checksum for {self.file_name} from \n{self.url}\n"
                    f"does not match the expected checksum:\n"
                    f"{sha256_hash.hexdigest()} (received) != {self.hashcode} (expected)\n"
                    f"\nPlease try again. You may need to manually delete {self.file_name}."
                )
            else:
                logging.debug("Checksum Successful")

    def download_file(self, dpath):

        download(self.url, dpath, self.file_name)

        self.checksum(dpath)

        if self.zipped:
            untar(dpath, self.file_name)

    def check_header(self):
        """
        Performs a HEAD request to check if the URL / Google Drive ID is live.
        """
        with get_http_session() as session:
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/77.0.3865.90 Safari/537.36"
                )
            }
            response = session.head(self.url, allow_redirects=True, headers=headers)
            status = response.status_code

        assert status == 200


def download(url, path, fname, redownload=False, num_retries=5):
    """
    Download file using `requests`.

    If ``redownload`` is set to false, then will not download tar file again if it is
    present (default ``False``).
    """
    outfile = os.path.join(path, fname)
    download = not PathManager.exists(outfile) or redownload
    logging.info(f"Downloading {url} to {outfile}")
    retry = num_retries
    exp_backoff = [2**r for r in reversed(range(retry))]

    pbar = tqdm.tqdm(unit="B", unit_scale=True, desc="Downloading {}".format(fname))

    while download and retry > 0:
        response = None

        with get_http_session() as session:
            try:
                response = session.get(url, stream=True, timeout=5)

                # negative reply could be 'none' or just missing
                CHUNK_SIZE = 32768
                total_size = int(response.headers.get("Content-Length", -1))
                # server returns remaining size if resuming, so adjust total
                pbar.total = total_size
                done = 0

                with PathManager.open(outfile, "wb") as f:
                    for chunk in response.iter_content(CHUNK_SIZE):
                        if chunk:  # filter out keep-alive new chunks
                            f.write(chunk)
                        if total_size > 0:
                            done += len(chunk)
                            if total_size < done:
                                # don't freak out if content-length was too small
                                total_size = done
                                pbar.total = total_size
                            pbar.update(len(chunk))
                    break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ReadTimeout,
            ):
                retry -= 1
                pbar.clear()
                if retry > 0:
                    pl = "y" if retry == 1 else "ies"
                    logging.debug(
                        f"Connection error, retrying. ({retry} retr{pl} left)"
                    )
                    time.sleep(exp_backoff[retry])
                else:
                    logging.error("Retried too many times, stopped retrying.")
            finally:
                if response:
                    response.close()
    if retry <= 0:
        raise RuntimeError("Connection broken too many times. Stopped retrying.")

    if download and retry > 0:
        pbar.update(done - pbar.n)
        if done < total_size:
            raise RuntimeError(
                f"Received less data than specified in Content-Length header for "
                f"{url}. There may be a download problem."
            )

    pbar.close()


def make_dir(path):
    """
    Make the directory and any nonexistent parent directories (`mkdir -p`).
    """
    # the current working directory is a fine path
    if path != "":
        PathManager.mkdirs(path)


def remove_dir(path):
    """
    Remove the given directory, if it exists.
    """
    shutil.rmtree(path, ignore_errors=True)


def untar(path, fname, delete=True, flatten_tar=False):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    if ".zip" in fname:
        return _unzip(path, fname, delete=delete)
    else:
        return _untar(path, fname, delete=delete, flatten=flatten_tar)


def _untar(path, fname, delete=True, flatten=False):
    """
    Unpack the given archive file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    import tarfile

    logging.debug(f"unpacking {fname}")
    fullpath = os.path.join(path, fname)
    # very painfully manually extract files so that we can use PathManger.open
    # instead, lest we are using fb internal file services

    with tarfile.open(fileobj=PathManager.open(fullpath, "rb")) as tf:
        for item in tf:
            item_name = item.name
            while item_name.startswith("./"):
                # internal file systems will actually create a literal "."
                # directory, so we gotta watch out for that
                item_name = item_name[2:]
            if flatten:
                # flatten the tar file if there are subdirectories
                fn = os.path.join(path, os.path.split(item_name)[-1])
            else:
                fn = os.path.join(path, item_name)
            logging.debug(f"Extracting to {fn}")
            if item.isdir():
                PathManager.mkdirs(fn)
            elif item.isfile():
                with PathManager.open(fn, "wb") as wf, tf.extractfile(item.name) as rf:
                    tarfile.copyfileobj(rf, wf)
            else:
                raise NotImplementedError("No support for symlinks etc. right now.")

    if delete:
        try:
            PathManager.rm(fullpath)
        except PermissionError:
            logging.error(
                f"Tried to delete {fullpath} but got a permission error. This "
                "is known to happen in Windows and is probably not fatal."
            )


def ungzip(path, fname, deleteGZip=True):
    """
    Unzips the given gzip compressed file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool deleteGZip:
        If true, the compressed file will be deleted after extraction.
    """

    def _get_output_filename(input_fname):
        GZIP_EXTENSIONS = (".gz", ".gzip", ".tgz", ".tar")
        for ext in GZIP_EXTENSIONS:
            if input_fname.endswith(ext):
                return input_fname[: -len(ext)]
        return f"{input_fname}_unzip"

    logging.debug(f"unzipping {fname}")
    fullpath = os.path.join(path, fname)

    with gzip.open(PathManager.open(fullpath, "rb"), "r") as fin, PathManager.open(
        _get_output_filename(fullpath), "wb"
    ) as fout:
        shutil.copyfileobj(fin, fout)

    if deleteGZip:
        os.remove(fullpath)


def _unzip(path, fname, delete=True):
    """
    Unpack the given zip file to the same directory.

    :param str path:
        The folder containing the archive. Will contain the contents.

    :param str fname:
        The filename of the archive file.

    :param bool delete:
        If true, the archive will be deleted after extraction.
    """
    import zipfile

    logging.debug(f"unpacking {fname}")
    fullpath = os.path.join(path, fname)
    with zipfile.ZipFile(PathManager.open(fullpath, "rb"), "r") as zf:
        for member in zf.namelist():
            outpath = os.path.join(path, member)
            if zf.getinfo(member).is_dir():
                logging.debug(f"Making directory {outpath}")
                PathManager.mkdirs(outpath)
                continue
            logging.debug(f"Extracting to {outpath}")
            try:
                with zf.open(member, "r") as inf, PathManager.open(
                    outpath, "wb"
                ) as outf:
                    shutil.copyfileobj(inf, outf)
            except FileNotFoundError:
                logging.error(f"Failed to open ${member} and extract to ${outpath}")
    if delete:
        try:
            PathManager.rm(fullpath)
        except PermissionError:
            logging.error(
                f"Tried to delete {fullpath} but got a permission error. This "
                "is known to happen in Windows and is probably not fatal."
            )


def built(path, version_string=None):
    """
    Check if '.built' flag has been set for that task.

    If a version_string is provided, this has to match, or the version is regarded as
    not built.
    """
    if version_string:
        fname = os.path.join(path, ".built")
        if not PathManager.exists(fname):
            return False
        else:
            with PathManager.open(fname, "r") as read:
                text = read.read().split("\n")
            return len(text) > 1 and text[1] == version_string
    else:
        return PathManager.exists(os.path.join(path, ".built"))


def mark_done(path, version_string=None):
    """
    Mark this path as prebuilt.

    Marks the path as done by adding a '.built' file with the current timestamp
    plus a version description string if specified.

    :param str path:
        The file path to mark as built.

    :param str version_string:
        The version of this dataset.
    """
    with PathManager.open(os.path.join(path, ".built"), "w") as write:
        write.write(str(datetime.datetime.today()))
        if version_string:
            write.write("\n" + version_string)


def load_from_jsonl(file_name: str) -> List[dict]:
    def load_json_line(line: str, i: int, file_name: str):
        try:
            return json.loads(line)
        except:
            raise ValueError(f"Error in line {i+1}\n{line} of {file_name}")

    with open(file_name, "r", encoding="UTF-8") as f:
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f)]
    return data


def save_to_jsonl(data, filename, write_mode="w"):
    with open(filename, write_mode) as file:
        for item in data:
            json_str = json.dumps(item)
            file.write(json_str + "\n")


def load_from_txt(file):
    with open(file, "r") as f:
        text = f.read()
    return text


def validate_key(k: List[str]):
    for _k in ["input", "text", "prompt", "src", "content"]:
        if _k in k:
            return _k
    raise RuntimeError(f"Unable to determine key for {k}")


def get_content_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip()
        try:
            x = json.loads(line)
        except UnicodeDecodeError as e:
            print(f"Error when trying to decode '{line}': {str(e)}")
            raise
        return validate_key(x)


def wrap_text_with_chat_template(
    text: str, tokenizer: AutoTokenizer, system_prompt: str = None
):
    """Wrapping the text using a huggingface-curated chat templates.
    The expectation is that you provide the tokenizer of the target model you
    will use with the exported data.
    Naive logic here as we build the single turn convo with user role.

    See reference on chat templates here: https://huggingface.co/docs/transformers/en/chat_templating

    Args:
        text (str): text of the user prompt
        tokenizer (AutoTokenizer): Instantiated tokenizer.
            Example: AutoTokenizer.from_pretrained("<local_path>")
        system_prompt (str): added following the model if tokenizer supports that, otherwise the error will be returned. Default is None.
    """

    conversation_object = []

    if system_prompt is not None:
        conversation_object.append({"role": "system", "content": system_prompt})

    conversation_object.append({"role": "user", "content": text})

    wrapped_text = tokenizer.apply_chat_template(conversation_object, tokenize=False)

    return wrapped_text


def strip_bos(text: str, bos_token: str):
    # strip bos from the beginning of the sequence
    assert (
        text[: len(bos_token)] == bos_token
    ), f"input text does not contain {bos_token} in the beginning: {text[:len(bos_token)+10]}"
    text = text[len(bos_token) :].lstrip()
    return text


def load_sharded_dir(load_dir: str):
    """Load sharded dir with chunks of jsonl files

    Args:
        load_dir (str): path to dir with sharded jsonls

    Returns:
        data: list of dicts
    """
    assert Path(load_dir).exists()
    chunk_filenames = list(Path(load_dir).glob("data.chunk.*.jsonl"))
    num_digit_mask = len(chunk_filenames[0].stem.split(".")[-1])
    n_shards = len(chunk_filenames)
    if n_shards == 0:
        raise RuntimeError(f"No shards detected, is {load_dir} a dir?")
    data = []
    for i_shard in range(n_shards):
        data.extend(
            load_from_jsonl(f"{load_dir}/data.chunk.{i_shard:0{num_digit_mask}d}.jsonl")
        )
    return data


def get_tmp_dir():
    """Returns the tmp dir on /scratch for each compute node

    Raises:
        RuntimeError: error if called not within the SLURM job

    Returns:
        str: path to the job related dir
    """
    if "SLURM_JOB_ID" in os.environ:
        tmpdir = Path("/scratch/slurm_tmpdir/" + os.environ["SLURM_JOB_ID"])
        if tmpdir.exists():
            return tmpdir.as_posix()
    else:
        raise RuntimeError(f"Not a slurm job, SLURM_JOB_ID not available")


def map_str_to_uuid(str_to_map: str):
    """Generate a hash for a given string using v5 UUID
    In this case the md5 of str_to_map is computed, then some bits are truncated (for uuid namespace) and then
    the uuid is created given the md5 as the input name.

    Usage example: mapping prompts or src+tgt sequences in the dataset to a unique id.

    Args:
        str_to_map (str): string to map to an ID

    Returns:
        str: UUID converted to string
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str_to_map))


def extract_winner_llm_judge_pairv2(generation: str):
    labels_dict = {
        "[[A]]": "model_a",
        "[[B]]": "model_b",
        "[[C]]": "tie",
    }
    if generation.count("[[") > 1:
        return "tie"
    for kw, label in labels_dict.items():
        if kw in generation:
            return label
    return None

