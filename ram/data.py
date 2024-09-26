"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import json
import logging
import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import jsonlines
from transformers import AutoTokenizer

from ram.data_utils import (
    DUMMY_VERSION,
    built,
    get_ram_task_data_download_cache,
    load_from_jsonl,
    load_sharded_dir,
    map_str_to_uuid,
    mark_done,
)
from ram.data_utils import strip_bos as _strip_bos
from ram.data_utils import wrap_text_with_chat_template

root_cache_dirpath = get_ram_task_data_download_cache(f"cache/huggingface/datasets")
os.environ["HF_DATASETS_CACHE"] = root_cache_dirpath


class Example:
    def __init__(self, data: Dict = None, **kwargs):
        self.data = {}
        if data is not None:
            self.data.update(data)
        self.data.update(kwargs)

    def __str__(self):
        return str(self.data)

    def set_key(self, key: str, value: Any):
        self.data[key] = value

    def __getitem__(self, index: int):
        return self.data[index]

    def __repr__(self):
        return f"{self.data}"

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def get(self, key, default=""):
        return self.data.get(key, default)

    def get_all_keys(self):
        return self.data.keys()

    def clear(self):
        self.data = {}

    def apply_prompt_template(self, prompt_template):
        prompt_template(self)

    def serialize(self):
        # return dict of data for jsonl dump
        return self.data

    @classmethod
    def input_as_hash_key(cls):
        return cls.data["input"]

    def set_unique_id(self, hash_fn=input_as_hash_key):
        self.data["id"] = map_str_to_uuid(hash_fn(self))

    @classmethod
    def deserialize(cls, data):
        return cls(data)


class ExampleGroup:
    def __init__(self, data=None):
        self.data = []
        if data is not None:
            assert all(isinstance(ex, Example) for ex in data)
            self.data = data

    def __str__(self):
        return str(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __repr__(self):
        return f"{self.data}"

    def set_key(self, idx, value):
        self.data[idx] = value

    def get(self, idx, default=""):
        return self.data.get(idx, default)

    def clear(self):
        self.data = []

    def apply_group_prompt_template(self, prompt_template):
        prompt_template(self)

    def __len__(self):
        return len(self.data)

    def serialize(self):
        # go over each example and collect serialized view
        # constructor assumes that each element from self.data is Example
        return [ex.serialize() for ex in self.data]

    @classmethod
    def deserialize(cls, data):
        #  handle a list if data is not ExampleGroup
        if not isinstance(data, list):
            data = [data]
        examples = [Example.deserialize(ex) for ex in data]
        return cls(examples)


def load_prompt_template(
    file_paths: Union[List[str], str], keys: Union[List[str], str] = "input"
) -> Callable:
    if type(file_paths) != list:
        file_paths = [file_paths]
        keys = [keys]
    templates = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            content = file.read()
            content = content.replace("[INPUT]", "{input}")
            content = content.replace("[LABEL]", "{label}")
        templates.append(content)

    # Define mutator that is returned
    def mutate(example):
        kv_pairs = example.data
        for i in range(0, len(templates)):
            new_prompt = templates[i].format(**kv_pairs)
            example.set_key(keys[i], new_prompt)

    return mutate


def load_group_prompt_template(file_path, key="input"):
    with open(file_path, "r") as file:
        content = file.read()
        content = content.replace("[INPUT]", "{input}")
        content = content.replace("[LABEL]", "{label}")
    template = content

    # Define mutator that is returned
    def mutate(examples):
        kv_pairs = examples.data[0].data
        # TODO: extend to > 2 items
        if len(examples.data) == 2:
            ex2 = examples.data[1]
            # copy over key/value pairs so we can access them in the prompt template.
            for k in ex2.get_all_keys():
                kv_pairs[k + "2"] = ex2[k]
        new_prompt = template.format(**kv_pairs)
        new_example = Example()
        new_example.set_key(key, new_prompt)
        examples.clear()
        examples.data.append(new_example)

    return mutate


def add_instruct_tokens(
    ex,
    instr_start_symbol: str = "[INST] ",  # for llama 3+, other tokens are used
    instr_end_symbol: str = " [/INST]",
    key: str = "input",
) -> str:
    new_text = ex.get(key)
    if not (
        new_text.startswith(instr_start_symbol) and new_text.endswith(instr_end_symbol)
    ):
        new_text = instr_start_symbol + new_text + instr_end_symbol
    return new_text


class Dataset:
    def __init__(self, data: List[Optional[ExampleGroup]]):
        self.data = data

    def __str__(self):
        return f"Dataset length={len(self.data)}"

    def __getitem__(self, index):
        if isinstance(index, int):
            # return example group when indexed with single item
            return self.data[index]
        elif isinstance(index, slice):
            # for slices return sliced dataset
            return Dataset(self.data[index])
        else:
            raise RuntimeError("Unsupported indexing format")

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Dataset({self.data})"

    def flattened_iter(self):
        # go over every example from each example group
        for ex_group in self.data:
            for ex in ex_group:
                yield ex

    def __iter__(self):
        return iter(self.data)

    def add_example_group(self, examples: ExampleGroup):
        assert all(isinstance(ex, Example) for ex in examples)
        if isinstance(examples, list):
            self.data.append(ExampleGroup(examples))
        elif isinstance(examples, ExampleGroup):
            self.data.append(examples)
        else:
            self.data.append(ExampleGroup([examples]))

    def add_example(self, example: Example):
        assert isinstance(example, Example)
        self.add_example_group([example])

    def apply_prompt_template(self, prompt: Callable):
        for ex_group in self.data:
            for ex in ex_group:
                ex.apply_prompt_template(prompt)

    def apply_group_prompt_template(self, prompt: Callable):
        for ex_group in self.data:
            ex_group.apply_group_prompt_template(prompt)

    def serialize(self):
        example_groups = [ex_group.serialize() for ex_group in self.data]
        return example_groups

    def remove_empty_example_groups(self):
        # helps when mutators leave empty groups after processing
        new_data = []
        for ex_group in self.data:
            if len(ex_group) > 0:
                new_data.append(ex_group)
        self.data = new_data

    def shuffle(self):
        random.shuffle(self.data)

    @classmethod
    def deserialize(cls, data: List[ExampleGroup]):
        example_groups = [ExampleGroup.deserialize(ex_group) for ex_group in data]
        return cls(example_groups)

    def set_id_for_each_example(self, example_hash_fn=Example.input_as_hash_key):
        for ex in self.flattened_iter():
            ex.set_unique_id(hash_fn=example_hash_fn)

    @staticmethod
    def store_sharded_data(data, dir, num_shards):
        def get_shard_data(shard_id):
            ret = []
            i = shard_id
            while i < len(data):
                ret.append(data[i])
                i += num_shards
            return ret

        os.makedirs(dir, exist_ok=True)
        for shard_id in range(num_shards):
            opath = os.path.join(dir, f"data.chunk.{shard_id:04}.jsonl")
            with jsonlines.open(opath, "w") as fo:
                for ex in get_shard_data(shard_id):
                    fo.write(ex)

    @staticmethod
    def check_built_and_store(data, save_dir, num_shards, version_string=DUMMY_VERSION):
        if built(save_dir, version_string):
            logging.warning(
                f"Data version {version_string} exists in {save_dir}, skipping data writing"
            )
        else:
            Dataset.store_sharded_data(data, save_dir, num_shards)
            mark_done(save_dir, version_string=version_string)

    def ExportSrcTgtJSONLData(
        self,
        save_dir: str,
        num_shards: int = 1,
        version_string: str = DUMMY_VERSION,
        src_key: str = "input",  # for datasets loaded outside of ram.tasks, keys might be different
        tgt_key: str = "label",
        wrap_with_chat_template: bool = False,
        instr_start_symbol: str = "[INST] ",  # for llamas >2, other tokens are used
        instr_end_symbol: str = " [/INST]",
    ):
        # assuming src tgt fields exist in dataset examples
        exs = []
        for ex in self.flattened_iter():
            keys = ex.get_all_keys()
            if (src_key not in keys) or (tgt_key not in keys):
                raise RuntimeError(
                    f"dataset examples must have '{src_key}' and '{tgt_key}' fields prior to call this"
                )
            if wrap_with_chat_template:
                new_ex = {
                    "src": instr_start_symbol + ex[src_key] + instr_end_symbol,
                    "tgt": ex[tgt_key],
                }
            else:
                new_ex = {"src": ex[src_key], "tgt": ex[tgt_key]}
            exs.append(new_ex)
        self.check_built_and_store(exs, save_dir, num_shards, version_string)

    def ExportPreferenceTrainingData(
        self,
        save_dir: str,
        num_shards: int = 1,
        version_string: str = DUMMY_VERSION,
        src_key: str = "input",  # for datasets loaded outside of ram.tasks, keys might be different
        tgt_pos_key: str = "pos_label",
        tgt_neg_key: str = "neg_label",
        wrap_with_chat_template: bool = False,
        instr_start_symbol: str = "[INST] ",  # for llamas >2, other tokens are used
        instr_end_symbol: str = " [/INST]",
    ):
        # assuming src tgt fields exist in dataset examples
        exs = []
        for ex in self.flattened_iter():
            keys = ex.get_all_keys()
            if (
                (src_key not in keys)
                or (tgt_pos_key not in keys)
                or (tgt_neg_key not in keys)
            ):
                raise RuntimeError(
                    f"dataset examples must have '{src_key}', '{tgt_neg_key}' and '{tgt_pos_key}' fields prior to call this"
                )
            if wrap_with_chat_template:
                new_ex = {
                    "src": instr_start_symbol + ex[src_key] + instr_end_symbol,
                    "tgt_chosen": ex[tgt_pos_key],
                    "tgt_rejected": ex[tgt_neg_key],
                    "pairwise": True,
                }
            else:
                new_ex = {
                    "src": ex[src_key],
                    "tgt_chosen": ex[tgt_pos_key],
                    "tgt_rejected": ex[tgt_neg_key],
                    "pairwise": True,
                }
            exs.append(new_ex)
        self.check_built_and_store(exs, save_dir, num_shards, version_string)

    def ExportEvaluationData(
        self, save_dir: str, num_shards: int = 1, version_string: str = DUMMY_VERSION
    ):
        """
        Only writes out the system prompt + prompt, so that evaluation can be run.
        By default dummy version (0.0) string is provided.
        """

        exs = []
        for ex_group in self.data:
            for ex in ex_group:
                new_text = add_instruct_tokens(ex)
                new_ex = {"text": new_text}
                if ex.get("metadata", "") != "":
                    new_ex["metadata"] = ex["metadata"]
                exs.append(new_ex)
        self.check_built_and_store(exs, save_dir, num_shards, version_string)

    def ExportMessageData(
        self,
        save_dir: str,
        num_shards: int = 1,
        version_string: str = DUMMY_VERSION,
        wrap_with_chat_template: bool = False,
        key: str = "input",
    ):
        """
        Parse input in multi-turn Llama3-Instruct format
        """

        exs = []
        for ex_group in self.data:
            for ex in ex_group:
                if wrap_with_chat_template:
                    text = add_instruct_tokens(ex, key=key)
                else:
                    text = ex.get(key)
                if "<|eot_id|>" in text:
                    # split into dialog
                    dialog = []
                    messages = text.split("<|eot_id|>")
                    for message in messages:
                        source, body = message.split("<|end_header_id|>")
                        body = body.strip()
                        if len(body) > 0:
                            dialog.append(
                                {
                                    "source": source.split("<|start_header_id|>")[1],
                                    "body": body,
                                }
                            )
                    new_ex = {"dialog": dialog}
                else:
                    new_ex = {"dialog": [{"source": "user", "body": text}]}
                for other_key in ex.get_all_keys():
                    if other_key != key and ex.get(other_key, "") != "":
                        new_ex[other_key] = ex[other_key]
                exs.append(new_ex)
        self.check_built_and_store(exs, save_dir, num_shards, version_string)

    def ExportVLLMGenerationData(
        self,
        save_dir: str,
        generation_model_dir: str,
        strip_bos: bool = True,
        wrap_with_chat_template: bool = True,
        num_shards: int = 1,
        version_string: str = DUMMY_VERSION,
    ):
        """Exporting examples to jsonl wrapped by the generation model tokenizer to match the prompt template.

        Args:
            save_dir (str): where data chunks will be saved
            generation_model_dir (str): path to the HuggingFace model which contains the tokenizer inside
            strip_bos (bool): remove bos from the beginning if True. VLLM adds extra BOS in the beginning using its own tokenizer.
            This arg should be True at least for llama-like models. Default is True.
            wrap_with_chat_template (bool): if False, then your text data MUST provide your own special tokens such as [INST] or
            other depending on which model you use.
            num_shards (int, optional): how many jsonl shards to save. Defaults to 1.
            version_string (str, optional): version saved in the dump dir. Defaults to DUMMY_VERSION.
        """

        tokenizer = AutoTokenizer.from_pretrained(generation_model_dir)

        exs = []
        for ex_group in self.data:
            for ex in ex_group:
                if wrap_with_chat_template:
                    new_text = wrap_text_with_chat_template(
                        ex["input"], tokenizer=tokenizer
                    )
                    if strip_bos:
                        new_text = _strip_bos(
                            new_text, tokenizer.special_tokens_map["bos_token"]
                        )
                else:
                    new_text = ex["input"]

                new_ex = {"text": new_text}
                if ex.get("metadata", "") != "":
                    new_ex["metadata"] = ex["metadata"]
                exs.append(new_ex)
        if num_shards == 1 and save_dir.endswith(".jsonl"):
            if Path(save_dir).exists():
                logging.warning(f"Data exists in {save_dir}, skipping data writing")
            else:
                with jsonlines.open(save_dir, "w") as fo:
                    for ex in exs:
                        fo.write(ex)
            return
        self.check_built_and_store(exs, save_dir, num_shards, version_string)

    def ExportJSONLData(
        self,
        save_dir: str,
        num_shards: int = 1,
        version_string: str = DUMMY_VERSION,
    ):
        """Exports the ram dataset with its full structure.
        The resulting jsonl could be loaded back, see ImportJSONLData.

        Args:
            save_dir (str): where data chunks will be saved
            num_shards (int, optional): how many jsonl shards to save. Defaults to 1.
            version_string (str, optional): version saved in the dump dir. Defaults to DUMMY_VERSION.
        """
        # serialize dataset
        data = self.serialize()

        self.check_built_and_store(data, save_dir, num_shards, version_string)

    def ExportExamples(
        self, save_dir: str, num_shards: int = 1, version_string: str = DUMMY_VERSION
    ):
        """Export examples (without example groups) to sharded jsonls, keeping all fields

        Args:
            save_dir (str): where data chunks will be saved
            num_shards (int, optional): how many jsonl shards to save. Defaults to 1.
            version_string (str, optional): version saved in the dump dir. Defaults to DUMMY_VERSION.
        """
        data = [ex for ex in self.flattened_iter()]
        self.check_built_and_store(data, save_dir, num_shards, version_string)

    @staticmethod
    def ImportInstructionData(
        file_path,
        instr_start_symbol: str = "[INST]",  # for llamas >2, other tokens are used
        instr_end_symbol: str = "[/INST]",
    ):
        dataset = Dataset([])
        with open(file_path, "r") as file:
            for line in file:
                d = json.loads(line)
                original_txt = d["text"]
                parts = original_txt.replace(instr_start_symbol, "").split(
                    instr_end_symbol
                )
                inst = parts[0].strip()
                label = parts[1].strip() if len(parts) > 1 else ""
                example = Example()
                example.set_key("input", inst)
                example.set_key("label", label)
                dataset.add_example(example)
        return dataset

    @classmethod
    def ImportJSONLData(cls, load_dir: str):
        """Imports RAM dataset from serialized jsonl form.
        Intended usage: load serialized data dumped using ExportJSONLData

        Args:
            load_dir (str): where data chunks will be loaded from

        Returns:
            Dataset: resulting dataset
        """
        try:
            data = load_sharded_dir(load_dir)
        except:
            raise RuntimeError(f"Sharded data loading failed: {load_dir}")

        # deserialize data into Dataset object
        return cls.deserialize(data)

    @classmethod
    def ImportAnyJSONL(cls, load_dir_or_jsonl: str):
        """Loading any jsonl data by assuming there is no example groups structure in it

        Args:
            load_dir_or_jsonl (str): directory with jsonls or the jsonl path itself
        """
        if load_dir_or_jsonl.endswith("jsonl"):
            data = load_from_jsonl(load_dir_or_jsonl)
            for line in data:
                assert isinstance(
                    line, dict
                ), "In this case we assume every input line in jsonl must be a dict"
        elif Path(load_dir_or_jsonl).is_dir():
            # load the dir of jsonls
            jsonl_filenames = Path(load_dir_or_jsonl).glob("*.jsonl")
            data = []
            for jsonl_fname in jsonl_filenames:
                data.extend(load_from_jsonl(jsonl_fname))
        else:
            raise NotImplementedError

        dataset = Dataset([])
        for sample in data:
            dataset.add_example(Example(sample))

        return dataset
