"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
from logging import getLogger
from typing import Dict, Iterator, List, Optional

import numpy as np

logger = getLogger()

from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class Batch:
    x: np.ndarray
    y: np.ndarray
    mask: Optional[np.ndarray] = None
    src_names: Optional[List[str]] = None

    def __post_init__(self):
        assert self.x.ndim == 2
        assert self.x.shape == self.y.shape
        assert self.x.dtype == np.int64
        assert self.y.dtype == np.int64
        assert self.mask is None or self.mask.shape == self.x.shape
        assert self.src_names is None or len(self.src_names) == len(self.x)


class DataIterator:
    @abstractmethod
    def __iter__(self) -> Iterator[Batch]: ...

    @abstractmethod
    def get_position(self) -> Optional[List[int]]: ...

    @abstractmethod
    def set_position(self, position: Optional[List[int]]): ...

    def close(self):
        pass


def get_content_key(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip()
        try:
            x = json.loads(line)
        except UnicodeDecodeError as e:
            print(f"Error when trying to decode '{line}': {str(e)}")
            raise
        for k in ["text", "content", "src"]:
            if k in x:
                return k
        raise RuntimeError(f"Unable to determine key for {path}")


class JSONLIterator:
    def __init__(
        self,
        fpath: str,
        world_size: int,
        world_rank: int,
        infinite: bool,
    ):
        assert 0 <= world_rank < world_size, (world_rank, world_size)
        self.f = open(fpath, "r", encoding="utf-8")
        self.fpath = fpath
        self.world_size = world_size
        self.world_rank = world_rank
        self.line_num = 0
        self.iter = iter(self.gen(infinite))
        self.iter_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.iter)

    def gen(self, infinite: bool) -> Iterator[Dict]:
        while True:
            logger.info(f"Starting iteration {self.iter_id} over {self.fpath} ...")
            self.iter_id += 1
            while True:
                line, self.line_num = self.f.readline(), self.line_num + 1
                if not line:
                    break
                if (self.line_num - 1) % self.world_size == self.world_rank:
                    yield json.loads(line)
            if not infinite:
                break
            self.set_position(None)
        self.f.close()


def batch_iterator(
    jsonl_iterator: JSONLIterator,
    tokenizer,
    seq_len: int,
    batch_size: int,
    buffer_size: int,
) -> Iterator[Batch]:
    """
    Take as input a JSONLIterator and return an iterator of batches.
    """
    content_key = get_content_key(jsonl_iterator.fpath)
    n_buffer_toks = (1 + buffer_size * seq_len) * batch_size
    tokens: List[int] = []
    mask: List[bool] = []
    for sample in jsonl_iterator:
        assert len(tokens) < n_buffer_toks
        if content_key == "src":
            src_toks = tokenizer.encode(
                sample[content_key], add_bos=True, add_eos=False
            )
            tokens.extend(src_toks)
            mask.extend([False] * len(src_toks))
            tgt_toks = tokenizer.encode(sample["tgt"], add_bos=False, add_eos=True)
            tokens.extend(tgt_toks)
            mask.extend([True] * len(tgt_toks))
        else:
            toks = tokenizer.encode(sample[content_key], add_bos=True, add_eos=True)
            tokens.extend(toks)
            mask.extend([True] * len(toks))
        while len(tokens) >= n_buffer_toks:
            x = np.array(tokens[:n_buffer_toks]).reshape(batch_size, -1)
            x_mask = np.array(mask[:n_buffer_toks]).reshape(batch_size, -1)
            tokens = tokens[n_buffer_toks:]
            mask = mask[n_buffer_toks:]
            assert x.shape[1] == 1 + buffer_size * seq_len
            assert x.shape[1] // seq_len == buffer_size
            for i in range(x.shape[1] // seq_len):
                a, b = i * seq_len, (i + 1) * seq_len
                yield Batch(
                    x=x[:, a:b], y=x[:, a + 1 : b + 1], mask=x_mask[:, a + 1 : b + 1]
                )


def subsample(
    iterator: Iterator[Batch], batch_size: int, world_rank: int, world_size: int
) -> Iterator[Batch]:
    assert 0 <= world_rank < world_size
    for batch in iterator:
        assert batch.x.shape[0] == world_size * batch_size
        a = batch_size * world_rank
        b = batch_size * world_rank + batch_size
        yield Batch(x=batch.x[a:b], y=batch.y[a:b])


class SingleJSONLIterator(DataIterator):
    """
    Iterator that sequentially iterate over a single .jsonl file.
    Loop indefinitely over the file if `infinite=True`.
    """

    def __init__(
        self,
        tokenizer,
        data_path: str,
        seq_len: int,
        batch_size: int,
        buffer_size: int,
        world_rank: int,
        world_size: int,
        infinite: bool,
        same_data: bool = False,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.world_rank = world_rank
        self.world_size = world_size
        self.infinite = infinite
        self.same_data = same_data
        assert 0 <= world_rank < world_size

        logger.info(f"Starting iteration on {data_path} ...")

        # jsonl iterator
        self.jsonl_iterator = JSONLIterator(
            fpath=data_path,
            world_rank=world_rank,
            world_size=world_size,
            infinite=self.infinite,
        )

    def __iter__(self) -> Iterator[Batch]:
        if self.same_data:
            iterator = batch_iterator(
                jsonl_iterator=self.jsonl_iterator,
                tokenizer=self.tokenizer,
                seq_len=self.seq_len,
                batch_size=self.world_size * self.batch_size,
                buffer_size=1,
            )
            return subsample(
                iterator, self.batch_size, self.world_rank, self.world_size
            )
        return batch_iterator(
            jsonl_iterator=self.jsonl_iterator,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
        )
