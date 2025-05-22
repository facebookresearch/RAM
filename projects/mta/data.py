"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Iterator, Optional, TypedDict

import numpy as np
from lingua.data import (
    TRAIN_DATA_FILE_PATTERN,
    PackTokensState,
    PrefetchState,
    async_iterator,
    choose_source,
    get_empty_buffer_state,
    init_choice_state,
    setup_sources,
)
from lingua.tokenizer import TokenizerArgs

from projects.mta.tokenizer import build_tokenizer

logger = logging.getLogger()

"""
Here we modify loaders to add mask for src/tgt data files
"""


class TokenizerState(TypedDict):
    it_state: Any
    name: str
    add_bos: bool
    add_eos: bool
    path: Optional[str]
    src_tgt_format: Optional[bool]
    no_loss_prompt: Optional[bool]  # for src_tgt_format
    src_tgt_sep: Optional[str]  # for src_tgt_format


# modified
def tokenize(
    iterator: Iterator,
    add_bos: bool,
    add_eos: bool,
    tokenizer_type: str,
    tokenizer_path: Optional[str] = None,
    src_tgt_format: bool = False,
    no_loss_prompt: bool = True,  # for src_tgt_format
    src_tgt_sep: str = "",  # for src_tgt_format
):
    """
    Tokenizes text from an iterator of content-state pairs using a specified tokenizer.

    Parameters:
    - iterator: An iterable of (content, state) pairs where content is a dict with a 'text' or 'content' key.
    - tokenizer: Tokenizer object with an `encode` method to convert text to tokens, supporting `add_bos` and `add_eos`.
    - add_bos (bool): Flag to add a beginning-of-sequence token.
    - add_eos (bool): Flag to add an end-of-sequence token.

    Yields:
    - (tokens, state) pairs, where `tokens` is a list of tokenized text, and `state` is the original state from the iterator.
    """
    tokenizer = build_tokenizer(name=tokenizer_type, path=tokenizer_path)
    for content, state in iterator:
        if src_tgt_format and "src" in content and "tgt" in content:
            text = content["src"] + src_tgt_sep + content["tgt"]
            tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)
            # now we need to find where answer starts
            toks_tgt = tokenizer.encode(content["tgt"], add_bos=False, add_eos=add_eos)
            toks_src = tokens[: -len(toks_tgt)]
            valid_ans = [False] * len(toks_src) + [True] * len(toks_tgt)
        else:
            assert (
                "text" in content or "content" in content
            ), "JSON line must contain either text or content key"
            content_key = "text" if ("text" in content) else "content"
            text = content[content_key]
            tokens = tokenizer.encode(text, add_bos=add_bos, add_eos=add_eos)

        mask = np.ones((len(tokens),), dtype=bool)

        if no_loss_prompt and src_tgt_format:
            mask = mask & np.array(valid_ans, dtype=bool)

        yield tokens, mask, TokenizerState(
            it_state=state,
            add_bos=add_bos,
            add_eos=add_eos,
            name=tokenizer_type,
            path=tokenizer_path,
        )


def pack_tokens(
    iterator: Iterator,
    empty_buffer_state: PackTokensState,
):
    """
    Iterates over tokens, packing them into chunks.

    This function aggregates tokens into a buffer and yields fixed-size chunks with dimensions `(output_seq_len, n_views)`,
    where each column represents shifted sequences of tokens. It ensures continuity in token sequences across chunks,
    preventing boundary effects and maintaining consistency regardless of `n_views`.

    Parameters:
    - iterator: An iterator that yields pairs of (tokens, state), where tokens is a 1D sequence of tokens and state contains all necessary information to resume iterator from current position.
    - it_state: State of the iterator currently.
    - start_token (int): The index of the first token to start reading from for the first sequence.
    - output_seq_len (int): The length of the output sequences to be generated.
    - n_views (int): The number of shifted views to include in each output chunk.

    Yields:
    - numpy.ndarray: An array of shape `(output_seq_len, n_views)` containing the packed tokens.
    - PackTokensState: The state required to resume packing tokens from where the last returned chunk.

    The function handles the complexity of determining the correct state for resuming iteration after the buffer is cleared, ensuring seamless continuation of token sequences.
    """
    buffer = []
    mask_buffer = []
    states = []
    output_seq_len = empty_buffer_state["output_seq_len"]
    n_views = empty_buffer_state["n_views"]
    start_token = empty_buffer_state["start_token"]
    previous_state = empty_buffer_state["it_state"]
    buffer_size = output_seq_len + n_views - 1
    for i, (tokens, mask, state) in enumerate(iterator):
        end_token = start_token
        sample_is_read = False
        while not sample_is_read:
            assert start_token < len(
                tokens
            ), f"Start token index {start_token} bigger than sequence {len(tokens)}"
            assert len(buffer) == len(mask_buffer)
            free_space = buffer_size - len(buffer)
            seq_len = min(free_space, len(tokens) - start_token)
            end_token = start_token + seq_len
            buffer.extend(tokens[start_token:end_token])
            mask_buffer.extend(mask[start_token:end_token])
            start_token = end_token

            states.append(
                PackTokensState(
                    start_token=start_token,
                    seq_len=seq_len,
                    it_state=previous_state,
                    output_seq_len=output_seq_len,
                    n_views=n_views,
                )
            )
            assert len(buffer) <= buffer_size, "Buffer overflow"
            assert len(mask_buffer) <= buffer_size, "Buffer overflow"

            if len(buffer) == buffer_size:
                out = np.array(buffer)
                mask_out = np.array(mask_buffer)
                assert out.ndim == 1, "Iterator should return 1D sequences"
                assert mask_out.ndim == 1, "Iterator should return 1D sequences"
                out = np.lib.stride_tricks.sliding_window_view(
                    out, n_views, axis=0
                )  # (output_seq_len, n_views)
                mask_out = mask_out[
                    n_views - 1 :
                ]  # (output_seq_len), mask on labels only

                # We rewind by n_views to account for the last tokens not having their targets
                rewinded_idx = start_token - (n_views - 1)
                empty_buffer_state = get_empty_buffer_state(rewinded_idx, states)
                buffer = buffer[output_seq_len:]
                assert len(buffer) == (n_views - 1)
                mask_buffer = mask_buffer[output_seq_len:]
                assert len(mask_buffer) == (n_views - 1)

                yield out, mask_out, empty_buffer_state

            if start_token == len(tokens):
                start_token = 0
                sample_is_read = True
                previous_state = state


def batch_and_shuffle_prefetched_sequences(
    data_loader: Iterator,
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    state: PrefetchState,
):
    """
    Prepare batch in advance and shuffle them to reduce correlation inside batches (for ex when very long document is encountered).

    This function aggregates batches into a buffer and yields fixed-size batch size and seqlen with dimensions `(batch_size, seqlen, n_views)`,

    It uses a prefetch buffer to store batches in advance and shuffles them, the prefetch buffer is similar to `reservoir sampling`,
    but by block to preserve a smooth, easy and deterministic reloading. To ensure more uniform sequence sampling -> prefetch_size * batch_size * seq_len >> max_document_seqlength.

    Parameters:
    - iterator: An iterator that yields pairs of (sequence, state), where is a random sequence sampled from a corpus (as done by pack_tokens for example).
    - batch_size: The desired batch size.
    - prefetch_size: The number of batches to prefetch in advance.
    - seq_len (int): The length of the output sequences to be generated.
    - n_views (int): The number of shifted views to include in each output chunk.

    Yields:
    - numpy.ndarray: An array of shape `(batch_size, seq_len, n_views)` containing the packed tokens.
    - PrefetchState: The state required to resume prefetched batch. Contains also the internal of iterator.
    """
    prefetch_buffer = -1 * np.ones(
        (prefetch_size * batch_size, seq_len, n_views), dtype=int
    )
    prefetch_mask_buffer = -1 * np.ones(
        (prefetch_size * batch_size, seq_len), dtype=int
    )
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng_state"]

    # Rewind the iterator to the correct position by skipping seq_idx sequences to roll the buffer accordingly
    seq_idx = state["seq_idx"]
    assert (
        seq_idx >= 0 and seq_idx < prefetch_size
    ), "Prefetch state seq_idx should be in 0 <= seq_idx < prefetch_size."

    _rng_state = state["rng_state"]
    _it_state = state["it_state"]

    for i in range(prefetch_size * batch_size):
        prefetch_buffer[i], prefetch_mask_buffer[i], next_it_state = next(data_loader)

    # shuffling two tensors in the same order
    # rng.shuffle(prefetch_buffer, axis=0)
    indices = rng.permutation(len(prefetch_buffer))
    prefetch_buffer = prefetch_buffer[indices]
    prefetch_mask_buffer = prefetch_mask_buffer[indices]

    for i in range(seq_idx * batch_size):
        prefetch_buffer[i], prefetch_mask_buffer[i], _ = next(data_loader)

    idx = seq_idx
    while True:
        if idx == prefetch_size - 1:
            _it_state = next_it_state
            _rng_state = rng.bit_generator.state

        state = PrefetchState(
            it_state=_it_state,
            seq_idx=(idx + 1) % prefetch_size,
            rng_state=_rng_state,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
        )

        yield prefetch_buffer[
            idx * batch_size : (idx + 1) * batch_size
        ].copy(), prefetch_mask_buffer[
            idx * batch_size : (idx + 1) * batch_size
        ].copy(), state

        for i in range(batch_size):
            (
                prefetch_buffer[idx * batch_size + i],
                prefetch_mask_buffer[idx * batch_size + i],
                pack_state,
            ) = next(data_loader)

        if idx == prefetch_size - 1:
            next_it_state = pack_state
            indices = rng.permutation(len(prefetch_buffer))
            prefetch_buffer = prefetch_buffer[indices]
            prefetch_mask_buffer = prefetch_mask_buffer[indices]

        idx = (idx + 1) % prefetch_size


def init_state(
    root_dir: str,
    sources: Dict[str, float],
    batch_size: int,
    prefetch_size: int,
    seq_len: int,
    n_views: int,
    seed: int,
    rank: int,
    world_size: int,
    add_bos: bool,
    add_eos: bool,
    tokenizer_name: str,
    tokenizer_path: Optional[str] = None,
    file_pattern: str = TRAIN_DATA_FILE_PATTERN,
    src_tgt_format: bool = False,
    no_loss_prompt: bool = True,  # for src_tgt_format
    src_tgt_sep: str = "",  # for src_tgt_format
):
    multi_choice_state = init_choice_state(
        root_dir=root_dir,
        sources=sources,
        seed=seed,
        rank=rank,
        world_size=world_size,
        file_pattern=file_pattern,
    )
    tokenizer_state = TokenizerState(
        it_state=multi_choice_state,
        add_bos=add_bos,
        add_eos=add_eos,
        name=tokenizer_name,
        path=tokenizer_path,
        src_tgt_format=src_tgt_format,
        no_loss_prompt=no_loss_prompt,
        src_tgt_sep=src_tgt_sep,
    )
    pack_state = PackTokensState(
        start_token=0,
        it_state=tokenizer_state,
        output_seq_len=seq_len,
        n_views=n_views,
        seq_len=0,
    )

    prefetch_rng_state = np.random.default_rng(
        (seed + 1, rank, world_size)
    ).bit_generator.state

    return PrefetchState(
        it_state=pack_state,
        seq_idx=0,
        rng_state=prefetch_rng_state,
        batch_size=batch_size,
        prefetch_size=prefetch_size,
    )


@contextlib.contextmanager
def build_dataloader(
    state: PrefetchState,
):
    pack_state = state["it_state"]
    tokenizer_state = pack_state["it_state"]
    multi_state = tokenizer_state["it_state"]

    path_to_iter = setup_sources(multi_state)
    data_it = choose_source(
        source_to_iterator=path_to_iter,
        source_to_state=multi_state["source_to_state"],
        root_dir=multi_state["root_dir"],
        sources=multi_state["sources"],
        rng_state=multi_state["rng_state"],
    )
    data_it = tokenize(
        data_it,
        tokenizer_state["add_bos"],
        tokenizer_state["add_eos"],
        tokenizer_state["name"],
        tokenizer_state["path"],
        tokenizer_state["src_tgt_format"],
        tokenizer_state["no_loss_prompt"],
        tokenizer_state["src_tgt_sep"],
    )

    data_it = pack_tokens(
        data_it,
        pack_state,
    )

    data_it = batch_and_shuffle_prefetched_sequences(
        data_loader=data_it,
        seq_len=pack_state["output_seq_len"],
        n_views=pack_state["n_views"],
        batch_size=state["batch_size"],
        prefetch_size=state["prefetch_size"],
        state=state,
    )
    yield data_it
    for it in path_to_iter.values():
        it.close()
    data_it.close()


@dataclass
class DataArgs:
    root_dir: Optional[str] = None
    sources: Dict[str, float] = field(default_factory=dict)
    batch_size: int = 2
    seq_len: int = 2048
    n_views: int = 2
    seed: int = 42
    add_bos: bool = True
    add_eos: bool = True
    load_async: bool = True
    prefetch_size: int = 64
    tokenizer: TokenizerArgs = field(default_factory=TokenizerArgs)
    src_tgt_format: bool = False
    no_loss_prompt: bool = True  # for src_tgt_format
    src_tgt_sep: str = ""  # for src_tgt_format


def init_dataloader_state_from_args(
    args: DataArgs,
    rank: int,
    world_size: int,
):
    return init_state(
        root_dir=args.root_dir,
        sources=args.sources,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        prefetch_size=args.prefetch_size,
        n_views=args.n_views,
        seed=args.seed,
        rank=rank,
        world_size=world_size,
        tokenizer_name=args.tokenizer.name,
        tokenizer_path=args.tokenizer.path,
        add_bos=args.add_bos,
        add_eos=args.add_eos,
        src_tgt_format=args.src_tgt_format,
        no_loss_prompt=args.no_loss_prompt,  # for src_tgt_format
        src_tgt_sep=args.src_tgt_sep,  # for src_tgt_format
    )


# do not remove
def build_dataloader_from_args(
    args: DataArgs,
    state: Optional[PrefetchState] = None,
):
    data_builder = partial(build_dataloader, state)
    if args.load_async:
        return async_iterator(args.prefetch_size, data_builder)
    else:
        return data_builder()
