"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import math
import random

import fire

from ram.data import Dataset, Example

""" In this task, Input is a sequence of random tokens split into blocks separated by “.”. Eg: “d k l s r . g p a m . r m z x …”
Now given several tokens in a query (e.g. “g and a”) - K < N, we need to find the block that contains both of them, 
then output that block. This is hard because the keys need to contain all tokens in the block, 
so it can match to any pair of them.
"""


def synthetize_data(
    block_length: int,
    query_legth: int,
    tot_samples: int,
    vocab_size: int,
    block_separator: str,
    query_separator: str,
    pad_len: int,
):

    alphabet = set()
    i = 0
    while len(alphabet) < vocab_size:
        new_char = chr(ord("a") + i)
        if new_char not in [block_separator, query_separator]:
            alphabet.add(new_char)
        i += 1
    alphabet = sorted(alphabet)

    input_query_output = {}

    while len(input_query_output) < tot_samples:
        # adversarial generations, from query
        target_block_list = random.sample(alphabet, block_length)
        query_list = random.sample(target_block_list, query_legth)

        target_block = "".join(target_block_list)
        query = "".join(query_list)

        # generate other blocks
        blocks = {target_block}
        number_of_blocks = math.ceil(
            (pad_len - len(query) - len(query_separator))
            / (block_length + len(block_separator))
        )
        while len(blocks) < number_of_blocks:
            sample = random.sample(alphabet, block_length)
            if not set(query_list).issubset(set(sample)):
                new_block = "".join(sample)
                blocks.add(new_block)

        # need to shuffle, converting to list
        blocks = list(blocks)
        random.shuffle(blocks)
        blocks = block_separator.join(blocks)[
            : pad_len - len(query) - len(query_separator)
        ]

        input_query_output[blocks] = (query, target_block)

        if len(input_query_output) % 10000 == 0:
            print(f"Collected {len(input_query_output)} points")

    return input_query_output


def main(
    output_dir: str,
    block_length: int,
    query_legth: int,
    train_samples: int,
    test_samples: int = 1000,
    vocab_size: int = 26,
    block_separator: str = ".",
    query_separator: str = "#",
    pad_len: int = 319,  # make all samples the same length so that each token sequence contains exactly one sample and lingua doesn't pack them
):
    """
    Parameters:
     block_length: total number of chars in the input
     number_of_blocks: total number of blocks in the input
     query_legth: number of chars in the query (should be no more than block_length)
     train_samples: number of train samples
     test_samples: number of test samples
     vocab_size: total number of chars to choose from, starting from 'a'
     block_separator: symbol that will separate/join blocks
     query_separator: symbol that will separate/join input and query
    """
    assert block_length >= query_legth
    assert block_length <= vocab_size

    data = synthetize_data(
        block_length,
        query_legth,
        tot_samples=train_samples + test_samples,
        vocab_size=vocab_size,
        block_separator=block_separator,
        query_separator=query_separator,
        pad_len=pad_len - block_length,  # minus target block length
    )

    train = Dataset([])
    test = Dataset([])

    for input, (query, target_block) in data.items():
        new_example = Example(
            {
                "input": input + query_separator + query,
                "label": target_block,
            }
        )
        slen = len(new_example["input"] + new_example["label"])
        assert slen == 319, new_example
        if len(test) < test_samples:
            test.add_example(new_example)
        else:
            train.add_example(new_example)

    print(f"Seq len: {slen}")

    train.ExportSrcTgtJSONLData(
        f"{output_dir}/train_{block_length}_{pad_len}pad_{query_legth}_{train_samples}",
        num_shards=1,
        wrap_with_chat_template=False,
    )
    test.ExportSrcTgtJSONLData(
        f"{output_dir}/test_{block_length}_{pad_len}pad_{query_legth}_{train_samples}",
        num_shards=1,
        wrap_with_chat_template=False,
    )


if __name__ == "__main__":
    fire.Fire(main)
