"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import List, Optional, Tuple

from lingua import tokenizer


class CharcterTokenizer(tokenizer.Tokenizer):
    def __init__(self):
        self.mapping = self.create_alphabet(
            vocab_size=26,
            block_separator=".",
            query_separator="#",
        )

        self.bos_id = len(self.mapping)
        self.mapping["<BOS>"] = self.bos_id
        self.eos_id = len(self.mapping)
        self.mapping["<EOS>"] = self.eos_id
        i = 0
        while len(self.mapping) % 8 != 0:
            self.mapping[f"<|reserved_special_token_{i}|>"] = len(self.mapping)
            i += 1
        self.n_words = len(self.mapping)

        self.reversed_mapping = {v: k for k, v in self.mapping.items()}

    def create_alphabet(
        self, vocab_size: int, block_separator: str, query_separator: str
    ):
        alphabet = {}
        i = 0
        j = 0
        while len(alphabet) < vocab_size:
            new_char = chr(ord("a") + i)
            if new_char not in [block_separator, query_separator]:
                assert new_char not in alphabet.keys()
                alphabet[new_char] = j
                j += 1
            i += 1
        alphabet[block_separator] = j
        alphabet[query_separator] = j + 1
        alphabet["_"] = j + 2  # normally should be an argument
        return alphabet

    def encode(self, s: str, add_bos: bool, add_eos: bool):
        assert type(s) is str, s
        assert " " not in s, s
        t = [self.mapping[char] for char in s]
        if add_bos:
            t.insert(0, self.bos_id)
        if add_eos:
            t.append(self.eos_id)
        return t

    def decode(self, tokens: List[int]):
        return "".join([self.reversed_mapping[tok_id] for tok_id in tokens])

    def get_token_offsets(
        self, text: str, tokens: Optional[List[int]] = None
    ) -> Tuple[List[str], List[int]]:
        if tokens is None:
            tokens = self.encode(text, False, False)

        decoded_chars, offsets = [], []
        char_pos = 0
        for token in tokens:
            if token < self.bos_id:
                char = self.reversed_mapping[token]
                decoded_chars.append(char)
                offsets.append(char_pos)
                char_pos += len(char)

        return decoded_chars, offsets


def build_tokenizer(name: str, path: Optional[str] = None) -> tokenizer.Tokenizer:
    if name == "char":
        return CharcterTokenizer()
    else:
        return tokenizer.build_tokenizer(name, path)
