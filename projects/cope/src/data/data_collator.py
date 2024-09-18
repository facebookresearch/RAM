"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch

from .constants import (
    ANSWER_KEY,
    CONTEXT_KEY,
    QUESTION_KEY,
    TOKEN_TYPE_ANSWER,
    TOKEN_TYPE_CONTEXT,
    TOKEN_TYPE_PAD,
    TOKEN_TYPE_QUESTION,
)


class DataCollatorForDecoder:
    def __init__(self, cfg, tokenizer, fixed_len: int, training=True):
        self.cfg = cfg
        self.tokenizer = tokenizer
        # Training argument is not put in use right now
        self.training = training
        self.fixed_len = fixed_len

    def pad_right(self, enc_seqs, enc_seqs_type):
        # Right padding for batching

        if self.fixed_len > 0:
            max_len = self.fixed_len + 1
        else:
            max_len = max([len(enc_seq) for enc_seq in enc_seqs])

        # trim from left for longer sequences
        max_len = min(max_len, self.cfg.block_size)
        enc_seqs = [x[-max_len:] for x in enc_seqs]
        enc_seqs_type = [x[-max_len:] for x in enc_seqs_type]

        enc_seqs = [
            (enc_seq + [self.tokenizer.pad_ind] * (max_len - len(enc_seq)))
            for enc_seq in enc_seqs
        ]
        enc_seqs_type = [
            (enc_seq_type + [TOKEN_TYPE_PAD] * (max_len - len(enc_seq_type)))
            for enc_seq_type in enc_seqs_type
        ]
        enc_seqs = torch.LongTensor(enc_seqs)
        enc_seqs_type = torch.LongTensor(enc_seqs_type)

        return enc_seqs, enc_seqs_type

    def __call__(self, instances):
        enc_seqs = [[] for _ in range(len(instances))]
        enc_seqs_type = [[] for _ in range(len(instances))]
        for i, instance in enumerate(instances):
            for sample_key, token_type in zip(
                [
                    CONTEXT_KEY,
                    QUESTION_KEY,
                    ANSWER_KEY,
                ],
                [
                    TOKEN_TYPE_CONTEXT,
                    TOKEN_TYPE_QUESTION,
                    TOKEN_TYPE_ANSWER,
                ],
            ):
                if sample_key in instance and instance[sample_key] != "":
                    # encode separately so we know which token belongs to what
                    add_bos = len(enc_seqs[i]) == 0  # add bos at the start only
                    enc_key = self.tokenizer.encode(
                        instance[sample_key], add_bos=add_bos
                    )
                    enc_seqs[i] += enc_key
                    enc_seqs_type[i] += [token_type] * len(enc_key)

        enc_seqs, enc_seqs_type = self.pad_right(enc_seqs, enc_seqs_type)

        # remove answer from x
        dec_x = enc_seqs[:, :-1]
        dec_x_type = enc_seqs_type[:, :-1]

        # shift y right by 1
        dec_y = enc_seqs[:, 1:]
        dec_y_type = enc_seqs_type[:, 1:]

        out = {
            "dec_x": dec_x,
            "dec_y": dec_y,
            "dec_x_type": dec_x_type,
            "dec_y_type": dec_y_type,
        }
        return out
