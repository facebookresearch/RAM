"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


def eval_tokens(row, tokenizer, model_name):
    total_think_tokens = []
    responses = (
        row["response"] if isinstance(row["response"], list) else [row["response"]]
    )
    for res in responses:
        total_think_tokens.append(len(tokenizer(res.split("</think>")[0])["input_ids"]))
    total_think_tokens = sum(total_think_tokens) / len(total_think_tokens)

    if model_name.find("o3") != -1:
        # Prefer tokens from file if available
        tokens = row.get("tokens", None)
        if tokens is None:
            return total_think_tokens
        try:
            total_think_tokens = tokens / len(responses)
        except Exception:
            total_think_tokens = total_think_tokens
    return total_think_tokens
