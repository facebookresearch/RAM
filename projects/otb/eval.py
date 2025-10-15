"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import sys

import pandas as pd
from evals.math_eval import eval_math
from evals.overthink_eval import eval_overthink
from evals.token_eval import eval_tokens
from evals.underthink_eval import eval_underthink
from utils import format_metrics, model_to_tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--model", required=True)
    parser.add_argument("--t_max", default=100, type=int)
    args = parser.parse_args()

    input_file = args.input_file
    model_name = args.model
    tokenizer = model_to_tokenizer(model_name)
    t_max = args.t_max

    df = pd.read_json(input_file, lines=True)

    all_rows = []
    for _, row in df.iterrows():
        subset = row.get("subset", "")
        if subset == "overthinking-bench" or "overthinking" in subset:
            acc = eval_overthink(row)
        elif subset == "underthinking-bench" or "underthinking" in subset:
            acc = eval_underthink(row)
        elif "math" in subset:
            acc = eval_math(row, tokenizer, model_name)
        else:
            acc = eval_math(row, tokenizer, model_name)

        tok = eval_tokens(row, tokenizer, model_name)
        if "overthink" in subset:
            score = max(0, 1000 - tok) / 1000 * acc
        else:
            score = acc

        all_rows.append(
            {"accuracy": acc, "tokens": tok, "score": score, "subset": subset}
        )

    out_df = pd.DataFrame(all_rows)
    print(out_df.groupby("subset").mean(numeric_only=True))
    print(
        "Report Macro Average between overthink splits and underthink splits, followed by F1 between the two."
    )
    (
        overthink_macro_acc,
        overthink_macro_score,
        overthink_macro_tokens,
        underthink_macro_acc,
        underthink_macro_score,
        underthink_macro_tokens,
        f1,
    ) = format_metrics(out_df)
    print(
        f"Overthink Macro Accuracy: {overthink_macro_acc}, Overthink Macro Score: {overthink_macro_score}, Overthink Macro Tokens: {overthink_macro_tokens}"
    )
    print(
        f"Underthink Macro Accuracy: {underthink_macro_acc}, Underthink Macro Score: {underthink_macro_score}, Underthink Macro Tokens: {underthink_macro_tokens}"
    )
    print(f"Overthink F1: {f1}")


if __name__ == "__main__":
    main()
