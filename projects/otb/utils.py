"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
import random
import time

import openai
from litellm import completion
from openai import AzureOpenAI
from transformers import AutoTokenizer

MAGISTRAL_PROMPT = """
A user will ask you to solve a task. You should first draft your thinking process (inner monologue) until you have derived the final answer. Afterwards, write a self-contained summary of your thoughts (i.e. your summary should be succinct but contain all the critical steps you needed to reach the conclusion). You should use Markdown to format your response. Write both your thoughts and summary in the same language as the task posed by the user. NEVER use \boxed{} in your response.

Your thinking process must follow the template below:
<think>
Your thoughts or/and draft, like working through an exercise on scratch paper. Be as casual and as long as you want until you are confident to generate a correct answer.
</think>

Here, provide a concise summary that reflects your reasoning and presents a clear final answer to the user. Don't mention that this is a summary.

Problem:
"""


def get_client(model_name):
    if model_name.find("o3") != -1 or model_name.find("gpt") != -1:
        return openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    else:
        return openai.OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:30000/v1",
        )


def call_llm(
    model_name, prompt, temperature=0.6, max_tokens=8192, n=1, is_confidence_based=False
):

    if model_name == "o3-med":
        model_name = "o3"
        max_tokens = 16384
        kwargs = {"reasoning_effort": "medium"}
    elif model_name == "o3-high":
        model_name = "o3"
        max_tokens = 32768
        kwargs = {"reasoning_effort": "high"}
    elif model_name == "o3":
        kwargs = {"reasoning_effort": "low"}
    else:
        kwargs = {}

    client = get_client(model_name)
    messages = [{"role": "user", "content": prompt}]
    if model_name.lower().find("magistral") != -1:
        messages = [
            {"role": "system", "content": MAGISTRAL_PROMPT},
            {"role": "user", "content": prompt},
        ]
    response = None
    for attempt in range(20):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature if model_name != "o3" else 1,
                max_completion_tokens=max_tokens,
                n=n,
                **kwargs,
            )
            break
        except Exception as e:
            print(f"Error calling {model_name}: {e}")
            time.sleep((attempt**2) * 5)

    return response


def model_to_tokenizer(model_name):
    if model_name == "o3" or model_name == "o3-med" or model_name == "o3-high":
        # Load a dummy tokenizer for o3 models
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def format_metrics(out_df):
    overthink_general_acc = out_df[out_df["subset"].str.contains("overthinking")][
        "accuracy"
    ].mean()
    overthink_math_acc = out_df[out_df["subset"].str.contains("overthinking-math")][
        "accuracy"
    ].mean()
    overthink_general_tokens = out_df[out_df["subset"].str.contains("overthinking")][
        "tokens"
    ].mean()
    overthink_math_tokens = out_df[out_df["subset"].str.contains("overthinking-math")][
        "tokens"
    ].mean()
    overthink_general_score = out_df[out_df["subset"].str.contains("overthinking")][
        "score"
    ].mean()
    overthink_math_score = out_df[out_df["subset"].str.contains("overthinking-math")][
        "score"
    ].mean()
    underthink_general_acc = out_df[out_df["subset"].str.contains("underthinking")][
        "accuracy"
    ].mean()
    underthink_math_acc = out_df[out_df["subset"].str.contains("underthinking-math")][
        "accuracy"
    ].mean()
    underthink_general_tokens = out_df[out_df["subset"].str.contains("underthinking")][
        "tokens"
    ].mean()
    underthink_math_tokens = out_df[
        out_df["subset"].str.contains("underthinking-math")
    ]["tokens"].mean()
    underthink_general_score = out_df[out_df["subset"].str.contains("underthinking")][
        "score"
    ].mean()
    underthink_math_score = out_df[out_df["subset"].str.contains("underthinking-math")][
        "score"
    ].mean()
    overthink_macro_acc = (overthink_general_acc + overthink_math_acc) / 2
    overthink_macro_score = (overthink_general_score + overthink_math_score) / 2
    overthink_macro_tokens = (overthink_general_tokens + overthink_math_tokens) / 2
    underthink_macro_acc = (underthink_general_acc + underthink_math_acc) / 2
    underthink_macro_score = (underthink_general_score + underthink_math_score) / 2
    underthink_macro_tokens = (underthink_general_tokens + underthink_math_tokens) / 2
    f1 = (
        2
        * overthink_macro_score
        * underthink_macro_score
        / (overthink_macro_score + underthink_macro_score)
    )
    return (
        overthink_macro_acc,
        overthink_macro_score,
        overthink_macro_tokens,
        underthink_macro_acc,
        underthink_macro_score,
        underthink_macro_tokens,
        f1,
    )
