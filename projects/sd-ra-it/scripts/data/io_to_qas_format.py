"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import json
import os
import sys

from tqdm import tqdm

infilename = sys.argv[1]
outfilename = sys.argv[2]
with open(infilename) as infile, open(outfilename, "w") as outfile:
    for line in tqdm(map(json.loads, infile)):
        question = line["input"]
        answers = [
            output.get("answer")
            for output in line["output"]
            if output.get("answer") is not None
        ]
        output_line = line | dict(question=question, answers=answers)
        print(json.dumps(output_line), file=outfile)
