import contextlib
import io
import json
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_math_splits_use_math_evaluator(self):
        subsets = ["underthinking-math", "overthinking-math", "underthinking-bench", "overthinking-bench"]
        rows = pd.DataFrame([
            {"question": "1+1?", "answer": "2", "response": ["2"],
             "metadata": json.dumps({"source": "aime", "id": "example"}), "subset": subset}
            for subset in subsets
        ])
        calls = []

        def evaluator(name):
            def evaluate(row, *args):
                calls.append((row["subset"], name))
                return 1.0
            return evaluate

        modules = {
            "evals": types.ModuleType("evals"),
            "evals.math_eval": types.SimpleNamespace(eval_math=evaluator("math")),
            "evals.overthink_eval": types.SimpleNamespace(eval_overthink=evaluator("overthink")),
            "evals.underthink_eval": types.SimpleNamespace(eval_underthink=evaluator("underthink")),
            "evals.token_eval": types.SimpleNamespace(eval_tokens=lambda *args: 10),
            "utils": types.SimpleNamespace(model_to_tokenizer=lambda name: object(),
                                           format_metrics=lambda frame: (0.0,) * 7),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "responses.jsonl"
            rows.to_json(path, orient="records", lines=True)
            with patch.dict(sys.modules, modules), patch.object(sys, "argv", ["eval.py", str(path), "--model", "test"]):
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(ROOT / "eval.py"), run_name="__main__")
        self.assertEqual(calls, [(subsets[0], "math"), (subsets[1], "math"),
                                 (subsets[2], "underthink"), (subsets[3], "overthink")])


if __name__ == "__main__":
    unittest.main()
