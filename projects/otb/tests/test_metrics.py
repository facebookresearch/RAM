import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_macro_average_weights_general_and_math_once_each(self):
        modules = {
            "openai": types.SimpleNamespace(AzureOpenAI=object),
            "litellm": types.SimpleNamespace(completion=object()),
            "transformers": types.SimpleNamespace(AutoTokenizer=object),
        }
        spec = importlib.util.spec_from_file_location("otb_metrics_utils", ROOT / "utils.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        rows = pd.DataFrame([
            {"subset": "overthinking-bench", "accuracy": 1.0, "score": 1.0, "tokens": 10},
            {"subset": "overthinking-math", "accuracy": 0.0, "score": 0.0, "tokens": 30},
            {"subset": "underthinking-bench", "accuracy": 1.0, "score": 1.0, "tokens": 10},
            {"subset": "underthinking-math", "accuracy": 0.0, "score": 0.0, "tokens": 30},
        ])
        expected = (0.5, 0.5, 20.0, 0.5, 0.5, 20.0, 0.5)
        self.assertEqual(module.format_metrics(rows), expected)
        # Adding a general row must not change the intended split weighting.
        extra_general = rows.iloc[[0, 2]]
        self.assertEqual(module.format_metrics(pd.concat([rows, extra_general])), expected)


if __name__ == "__main__":
    unittest.main()
