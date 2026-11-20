import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1] / "src"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegressionTests(unittest.TestCase):
    def test_prepare_dpo_uses_shared_judgement_parser(self):
        saved = []

        def read_jsonl(path):
            return [json.loads(line) for line in Path(path).read_text().splitlines()]

        modules = {
            "rewardbench": types.SimpleNamespace(load_eval_dataset=object()),
            "rewardbench.constants": types.SimpleNamespace(EXAMPLE_COUNTS={}, SUBSET_MAPPING={}),
            "rewardbench.utils": types.SimpleNamespace(calculate_scores_per_section=object()),
            "transformers": types.SimpleNamespace(LlamaTokenizer=object),
            "fire": types.ModuleType("fire"),
            "tqdm": types.SimpleNamespace(tqdm=lambda rows: rows, trange=range),
            "ram": types.ModuleType("ram"),
            "ram.data": types.SimpleNamespace(Dataset=types.SimpleNamespace(
                store_sharded_data=lambda examples, *args: saved.extend(examples))),
            "ram.data_utils": types.SimpleNamespace(load_from_jsonl=read_jsonl, map_str_to_uuid=lambda text: text),
        }
        with patch.dict(sys.modules, modules):
            utils = load("shared_judge_utils", ROOT / "utils.py")
            with patch.dict(sys.modules, {"utils": utils}):
                module = load("prepare_dpo_data", ROOT / "prepare_dpo_data.py")
            rows = []
            for question, ranks in [("first", [1, 2]), ("second", [2, 1])]:
                for verdict in ["[[A]]", "[[B]]"]:
                    rows.append({"text": "system\n\n[User Question]" + question,
                                 "metadata": {"ranks": ranks},
                                 "vllm_output": {"output": verdict}})
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "generations"
                source.mkdir()
                (source / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
                negative_source = Path(tmp) / "negative_generations"
                negative_source.mkdir()
                negative_rows = [dict(rows[0], vllm_output={"output": "unrecognized verdict"}),
                                 dict(rows[0], vllm_output={"output": "[[A]]"})]
                (negative_source / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in negative_rows))
                parsed = []
                def parse_and_record(output):
                    parsed.append(output)
                    return utils.parse_judgement(output)
                module.parse_judgement = parse_and_record
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    module.prepare_dpo_data(str(source), str(Path(tmp) / "output"),
                                            negative_generation_dir=str(negative_source))
                self.assertEqual(len(parsed), 6)
                self.assertIn("unrecognized verdict", parsed)
        self.assertEqual(len(saved), 2)
        by_question = {"first" if "first" in row["src"] else "second": row for row in saved}
        self.assertEqual(by_question["first"]["tgt_chosen"], "[[A]]")
        self.assertEqual(by_question["second"]["tgt_chosen"], "[[B]]")
        self.assertNotEqual(by_question["first"]["tgt_chosen"], by_question["first"]["tgt_rejected"])


if __name__ == "__main__":
    unittest.main()
