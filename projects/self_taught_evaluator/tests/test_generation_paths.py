import contextlib
import importlib.util
import io
import json
import os
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
    def test_relative_recursive_generation_paths(self):
        def read_jsonl(path):
            return [json.loads(line) for line in Path(path).read_text().splitlines()]

        def store(examples, directory, num_shards):
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "examples.jsonl").write_text("\n".join(json.dumps(row) for row in examples))

        modules = {
            "rewardbench": types.SimpleNamespace(load_eval_dataset=object()),
            "rewardbench.constants": types.SimpleNamespace(EXAMPLE_COUNTS={}, SUBSET_MAPPING={}),
            "rewardbench.utils": types.SimpleNamespace(calculate_scores_per_section=object()),
            "transformers": types.SimpleNamespace(LlamaTokenizer=object),
            "fire": types.ModuleType("fire"),
            "tqdm": types.SimpleNamespace(tqdm=lambda rows: rows, trange=range),
            "ram": types.ModuleType("ram"),
            "ram.data": types.SimpleNamespace(Dataset=types.SimpleNamespace(store_sharded_data=store)),
            "ram.data_utils": types.SimpleNamespace(load_from_jsonl=read_jsonl,
                save_to_jsonl=object(), map_str_to_uuid=lambda text: text),
        }
        with patch.dict(sys.modules, modules):
            utils = load("shared_judge_utils", ROOT / "utils.py")
            with patch.dict(sys.modules, {"utils": utils}):
                sft = load("prepare_sft_data", ROOT / "prepare_sft_data.py")
                dpo = load("prepare_dpo_data", ROOT / "prepare_dpo_data.py")
            original_cwd = os.getcwd()
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    os.chdir(tmp)
                    source = Path("data/generations/nested")
                    source.mkdir(parents=True)
                    rows = []
                    for question, ranks in [("first", [1, 2]), ("second", [2, 1])]:
                        for verdict in ["[[A]]", "[[B]]"]:
                            rows.append({"text": "system\n\n[User Question]" + question,
                                         "metadata": {"ranks": ranks},
                                         "vllm_output": {"output": verdict}})
                    (source / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
                    with contextlib.redirect_stdout(io.StringIO()):
                        sft.prepare_positive_data("data/generations", "out/sft")
                    class FileReadComplete(Exception):
                        pass
                    def stop_after_read(path):
                        self.assertEqual(len(read_jsonl(path)), 4)
                        raise FileReadComplete
                    with patch.object(dpo, "load_from_jsonl", stop_after_read), contextlib.redirect_stdout(io.StringIO()):
                        with self.assertRaises(FileReadComplete):
                            dpo.prepare_dpo_data("data/generations", "out/dpo")
                    self.assertEqual(len(read_jsonl("out/sft/all/examples.jsonl")), 2)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
