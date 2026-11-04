import contextlib
import importlib
import io
from pathlib import Path
import runpy
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_run_evaluates_the_file_generation_writes(self):
        class Frame:
            def iterrows(self): return iter([])
            def __setitem__(self, key, value): pass
            def to_json(self, path, **kwargs): Path(path).write_text("")

        class Pool:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, fn, rows): return []

        package = types.ModuleType("otbench")
        package.__path__ = [str(ROOT / "otbench")]
        modules = {
            "otbench": package,
            "datasets": types.SimpleNamespace(load_from_disk=lambda path: {
                "train": types.SimpleNamespace(to_pandas=lambda: Frame())}),
            "pandas": types.ModuleType("pandas"),
            "requests": types.ModuleType("requests"),
            "tqdm": types.SimpleNamespace(tqdm=lambda rows: rows),
            "utils": types.SimpleNamespace(call_llm=lambda *args: self.fail("model call")),
            "multiprocessing": types.SimpleNamespace(Pool=Pool),
        }
        with patch.dict(sys.modules, modules):
            for name in ["otbench.cli", "otbench.paths"]:
                sys.modules.pop(name, None)
            cli = importlib.import_module("otbench.cli")
            cases = [
                (False, 0.6, 8, ""), (False, 0.6, 2, "-num_gens=2"),
                (False, 0.2, 8, "-temp=0.2"), (False, 0.2, 2, "-temp=0.2-num_gens=2"),
                (True, 0.6, 8, "-nothink"), (True, 0.6, 2, "-nothink-num_gens=2"),
                (True, 0.2, 8, "-nothink-temp=0.2"),
                (True, 0.2, 2, "-nothink-temp=0.2-num_gens=2"),
            ]
            for nothink, temperature, num_gens, suffix in cases:
                with self.subTest(nothink=nothink, temperature=temperature, num_gens=num_gens):
                    with tempfile.TemporaryDirectory() as tmp:
                        output = str(Path(tmp) / "outputs" / "{{model}}.jsonl")
                        argv = ["run", "--model", "org/model", "--output", output,
                                "--temperature", str(temperature), "--num_gens", str(num_gens)]
                        if nothink:
                            argv.append("--enable_nothink")
                        args = cli.build_parser().parse_args(argv)
                        evaluated = []

                        def run(command):
                            parts = shlex.split(command)
                            if parts[1].endswith("generate.py"):
                                with patch.object(sys, "argv", parts[1:]), contextlib.redirect_stdout(io.StringIO()):
                                    runpy.run_path(str(ROOT / "generate.py"), run_name="__main__")
                            elif parts[1].endswith("eval.py"):
                                evaluated.append(parts[2])
                                self.assertTrue(Path(parts[2]).is_file(), parts[2])
                            return 0

                        with patch.object(cli, "_run", run):
                            self.assertEqual(args.func(args), 0)
                        self.assertEqual(evaluated, [str(Path(tmp) / "outputs" / ("org-model" + suffix + ".jsonl"))])


if __name__ == "__main__":
    unittest.main()
