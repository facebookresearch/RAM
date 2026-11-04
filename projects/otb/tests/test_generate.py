import contextlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class RegressionTests(unittest.TestCase):
    def test_generate_uses_requested_dataset(self):
        class Frame:
            def iterrows(self):
                return iter([])
            def __setitem__(self, key, value):
                pass
            def to_json(self, path, **kwargs):
                Path(path).write_text('')
        frame = Frame()
        seen = []
        def dataset(path):
            seen.append(path)
            return {'train': types.SimpleNamespace(to_pandas=lambda: frame)}
        class Pool:
            def __init__(self, **kwargs): pass
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def map(self, fn, rows): return []
        modules = {
            'datasets': types.SimpleNamespace(load_from_disk=dataset),
            'pandas': types.ModuleType('pandas'),
            'requests': types.ModuleType('requests'),
            'tqdm': types.SimpleNamespace(tqdm=lambda rows: rows),
            'utils': types.SimpleNamespace(call_llm=lambda *a, **kw: self.fail('model called')),
            'multiprocessing': types.SimpleNamespace(Pool=Pool),
        }
        with tempfile.TemporaryDirectory() as tmp:
            output = str(Path(tmp) / 'outputs' / 'result.jsonl')
            args = ['generate.py', '--dataset', 'custom/holdout', '--output', output]
            with patch.dict(sys.modules, modules), patch.object(sys, 'argv', args), contextlib.redirect_stdout(io.StringIO()):
                runpy.run_path(str(ROOT / 'generate.py'), run_name='__main__')
            self.assertEqual(seen, ['custom/holdout'])
            self.assertTrue(Path(output).exists())


if __name__ == "__main__":
    unittest.main()
