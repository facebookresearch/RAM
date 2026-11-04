import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1] / "src" / "utils"

def load(name):
    spec = importlib.util.spec_from_file_location("cope_checkpoint", ROOT / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class RegressionTests(unittest.TestCase):
    def test_checkpoint_without_parent_directory(self):
        module = load('checkpoint.py')
        cfg = types.SimpleNamespace(checkpoint_freq=1, distributed=False, rank=0,
            checkpoint='checkpoint.pt', checkpoint_keep=None, valid_metric=None)
        stateful = types.SimpleNamespace(state_dict=lambda: {'a': 1}, get_state=lambda: {'a': 1})
        old_cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                module.save(cfg, stateful, stateful, stateful, stateful, 0)
                self.assertTrue(Path('checkpoint.pt').is_file())
                self.assertTrue(Path('checkpoint.pt.cfg').is_file())
        finally:
            os.chdir(old_cwd)

if __name__ == "__main__":
    unittest.main()
