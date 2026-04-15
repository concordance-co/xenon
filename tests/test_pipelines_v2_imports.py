from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_remote_executor_import_does_not_require_matplotlib() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = textwrap.dedent(
        """
        import builtins
        import importlib

        real_import = builtins.__import__

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "matplotlib" or name.startswith("matplotlib."):
                raise ModuleNotFoundError("No module named 'matplotlib'")
            return real_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        importlib.import_module("pipelines_v2.runtime.remote_executor")
        print("ok")
        """
    )
    env = os.environ.copy()
    pythonpath = str(repo_root)
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "ok"
