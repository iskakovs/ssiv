import os
import re
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_readme_quickstart_runs(capsys):
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert blocks
    exec(compile(blocks[0], "README.md", "exec"), {})
    out = capsys.readouterr().out
    assert "AKM0" in out and "Shock-level IV" in out


@pytest.mark.skipif(not os.environ.get("SSIV_ADH_DIR"), reason="set SSIV_ADH_DIR to run")
def test_adh_example_runs(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["adh_replication.py", os.environ["SSIV_ADH_DIR"]])
    runpy.run_path(str(ROOT / "examples" / "adh_replication.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "-0.7742267" in out and "Rotemberg" in out
