import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_env_example_has_sections():
    spec = importlib.util.spec_from_file_location("check_env", ROOT / "scripts" / "check_env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    parsed = mod.parse(ROOT / ".env.example")
    assert any(sec == "REQUIRED" for sec, _ in parsed.values())
