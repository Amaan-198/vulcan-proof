"""Cross-platform task runner. Replaces make. Works on Windows, macOS, Linux without a shell.

Usage (inside the activated .venv):
    python scripts/task.py verify-env
    python scripts/task.py lint
    python scripts/task.py test               # always-on tests
    python scripts/task.py check-phase 2      # phase done-criteria
    python scripts/task.py lock               # regenerate requirements.lock (prints the exact pip commands)

Every subcommand refuses to run unless sys.prefix is inside <repo>/.venv.
"""
from __future__ import annotations
import importlib, importlib.metadata as md, pathlib, subprocess, sys
import os

ROOT = pathlib.Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
ALWAYS_TESTS = ["tests/test_params.py", "tests/test_ev_reference.py", "tests/test_claims.py"]
os.environ.setdefault("PYTHONUTF8", "1")


def _in_venv() -> None:
    prefix = pathlib.Path(sys.prefix).resolve()
    if VENV.resolve() not in (prefix, *prefix.parents):
        sys.exit(f"REFUSING: not running inside {VENV}. Activate it first (.venv\\Scripts\\Activate.ps1).")


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    r = subprocess.run(args, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def verify_env() -> None:
    _in_venv()
    if sys.version_info[:2] != (3, 13):
        sys.exit(f"REFUSING: Python 3.13 required, found {sys.version}")
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").splitlines()
    pins = [l.split("==") for l in lock if "==" in l and not l.startswith("#")]
    for name, ver in pins:
        ver = ver.split(" ")[0].strip(" \\")
        got = md.version(name)
        if got != ver:
            sys.exit(f"REFUSING: {name} is {got}, lock says {ver}")
    import numpy as np, lightgbm as lgb
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3)); y = (X[:, 0] > 0).astype(int)
    lgb.LGBMClassifier(n_estimators=5, deterministic=True, force_row_wise=True, verbose=-1, num_threads=1).fit(X, y)
    print("LightGBM OK")
    _run([sys.executable, "-m", "pytest", "-q", "tests/test_ev_reference.py"])
    print("ENVIRONMENT OK")


def lint() -> None:
    _in_venv()
    _run([sys.executable, "-m", "vulcan_proof.params", "--lint", str(ROOT / "params" / "params.yaml")])


def test() -> None:
    _in_venv()
    _run([sys.executable, "-m", "pytest", "-q", "--basetemp", str(ROOT / "outputs" / ".pytest_tmp"), *[t for t in ALWAYS_TESTS if (ROOT / t).exists()]])


def check_phase(n: str) -> None:
    _in_venv()
    lint()
    tests = [t for t in ALWAYS_TESTS if (ROOT / t).exists()]
    extra = {
        "0": ["tests/test_seeds.py", "tests/test_manifest.py", "tests/test_phase0.py"],
        "3": ["tests/test_firewall.py", "tests/test_repro.py"],
        "4": ["tests/test_firewall.py", "tests/test_repro.py"],
        "5": ["tests/test_firewall.py"],
    }.get(n, [])
    tests += [t for t in extra if (ROOT / t).exists()]
    for k in range(int(n) + 1):
        p = f"tests/test_phase{k}.py"
        if (ROOT / p).exists():
            tests.append(p)
    # Phase-specific tests can also appear in ``extra``.  Preserve order while
    # avoiding duplicate collection (Phase 0 previously ran test_phase0 twice).
    tests = list(dict.fromkeys(tests))
    _run([sys.executable, "-m", "pytest", "-q", "--basetemp", str(ROOT / "outputs" / ".pytest_tmp"), *tests])
    _run([sys.executable, str(ROOT / "scripts" / "check_phase.py"), n])


def lock() -> None:
    print("Regenerate the lock on any machine (does not need Windows) with:\n"
          "  python -m pip download -r requirements.in --only-binary=:all: --platform win_amd64 "
          "--python-version 3.13 --implementation cp --abi cp313 --abi none --dest _wheels\n"
          "then hash every wheel in _wheels into requirements.lock (see scripts/make_lock.py). "
          "Commit requirements.in and requirements.lock together.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    {"verify-env": verify_env, "lint": lint, "test": test,
     "check-phase": lambda: check_phase(sys.argv[2]), "lock": lock}.get(cmd, lambda: sys.exit(__doc__))()
