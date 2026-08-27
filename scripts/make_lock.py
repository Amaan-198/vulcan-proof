"""Build requirements.lock from a directory of downloaded wheels. See scripts/task.py lock."""
import hashlib, pathlib, re, sys
src = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "_wheels")
rows = []
for f in sorted(src.glob("*.whl")):
    m = re.match(r"([A-Za-z0-9_.]+?)-(\d[^-]*)-", f.name)
    rows.append((m.group(1).replace("_", "-").lower(), m.group(2), hashlib.sha256(f.read_bytes()).hexdigest()))
out = ["# LOCKFILE — win_amd64 / CPython 3.13 binary wheels only. Install with:",
       "#   python -m pip install --require-hashes --only-binary=:all: -r requirements.lock",
       "# No substitution path exists: a missing wheel or hash mismatch fails the install."]
for n, v, h in rows:
    out += [f"{n}=={v} \\", f"    --hash=sha256:{h}"]
pathlib.Path("requirements.lock").write_text("\n".join(out) + "\n", encoding="utf-8")
print(len(rows), "entries written")
