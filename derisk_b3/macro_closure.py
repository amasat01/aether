"""Every AETHER_* macro the shared layers (root, components) USE is defined by root or by EACH variant (closed per variant)."""
import pathlib, re, sys
h = pathlib.Path(sys.argv[1])
use = re.compile(r"\b(AETHER_[A-Z0-9_]+)\b"); dfn = re.compile(r"^\s*#\s*define\s+(AETHER_[A-Z0-9_]+)", re.M)
def defs(d): return {m for f in d.rglob("*.h") for m in dfn.findall(f.read_text(errors="replace"))}
def strip(t): return re.sub(r"//[^\n]*|/\*.*?\*/", "", t, flags=re.S)
shared = {}
for layer in ("root", "components"):
    for f in (h / layer).rglob("*.h"):
        for m in use.findall(strip(f.read_text(errors="replace"))):
            shared.setdefault(m, f.relative_to(h).as_posix())
base = defs(h / "root") | defs(h / "components")
bad = 0
for var in sorted(p for p in (h / "variants").iterdir() if p.is_dir()):
    missing = sorted(m for m in shared if m not in base and m not in defs(var))
    print(f"{var.name}: {len(missing)} missing" + "".join(f"\n  {m} (used in {shared[m]})" for m in missing))
    bad += len(missing)
sys.exit(1 if bad else 0)
