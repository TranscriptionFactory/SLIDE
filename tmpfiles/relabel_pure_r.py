"""Apply HEAD's run_slide_r.R 0-index shift to existing pure_r benchmark outputs.

Mirrors run_slide_r.R:shift_to_0idx / shift_interacts_to_0idx / z_matrix rename.
Idempotent: skips a combo whose z_matrix.csv is already 0-indexed (has 'Z0').
"""
import sys, re
from pathlib import Path
import pandas as pd

DROOT = Path(sys.argv[1])  # e.g. benchmark_results/20260601_155411_covgate_fix/pure_r

def shift_token(tok: str) -> str:
    m = re.fullmatch(r"[Zz](\d+)", tok.strip())
    return f"Z{int(m.group(1)) - 1}" if m else tok

def shift_interact(tok: str) -> str:
    return ".".join(shift_token(p) for p in tok.strip().split("."))

for combo_dir in sorted(DROOT.glob("*_out")):
    z_path = combo_dir / "z_matrix.csv"
    if not z_path.exists():
        print(f"SKIP {combo_dir.name}: no z_matrix.csv")
        continue
    zdf = pd.read_csv(z_path, index_col=0)
    cols = list(zdf.columns)
    already = any(c == "Z0" for c in cols)
    if already:
        print(f"SKIP {combo_dir.name}: already 0-indexed")
        continue
    if not all(re.fullmatch(r"[Zz]\d+", c) for c in cols):
        print(f"SKIP {combo_dir.name}: unexpected z cols {cols[:3]}")
        continue

    # z_matrix.csv
    zdf.columns = [shift_token(c) for c in cols]
    zdf.to_csv(z_path)

    # sig_LFs.txt
    p = combo_dir / "sig_LFs.txt"
    if p.exists() and p.stat().st_size > 0:
        toks = [t for t in p.read_text().split() if t]
        p.write_text("\n".join(shift_token(t) for t in toks) + "\n")

    # sig_interacts.txt
    p = combo_dir / "sig_interacts.txt"
    if p.exists() and p.stat().st_size > 0:
        toks = [t for t in p.read_text().split() if t]
        p.write_text("\n".join(shift_interact(t) for t in toks) + "\n")

    print(f"SHIFTED {combo_dir.name}: {cols[0]}..{cols[-1]} -> Z0..{zdf.columns[-1]}")
