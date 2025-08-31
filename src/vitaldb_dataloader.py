"""vitaldb_dataloader.py
fetch ECG (ECG_II) + ABP (ART) from VitalDB and save as Parquet.
Usage:
    python vitaldb_dataloader.py --n_cases 3000 --out data/raw --overwrite
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import vitaldb

DEFAULT_SIGNALS = ["ECG_II", "ART"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--signals", nargs="*", default=DEFAULT_SIGNALS)
    ap.add_argument("--n_cases", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = (Path.cwd().parent / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    case_ids = list(map(int, vitaldb.find_cases(args.signals)))[: args.n_cases]
    print(f"Fetching {len(case_ids)} case(s)python vitaldb_dataloader.py --n_cases 5 --out data/raw --overwrite with {args.signals} → {out}")

    written = 0
    for cid in case_ids:
        f = out / f"CASE_{cid:06d}.parquet"
        if f.exists() and not args.overwrite:
            print(f"skip {f} (exists)")
            continue
        arr = vitaldb.load_case(cid, args.signals)
        df = pd.DataFrame(arr, columns=args.signals)
        #df["subject_id"] = cid
        df.to_parquet(f, index=False)
        print(f"saved {f}")
        written += 1

    print(f"Done. {written} file(s) saved.")


if __name__ == "__main__":
    main()
