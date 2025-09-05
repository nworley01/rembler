#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from src.preprocessing.epocher import epochize

from src.dataio.edf_reader import EDFLoadConfig, load_edf
from src.utils.labels import labels_from_annotations, labels_from_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edf_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epoch_sec", type=int, default=4)
    ap.add_argument("--target_sr", type=int, default=512)
    ap.add_argument("--eeg", nargs="*", default=None, help="EEG channel names or regex (MNE-style)")
    ap.add_argument("--emg", nargs="*", default=None, help="EMG channel names or regex (MNE-style)")
    ap.add_argument("--label_csv", default=None, help="Optional path to CSV labels (per recording): expects <recording_stem>.csv in this folder")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    edf_paths = sorted(Path(args.edf_dir).glob("*.edf"))

    cfg = EDFLoadConfig(target_sr=args.target_sr, eeg_channels=args.eeg, emg_channels=args.emg)

    index = []
    for p in edf_paths:
        rec = load_edf(str(p), cfg)
        X, info = epochize(rec.data, rec.sr, epoch_sec=args.epoch_sec, add_artifact_flag=True)
        total_sec = X.shape[0] * args.epoch_sec

        # Labels priority: CSV (if provided for this file) > EDF annotations > unknown
        y = None
        if args.label_csv:
            csv_path = Path(args.label_csv) / (p.stem + ".csv")
            if csv_path.exists():
                y = labels_from_csv(str(csv_path), epoch_sec=args.epoch_sec, total_sec=total_sec)
        if y is None and len(rec.annotations) > 0:
            y = labels_from_annotations(rec.annotations, total_sec=total_sec, epoch_sec=args.epoch_sec)
        if y is None:
            y = -1 * np.ones((X.shape[0],), dtype=np.int16)

        # Align lengths defensively
        n = min(len(X), len(y))
        X, y = X[:n], y[:n]
        art = info["artifact_flags"][:n] if info["artifact_flags"] is not None else np.zeros((n,), dtype=np.uint8)

        out_npz = out / f"{p.stem}.npz"
        np.savez_compressed(out_npz, X=X, y=y, sr=rec.sr, epoch_sec=args.epoch_sec, ch_names=np.array(rec.ch_names), artifact=art)
        index.append({"file": p.name, "npz": str(out_npz), "n_epochs": int(n)})

    with open(out / "index.json", "w") as f:
        json.dump(index, f, indent=2)
    print(f"Wrote {len(index)} recordings to {out}")

if __name__ == "__main__":
    main()