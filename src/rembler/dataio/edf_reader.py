from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np


@dataclass
class EDFLoadConfig:
    target_sample_rate: int = 500
    eeg_channels: Optional[List[str]] = None     # exact names or regex handled downstream
    emg_channels: Optional[List[str]] = None
    bandpass: Tuple[float, float] = (0.5, 100.0)
    notch_hz: Optional[float] = 60.0
    reref: Optional[str] = None                  # e.g. 'average' or 'EEG2' to re-reference EEG1-EEG2
    preload: bool = True

@dataclass
class Recording:
    data: np.ndarray               # shape (C, T) after resample/filter; EEG first, then EMG
    ch_names: List[str]
    sr: int
    start_time_utc: Optional[str]
    annotations: List[Tuple[float, float, str]]  # (onset_sec, duration_sec, desc)
    metadata: Dict

def _pick_channels(raw: mne.io.BaseRaw, wanted: Optional[List[str]]) -> List[int]:
    if not wanted:
        return list(range(len(raw.ch_names)))
    picks = []
    for w in wanted:
        matches = mne.pick_channels_regexp(raw.ch_names, regexp=w) if any(x in w for x in r".*[]()|^$") else \
                  mne.pick_channels(raw.ch_names, include=[w], ordered=False)
        picks.extend(matches.tolist() if hasattr(matches, "tolist") else list(matches))
    return sorted(set(picks))

def load_edf(path: str, cfg: EDFLoadConfig) -> Recording:
    raw = mne.io.read_raw_edf(path, preload=cfg.preload, verbose="ERROR")
    # Standardize channel types best-effort
    ch_types = {ch: ('emg' if 'emg' in ch.lower() else 'eeg') for ch in raw.ch_names}
    raw.set_channel_types(ch_types, verbose="ERROR")

    # Pick EEG+EMG in requested order: EEG first then EMG
    eeg_ix = _pick_channels(raw, cfg.eeg_channels) if cfg.eeg_channels else mne.pick_types(raw, eeg=True)
    emg_ix = _pick_channels(raw, cfg.emg_channels) if cfg.emg_channels else mne.pick_types(raw, emg=True)
    picks = list(eeg_ix) + list(emg_ix)
    raw.pick(picks)

    # Re-reference (EEG only)
    if cfg.reref == 'average':
        eeg_picks = mne.pick_types(raw, eeg=True, emg=False)
        if len(eeg_picks) >= 1:
            raw.set_eeg_reference('average', projection=False, verbose="ERROR")
    elif isinstance(cfg.reref, str) and cfg.reref in raw.ch_names:
        raw.set_eeg_reference([cfg.reref], projection=False, verbose="ERROR")

    # Filtering
    l_freq, h_freq = cfg.bandpass
    raw.filter(l_freq=l_freq, h_freq=h_freq, phase='zero', verbose="ERROR")
    if cfg.notch_hz:
        raw.notch_filter(freqs=[cfg.notch_hz], phase='zero', verbose="ERROR")

    # Resample
    if int(raw.info['sfreq']) != cfg.target_sample_rate:
        raw.resample(cfg.target_sample_rate, npad="auto", verbose="ERROR")

    data = raw.get_data()  # (C, T)
    # Extract annotations (if present)
    ann = []
    if raw.annotations and len(raw.annotations) > 0:
        for onset, duration, desc in zip(raw.annotations.onset, raw.annotations.duration, raw.annotations.description):
            # MNE stores onset in seconds from recording start
            ann.append((float(onset), float(duration), str(desc)))

    start_dt = raw.info['meas_date'].isoformat() if raw.info.get('meas_date') else None
    meta = {
        "edf_path": path,
        "orig_sr": int(raw.info['sfreq']),
        "mapped_ch_types": ch_types,
    }
    return Recording(
        data=data,
        ch_names=raw.ch_names,
        sr=cfg.target_sample_rate,
        start_time_utc=start_dt,
        annotations=ann,
        metadata=meta,
    )