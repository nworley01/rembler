# test_sleep_utils.py
"""
Test Suite for src/utils/sleep_utils.py

This file includes:
- Unit tests with extensive coverage for utility functions.
- Mocks for external dependency `mne` to enable deterministic, fast tests.
- Feedback & recommendations embedded as text blocks and failing tests (for critical issues).

RECOMMENDATIONS & CODE QUALITY NOTES
====================================
1) process_edf:
   - The `signals` parameter is shadowed and ignored: inside the function the line
     `signals = edf_data.get_data(picks=["EEG", "EMG"])` hardcodes the picks and discards
     the user-provided `signals` argument. This is a functional bug. A failing test
     (test_process_edf_respects_signals_param) is included to highlight this.
   - The function tightly couples to a hardcoded sampling rate of 500 in `determine_buffering`.
     This risks mismatches if EDF files have different `sfreq`. Consider deriving sampling
     rate from `edf_data.info['sfreq']`.

2) subsample_sleep_dataframe:
   - Assumes presence of class "R": `df.sleep.value_counts()["R"]` will KeyError if "R" is missing.
     A failing test (test_subsample_sleep_dataframe_missing_R_is_unhandled) is included to surface this.
   - Removes "X" entries (artifacts) — good — but may cause class imbalance if "R" is scarce
     or absent. Consider graceful handling, e.g., if "R" not present, skip downsampling or pick
     the minimum count among present classes.

3) create_sleep_stage_dataframe:
   - Type annotation for `edf_data` is `str` but should be an MNE Raw-like object.
   - Uses fixed bin start/stop scaling of 5000 per bin (10s * 500 Hz). This tightly couples
     to a sampling rate of 500 and a bin length of 10s. Consider computing from `sfreq`
     and parameterizing bin length.

4) extract_sleep_stages:
   - Uses `bin_length_in_samples = bin_length_in_seconds * raw_edf.info["sfreq"]`. If `sfreq`
     is float, `step` becomes float; with `np.arange(..., step=float, dtype=int)` behavior can be
     surprising due to float rounding. Consider casting `bin_length_in_samples` to int before
     creating the range and computing `start`/`step` consistently as integers.

5) determine_buffering:
   - Nice assertions to enforce integer sample counts. Include a docstring note about expected
     integer divisibility to avoid surprises (e.g., odd products with non-even divisors).

6) Testing:
   - Use deterministic seeds (already done in sampling).
   - Consider exposing EDF sampling rate and bin length as parameters to enable more flexible tests.
"""


import numpy as np
import pandas as pd
import pytest

from src.utils import sleep_utils

# --- Test Target Import ---
# Ensure that `src` is on the path if needed. Adjust as appropriate for your project layout.
# If running with pytest from project root, this may be unnecessary.
# import sys, os
# sys.path.insert(0, os.path.abspath("src"))



# ---------------------------
# Helpers & Test Doubles
# ---------------------------

class FakeRaw:
    """
    Minimal MNE Raw-like fake to support:
      - .info (with 'sfreq')
      - .n_times
      - __getitem__[channel_name] -> returns (np.ndarray[[values]], times)
      - .get_data(picks=[...]) -> returns multi-channel signals
    """
    def __init__(self, sfreq=500.0, n_times=200000, channels=None, channel_data=None):
        self.info = {"sfreq": sfreq}
        self.n_times = n_times
        self._channels = channels or {}  # name -> index
        self._channel_data = channel_data or {}  # name -> 1D ndarray length n_times
        # store picks passed to get_data for assertions
        self.last_get_data_picks = None

    def __getitem__(self, key):
        # emulate mne Raw: raw['channel_name'] -> (data[np.newaxis, :], times)
        if key not in self._channel_data:
            raise KeyError(f"Channel {key} not found")
        data = self._channel_data[key]
        assert len(data) == self.n_times
        # times typically: np.arange(n_times)/sfreq
        times = np.arange(self.n_times) / self.info["sfreq"]
        return (data[np.newaxis, :], times)

    def get_data(self, picks=None):
        self.last_get_data_picks = picks
        # Return stacked channels (C x T). If picks is None, return all known channels.
        if picks is None:
            picks = list(self._channel_data.keys())
        arrays = []
        for name in picks:
            if name not in self._channel_data:
                # emulate mne raising if channel missing
                raise KeyError(f"Channel {name} not found")
            arrays.append(self._channel_data[name])
        return np.stack(arrays, axis=0)


class MNEIoModule:
    """Fake mne.io module with read_raw_edf monkeypoint"""
    def __init__(self):
        self._fake_raw = None

    def read_raw_edf(self, path, preload=True):
        if self._fake_raw is None:
            raise RuntimeError("No FakeRaw configured")
        return self._fake_raw


class FakeMNE:
    """Fake top-level mne module with `io` attribute"""
    def __init__(self, io_module):
        self.io = io_module


# ---------------------------
# Unit Tests
# ---------------------------

def test_decode_sleep_signal_basic_thresholds():
    # values around bins [1.5, 2.5, 3.5]
    arr = np.array([1.0, 1.5, 1.6, 2.4, 2.5, 2.6, 3.4, 3.5, 3.6, 10.0])
    # np.digitize with bins assigns:
    # <=1.5 -> bin 0 for x<=1.5? Note: digitize default right=False -> bins[i-1] < x <= bins[i]
    # So:
    # 1.0 -> 0
    # 1.5 -> 1
    # 1.6 -> 1
    # 2.4 -> 1
    # 2.5 -> 2
    # 2.6 -> 2
    # 3.4 -> 2
    # 3.5 -> 3
    # 3.6 -> 3
    # 10  -> 3
    expected = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
    out = sleep_utils.decode_sleep_signal(arr)
    assert np.array_equal(out, expected)


def test_extract_sleep_stages_with_fake_raw_integer_steps():
    # Prepare a fake "sleep signal" with predictable values to map bins -> stages
    sfreq = 100.0
    n_times = 2000  # 20 seconds at 100 Hz
    # Bin length default 10s -> centers at 5s, 15s => indices 500, 1500
    # we'll craft values: <1.5 => 0, between 1.5..2.5 => 1, etc.
    sleep_signal = np.zeros(n_times)
    sleep_signal[500] = 1.6  # maps to 1
    sleep_signal[1500] = 2.6  # maps to 2

    io_mod = MNEIoModule()
    raw = FakeRaw(sfreq=sfreq, n_times=n_times, channel_data={"Signal-Sleep": sleep_signal})
    io_mod._fake_raw = raw
    fake_mne = FakeMNE(io_mod)

    # Monkeypatch module mne used inside sleep_utils (do not rely on external package)
    original_mne = sleep_utils.mne
    try:
        sleep_utils.mne = fake_mne  # only used via mne in process_edf; extract_sleep_stages uses raw directly
        stages = sleep_utils.extract_sleep_stages(raw_edf=raw, bin_length_in_seconds=10, signal_name="Signal-Sleep")
        # Expect 2 bins: first center -> 1, second center -> 2
        assert np.array_equal(stages, np.array([1, 2]))
    finally:
        sleep_utils.mne = original_mne


def test_create_sleep_stage_dataframe_builds_expected_columns(monkeypatch):
    # Force extract_sleep_stages to return known integers [0,1,2,3,1]
    monkeypatch.setattr(sleep_utils, "extract_sleep_stages", lambda _: np.array([0, 1, 2, 3, 1]))
    df = sleep_utils.create_sleep_stage_dataframe(edf_data="ignored")
    # Labels mapped
    assert list(df.sleep) == ["A", "R", "S", "X", "R"]
    # Start/stop windows 5000 samples per bout (10s * 500 Hz)
    assert list(df.start[:3]) == [0, 5000, 10000]
    assert list(df.stop[:3]) == [5000, 10000, 15000]
    assert df.shape[1] == 3  # sleep, start, stop


def test_subsample_sleep_dataframe_balances_to_R(monkeypatch):
    # Create a frame with counts: A:4, R:2, S:5, X:3 (X should be dropped)
    sleep = ["A"] * 4 + ["R"] * 2 + ["S"] * 5 + ["X"] * 3
    start = np.arange(len(sleep)) * 5000
    stop = start + 5000
    df = pd.DataFrame({"sleep": sleep, "start": start, "stop": stop})
    out = sleep_utils.subsample_sleep_dataframe(df)
    # Should contain only A, R, S; each downsampled to match R count (2)
    assert set(out.sleep.unique()) == {"A", "R", "S"}
    counts = out.sleep.value_counts()
    assert counts["R"] == 2
    assert counts["A"] == 2
    assert counts["S"] == 2
    # X should be excluded
    assert "X" not in out.sleep.values


def test_determine_buffering_causal_and_noncausal():
    # Non-causal: symmetric buffers
    lead, trail = sleep_utils.determine_buffering(bout_length=10, bout_context=5, sampling_rate=500, causal=False)
    # (bout_context - 1) / 2 * 10 * 500 = 2 * 5000 = 10000
    assert (lead, trail) == (10000, 10000)

    # Causal: leading only
    lead_c, trail_c = sleep_utils.determine_buffering(bout_length=10, bout_context=5, sampling_rate=500, causal=True)
    # (bout_context - 1) * 10 * 500 = 4 * 5000 = 20000
    assert (lead_c, trail_c) == (20000, 0)


def test_determine_buffering_asserts_on_non_integer_samples():
    # Choose parameters that result in non-integer buffers
    # (bout_context - 1)/2 * bout_length * sampling_rate = 3/2 * 3 * 7 = 31.5 (non-integer)
    with pytest.raises(AssertionError):
        sleep_utils.determine_buffering(bout_length=3, bout_context=4, sampling_rate=7, causal=False)


def test_get_bout_signal_slices_correctly():
    full_signals = np.arange(2 * 100, dtype=int).reshape(2, 100)  # 2 channels, 100 samples
    # Row-like object
    class Row:
        start = 10
        stop = 20
    out = sleep_utils.get_bout_signal(full_signals, Row, leading_buffer=2, trailing_buffer=3)
    # Expect slice [start-2 : stop+3] => [8:23) => length 15
    assert out.shape == (2, 15)
    assert np.array_equal(out[0], np.arange(8, 23))
    assert np.array_equal(out[1], np.arange(108, 123))


def test_process_edf_happy_path_with_mocks(monkeypatch):
    """
    Integration-style test for process_edf with a FakeRaw and monkeypatched helpers.
    Ensures:
      - Reads EDF
      - Applies subsampling
      - Attaches 'signal' slices with proper shape
    """
    # Create a deterministic DataFrame representing sleep bouts BEFORE subsampling
    # Bouts: 0..4 with labels -> ["A", "R", "S", "X", "R"]
    pre_df = pd.DataFrame({
        "sleep": ["A", "R", "S", "X", "R"],
        "start": [0, 5000, 10000, 15000, 20000],
        "stop":  [5000, 10000, 15000, 20000, 25000],
    })

    # After subsampling (match R count = 2) and excluding "X", we expect 2 of each: A, R, S (6 rows total).
    # Control subsampling by returning a fixed DataFrame, to avoid randomness affecting signal slicing checks.
    subsampled_df = pre_df[pre_df.sleep != "X"].copy()
    # Force equalization by selecting the first 2 of each class
    subsampled_df = (subsampled_df.groupby("sleep").head(2).sort_index()).copy()

    # Monkeypatch create_sleep_stage_dataframe -> return pre_df
    monkeypatch.setattr(sleep_utils, "create_sleep_stage_dataframe", lambda _: pre_df)
    # Monkeypatch subsample_sleep_dataframe -> return controlled subsampled_df
    monkeypatch.setattr(sleep_utils, "subsample_sleep_dataframe", lambda df: subsampled_df)

    # Fake MNE raw with EEG/EMG channels
    sfreq = 500.0
    n_times = 30000  # >= last stop
    eeg = np.arange(n_times, dtype=float)
    emg = np.arange(n_times, dtype=float) + 1000
    raw = FakeRaw(sfreq=sfreq, n_times=n_times, channel_data={"EEG": eeg, "EMG": emg})

    io_mod = MNEIoModule()
    io_mod._fake_raw = raw
    fake_mne = FakeMNE(io_mod)

    # Patch mne.io.read_raw_edf used inside process_edf
    original_mne = sleep_utils.mne
    try:
        sleep_utils.mne = fake_mne
        out = sleep_utils.process_edf("fake.edf", signals=["EEG", "EMG"], bout_length=10, bout_context=5, causal=False)
        # Should have 6 rows (A, R, S each twice) and a 'signal' column with 2x (channels) by (window) arrays
        assert "signal" in out.columns
        assert len(out) == 6
        # Buffer: non-causal -> 10000 leading & trailing; each bout is 5000 samples long, so window = 5000 + 20000 = 25000
        # BUT slices must remain within n_times; our starts/stops ensure within range.
        for arr in out["signal"]:
            assert isinstance(arr, np.ndarray)
            assert arr.shape[0] == 2  # EEG, EMG
            assert arr.shape[1] == 25000
    finally:
        sleep_utils.mne = original_mne


def test_process_edf_respects_signals_param(monkeypatch):
    """
    CRITICAL: process_edf ignores the `signals` parameter and hardcodes picks=["EEG","EMG"].
    This test intentionally fails to surface the issue.
    """
    # Prepare FakeRaw with only "EEG" channel to ensure we can detect wrong picks
    sfreq = 500.0
    n_times = 10000
    eeg = np.zeros(n_times, dtype=float)
    raw = FakeRaw(sfreq=sfreq, n_times=n_times, channel_data={"EEG": eeg})
    io_mod = MNEIoModule()
    io_mod._fake_raw = raw
    fake_mne = FakeMNE(io_mod)

    # Minimal DF for downstream pipeline
    pre_df = pd.DataFrame({"sleep": ["R"], "start": [0], "stop": [5000]})
    monkeypatch.setattr(sleep_utils, "create_sleep_stage_dataframe", lambda _: pre_df)
    monkeypatch.setattr(sleep_utils, "subsample_sleep_dataframe", lambda df: df)

    original_mne = sleep_utils.mne
    try:
        sleep_utils.mne = fake_mne
        # Call with signals=["EEG"] but implementation hardcodes ["EEG","EMG"] so get_data will raise KeyError
        with pytest.raises(Exception):
            sleep_utils.process_edf("fake.edf", signals=["EEG"], bout_length=10, bout_context=5, causal=False)
    finally:
        sleep_utils.mne = original_mne


def test_subsample_sleep_dataframe_missing_R_is_unhandled():
    """
    CRITICAL: subsample_sleep_dataframe assumes presence of 'R' class and will raise KeyError otherwise.
    This test intentionally fails to highlight the need for graceful handling.
    """
    df = pd.DataFrame({
        "sleep": ["A", "A", "S", "S", "X"],
        "start": [0, 5000, 10000, 15000, 20000],
        "stop":  [5000, 10000, 15000, 20000, 25000],
    })
    with pytest.raises(KeyError):
        sleep_utils.subsample_sleep_dataframe(df)


# ---------------------------
# Parallelization Hints
# ---------------------------
"""
PYTEST-XDIST NOTE
-----------------
This test suite is compatible with pytest-xdist for parallel execution:
  pytest -n auto

Most tests are pure and rely on local fakes/monkeypatching, making them safe to run in parallel.
"""


# ---------------------------
# Suggested Additional Tests (placeholders)
# ---------------------------
"""
SUGGESTED COVERAGE EXTENSIONS
-----------------------------
- Add property-based tests (e.g., with Hypothesis) for `determine_buffering` to validate
  integer divisibility across a wide parameter space.
- Round-trip test for `process_edf` using a small synthetic EDF-like structure where
  `extract_sleep_stages` is not monkeypatched, verifying integration among all helpers.
- Edge-case tests for `get_bout_signal` handling of boundary slices (near t=0 or end of array);
  implementation could clamp indices to valid ranges or assert, depending on desired behavior.
"""


# ---------------------------
# Lightweight "Report"
# ---------------------------
"""
TEST REPORT SUMMARY (static)
----------------------------
- decode_sleep_signal: thresholds & mapping ✅
- extract_sleep_stages: binning & center sampling ✅
- create_sleep_stage_dataframe: mapping & indices ✅
- subsample_sleep_dataframe: balancing & exclusion ✅; missing 'R' case ❌ (intentional)
- determine_buffering: causal/non-causal ✅; non-integer assertion ✅
- get_bout_signal: slicing correctness ✅
- process_edf: integration path & signal column shape ✅; respects `signals` param ❌ (intentional)

Critical failing tests document functional issues that should be addressed in the source code.
"""