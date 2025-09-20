import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from src.utils import sleep_utils


class FakeRaw:
    """Minimal stand-in for an mne.io.Raw object."""

    def __init__(self, sfreq: float, channel_data: dict[str, np.ndarray]):
        self.info = {"sfreq": sfreq}
        lengths = {len(values) for values in channel_data.values()}
        if len(lengths) != 1:
            raise ValueError("All channels must have the same length")
        self.n_times = lengths.pop()
        self._channel_data = channel_data

    def __getitem__(self, key: str):
        if key not in self._channel_data:
            raise KeyError(f"Channel {key} not found")
        data = self._channel_data[key]
        times = np.arange(self.n_times) / self.info["sfreq"]
        return data[np.newaxis, :], times

    def get_data(self, picks=None):
        if picks is None:
            picks = list(self._channel_data.keys())
        arrays = []
        for name in picks:
            if name not in self._channel_data:
                raise KeyError(f"Channel {name} not found")
            arrays.append(self._channel_data[name])
        return np.stack(arrays, axis=0)


def test_decode_sleep_signal_assigns_expected_bins():
    values = np.array([1.0, 1.5, 1.6, 2.4, 2.5, 2.6, 3.4, 3.5, 3.6])
    expected = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3])
    result = sleep_utils.decode_sleep_signal(values)
    assert np.array_equal(result, expected)


def test_determine_buffering_handles_causal_and_noncausal():
    lead, trail = sleep_utils.determine_buffering(
        bout_length=10, bout_context=5, sampling_rate=500, causal=False
    )
    assert (lead, trail) == (10000, 10000)

    lead_causal, trail_causal = sleep_utils.determine_buffering(
        bout_length=10, bout_context=5, sampling_rate=500, causal=True
    )
    assert (lead_causal, trail_causal) == (20000, 0)


def test_determine_buffering_requires_integer_sample_counts():
    with pytest.raises(AssertionError):
        sleep_utils.determine_buffering(
            bout_length=3, bout_context=4, sampling_rate=7, causal=False
        )


def test_get_bout_signal_extracts_requested_window():
    full_signals = np.arange(2 * 100, dtype=int).reshape(2, 100)

    class Row:
        start = 10
        stop = 20

    result = sleep_utils.get_bout_signal(
        full_signals, Row, leading_buffer=2, trailing_buffer=3
    )
    assert result.shape == (2, 15)
    assert np.array_equal(result[0], np.arange(8, 23))
    assert np.array_equal(result[1], np.arange(108, 123))


def test_extract_sleep_context_uses_neighboring_labels():
    df = pd.DataFrame({"sleep": ["A", "R", "S", "A", "R"]})
    context = sleep_utils.extract_sleep_context(df)
    assert list(context) == ["ARS", "ARSA", "ARSAR", "RSAR", "SAR"]


def test_subsample_sleep_dataframe_balances_to_r_class():
    sleep = ["A"] * 4 + ["R"] * 2 + ["S"] * 5 + ["X"] * 3
    start = np.arange(len(sleep)) * 5000
    stop = start + 5000
    df = pd.DataFrame({"sleep": sleep, "start": start, "stop": stop})

    result = sleep_utils.subsample_sleep_dataframe(df)
    counts = result.sleep.value_counts()
    assert set(result.sleep.unique()) == {"A", "R", "S"}
    assert counts["R"] == 2
    assert counts["A"] == 2
    assert counts["S"] == 2


def test_extract_sleep_stages_uses_bin_centers():
    sfreq = 2.0
    n_times = 80
    sleep_signal = np.zeros(n_times)
    sleep_signal[10] = 1.6
    sleep_signal[30] = 2.6
    sleep_signal[50] = 3.6
    sleep_signal[70] = 0.4
    raw = FakeRaw(sfreq=sfreq, channel_data={"Signal-Sleep": sleep_signal})

    stages = sleep_utils.extract_sleep_stages(
        raw_edf=raw, bin_length_in_seconds=10, signal_name="Signal-Sleep"
    )
    assert np.array_equal(stages, np.array([1, 2, 3, 0]))


def test_create_sleep_stage_dataframe_builds_expected_columns(monkeypatch):
    monkeypatch.setattr(
        sleep_utils, "extract_sleep_stages", lambda _: np.array([0, 1, 2, 1])
    )
    monkeypatch.setattr(
        sleep_utils, "extract_activity_signal", lambda _: pd.Series([0.1, 0.2, 0.3, 0.4])
    )

    df = sleep_utils.create_sleep_stage_dataframe(edf_data="ignored")

    assert list(df.columns) == ["sleep", "start", "stop", "activity", "context"]
    assert list(df.sleep) == ["A", "R", "S", "R"]
    assert list(df.start) == [0, 5000, 10000, 15000]
    assert list(df.stop) == [5000, 10000, 15000, 20000]
    assert list(df.activity) == [0.1, 0.2, 0.3, 0.4]
    assert list(df.context) == ["ARS", "ARSR", "ARSR", "RSR"]
