import numpy as np
import neurokit2 as nk


# -----------------------------
# R-peaks + HR
# -----------------------------
def detect_rpeaks(ecg: np.ndarray, fs: float) -> np.ndarray:
    """R-peak indices (samples). Returns empty array on failure."""
    try:
        _, info = nk.ecg_peaks(ecg, sampling_rate=fs)
        return np.array(info["ECG_R_Peaks"], dtype=int)
    except Exception:
        return np.array([], dtype=int)


def heart_rate(rpeaks: np.ndarray, fs: float) -> np.ndarray:
    """Instantaneous HR (bpm) aligned to R-peaks (first value repeated)."""
    if len(rpeaks) < 2:
        return np.zeros(len(rpeaks))
    rr = np.diff(rpeaks) / fs
    hr = 60.0 / np.clip(rr, 1e-3, None)
    return np.concatenate([[hr[0]], hr])


# -----------------------------
# Delineation & beats
# -----------------------------
def delineate_waves(ecg: np.ndarray, rpeaks: np.ndarray, fs: float, method: str = "dwt") -> Dict[str, np.ndarray]:
    """
    P/QRS/T delineation using NeuroKit2.
    Returns a dict with keys like:
      'ECG_P_Peaks', 'ECG_T_Peaks',
      'ECG_R_Onsets', 'ECG_R_Offsets',
      'ECG_P_Onsets', 'ECG_T_Offsets', ...
    Missing keys are handled as empty arrays.
    """
    try:
        _, waves = nk.ecg_delineate(ecg, rpeaks, sampling_rate=fs, method=method, show=False)
        # Ensure all keys exist with np.array
        out = {}
        for k, v in waves.items():
            out[k] = np.asarray(v if v is not None else [], dtype=int)
        return out
    except Exception:
        return {}


def segment_beats(
    ecg: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
    window: Tuple[float, float] = (-0.20, 0.40),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract fixed windows around each R-peak.
    Returns:
      beats: (n_beats, n_samples)
      mask:  (n_beats,) bool indicating segments fully inside signal bounds
    """
    pre, post = int(abs(window[0]) * fs), int(window[1] * fs)
    n = len(ecg)
    beats, mask = [], []

    for r in rpeaks:
        a = r - pre
        b = r + post
        if a >= 0 and b <= n:
            beats.append(ecg[a:b])
            mask.append(True)
        else:
            # keep length consistent by skipping partial windows
            mask.append(False)

    if not beats:
        return np.zeros((0, pre + post)), np.array(mask, dtype=bool)
    return np.vstack(beats), np.array(mask, dtype=bool)


# -----------------------------
# Morphology features (per beat)
# -----------------------------
def _nearest_index(arr: np.ndarray, target: int) -> Optional[int]:
    if arr.size == 0:
        return None
    idx = int(np.argmin(np.abs(arr - target)))
    return int(arr[idx])


def morphology_from_waves(
    ecg: np.ndarray,
    rpeaks: np.ndarray,
    waves: Dict[str, np.ndarray],
    fs: float
) -> Dict[str, np.ndarray]:
    """
    Compute simple per-beat morphology & interval features using delineation:
      - qrs_width (ms) from R_onset to R_offset
      - r_amp, q_amp, s_amp (mV units of input)
      - p_amp, t_amp (if peaks found)
      - pr_interval, qt_interval (ms) when onsets/offsets available

    Returns a dict of arrays aligned to rpeaks. Missing values = np.nan.
    """
    n = len(rpeaks)
    qrs_w = np.full(n, np.nan)
    r_amp = np.full(n, np.nan)
    q_amp = np.full(n, np.nan)
    s_amp = np.full(n, np.nan)
    p_amp = np.full(n, np.nan)
    t_amp = np.full(n, np.nan)
    pr_int = np.full(n, np.nan)
    qt_int = np.full(n, np.nan)

    r_on = np.asarray(waves.get("ECG_R_Onsets", []), dtype=int)
    r_off = np.asarray(waves.get("ECG_R_Offsets", []), dtype=int)
    p_peaks = np.asarray(waves.get("ECG_P_Peaks", []), dtype=int)
    t_peaks = np.asarray(waves.get("ECG_T_Peaks", []), dtype=int)
    p_on = np.asarray(waves.get("ECG_P_Onsets", []), dtype=int)
    t_off = np.asarray(waves.get("ECG_T_Offsets", []), dtype=int)

    for i, r in enumerate(rpeaks):
        # amplitudes
        r_amp[i] = float(ecg[r]) if 0 <= r < len(ecg) else np.nan

        # Q ~ nearest onset before R; S ~ nearest offset after R
        q_idx = _nearest_index(r_on[r_on <= r], r)
        s_idx = _nearest_index(r_off[r_off >= r], r)
        if q_idx is not None and 0 <= q_idx < len(ecg):
            q_amp[i] = float(ecg[q_idx])
        if s_idx is not None and 0 <= s_idx < len(ecg):
            s_amp[i] = float(ecg[s_idx])

        # QRS width
        if q_idx is not None and s_idx is not None:
            qrs_w[i] = 1000.0 * (s_idx - q_idx) / fs  # ms

        # P and T peak amplitudes (nearest within a window around R)
        p_idx = _nearest_index(p_peaks[p_peaks <= r], r)
        if p_idx is not None:
            p_amp[i] = float(ecg[p_idx])

        t_idx = _nearest_index(t_peaks[t_peaks >= r], r)
        if t_idx is not None:
            t_amp[i] = float(ecg[t_idx])

        # PR & QT intervals if onsets/offsets exist around R
        p_on_idx = _nearest_index(p_on[p_on <= r], r)
        t_off_idx = _nearest_index(t_off[t_off >= r], r)
        if p_on_idx is not None and q_idx is not None:
            pr_int[i] = 1000.0 * (q_idx - p_on_idx) / fs
        if q_idx is not None and t_off_idx is not None:
            qt_int[i] = 1000.0 * (t_off_idx - q_idx) / fs

    return {
        "qrs_width_ms": qrs_w,
        "r_amp": r_amp,
        "q_amp": q_amp,
        "s_amp": s_amp,
        "p_amp": p_amp,
        "t_amp": t_amp,
        "pr_interval_ms": pr_int,
        "qt_interval_ms": qt_int,
    }


# -----------------------------
# HRV features (from R-peaks)
# -----------------------------
def hrv_features(rpeaks: np.ndarray, fs: float) -> Dict[str, float]:
    """
    Time & frequency HRV summary using NeuroKit2.
    Returns common metrics; if not enough beats, returns empty dict.
    """
    if len(rpeaks) < 3:
        return {}

    try:
        # Build an R-peak series for nk.hrv
        # Use a dataframe with "ECG_R_Peaks" and sampling rate
        signal = np.zeros(rpeaks[-1] + 1, dtype=int)
        signal[rpeaks] = 1

        # nk.hrv() expects a DataFrame of peaks; use ecg_process then hrv_time/frequency
        _, info = nk.ecg_peaks(signal, sampling_rate=fs)
        rr_df = nk.hrv_time(info, sampling_rate=fs, show=False)
        fr_df = nk.hrv_frequency(info, sampling_rate=fs, show=False)

        out = {}
        for col in ("HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50"):
            if col in rr_df.columns:
                out[col] = float(rr_df[col].iloc[0])
        for col in ("HRV_LF", "HRV_HF", "HRV_LFHF"):
            if col in fr_df.columns:
                out[col] = float(fr_df[col].iloc[0])
        return out
    except Exception:
        return {}
