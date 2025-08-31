import numpy as np
from scipy.signal import butter, filtfilt, resample_poly

def bandpass_filter(signal, fs, low, high, order = 4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def lowpass_filter(signal: np.ndarray, fs: float, cutoff: float, order: int = 4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype="low")
    return filtfilt(b, a, signal)


def resample_to(signal: np.ndarray, fs_in: float, fs_out: float):
    if fs_in == fs_out:
        return signal
    g = np.gcd(int(round(fs_in)), int(round(fs_out)))
    up = int(round(fs_out)) // g
    down = int(round(fs_in)) // g
    return resample_poly(signal, up, down)