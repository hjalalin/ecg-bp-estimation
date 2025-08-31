import numpy as np

def abp_from_rpeaks(abp, rpeaks, fs, sbp_win, dbp_win):
    sbp_list, dbp_list, mbp_list = [], [], []

    for i, r in enumerate(rpeaks):
        r_next = rpeaks[i + 1] if i + 1 < len(rpeaks) else len(abp) - 1

        # systolic max
        a = int(r + sbp_win[0] * fs)
        b = int(r + sbp_win[1] * fs)
        sbp = np.max(abp[max(0, a):min(r_next, b)]) if b > a else np.nan

        # diastolic min
        a = int(r + dbp_win[0] * fs)
        b = int(r + dbp_win[1] * fs)
        dbp = np.min(abp[max(0, a):min(r_next, b)]) if b > a else np.nan

        mbp = dbp + (sbp - dbp) / 3 if (not np.isnan(sbp) and not np.isnan(dbp)) else np.nan

        sbp_list.append(sbp)
        dbp_list.append(dbp)
        mbp_list.append(mbp)

    return np.array(sbp_list), np.array(dbp_list), np.array(mbp_list)