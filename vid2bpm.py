#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import re
import matplotlib.pyplot as plt


def extract_frame_features(video_path, resize_dim=(64, 64), start_time=None, end_time=None):
    """
    Reads video frames from a specified timestamp range, converts to grayscale,
    resizes and flattens them into normalized feature vectors.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    features = []

    if start_time is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end_time is not None and current_time > end_time:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize_dim)
        features.append(gray.flatten().astype(np.float32) / 255.0)
    cap.release()
    return np.array(features), fps


def compute_autocorrelation(signal):
    """
    Computes the normalized autocorrelation of a 1D signal.
    """
    signal = signal - np.mean(signal)
    full = np.correlate(signal, signal, mode='full')
    ac = full[full.size // 2:]
    return ac / np.max(ac) if np.max(ac) != 0 else ac


def estimate_period(T, smooth_sigma=2, min_lag=1, max_lag=None, plot=False):
    """
    Smooths the time series T, computes autocorrelation, and returns
    the lag (in frames) of the dominant peak as the period.
    """
    T_smooth = gaussian_filter1d(T, sigma=smooth_sigma)
    ac = compute_autocorrelation(T_smooth)
    max_l = len(ac) // 2 if max_lag is None else max_lag
    search = ac[min_lag:max_l]
    period = np.argmax(search) + min_lag

    if plot:
        lags = np.arange(len(ac))
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(T_smooth)
        plt.title("Smoothed Time Series")
        plt.xlabel("Frame Index")
        plt.ylabel("Difference Magnitude")
        plt.subplot(1,2,2)
        plt.plot(lags, ac)
        plt.axvline(period, linestyle='--', label=f'Period={period} frames')
        plt.title("Autocorrelation")
        plt.xlabel("Lag (frames)")
        plt.ylabel("Normalized Autocorrelation")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return period, ac


def sliding_window_bpm(features, fps, window_sec, hop_sec, tolerance=2.0, smooth_sigma=2, min_lag=5):
    """
    Splits features into overlapping windows, estimates BPM per window,
    and merges adjacent windows within a BPM tolerance.
    Returns a list of (start_frame, end_frame, bpm).
    """
    window_f = int(window_sec * fps)
    hop_f = int(hop_sec * fps)
    total = features.shape[0]
    raw = []

    for start in range(0, total - window_f + 1, hop_f):
        win = features[start:start + window_f]
        T = np.linalg.norm(win[1:] - win[:-1], axis=1)
        period_frames, _ = estimate_period(T, smooth_sigma=smooth_sigma, min_lag=min_lag)
        bpm = (fps / period_frames) * 60.0
        raw.append((start, start + window_f, bpm))

    # Merge adjacent segments with similar BPM
    merged = []
    for s, e, bpm in raw:
        if not merged:
            merged.append([s, e, bpm])
        else:
            ps, pe, pb = merged[-1]
            if abs(bpm - pb) <= tolerance:
                w1 = pe - ps
                w2 = e - s
                merged[-1][2] = (pb * w1 + bpm * w2) / (w1 + w2)
                merged[-1][1] = e
            else:
                merged.append([s, e, bpm])
    return merged


def parse_timestamp(timestamp: str) -> float:
    """
    Converts 'HH:MM:SS', 'MM:SS', '1h2m3s', or plain seconds into float seconds.
    """
    if not timestamp:
        raise ValueError("Empty timestamp.")
    ts = timestamp.strip()
    if ':' in ts:
        parts = [float(p) for p in ts.split(':')]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0; m, s = parts
        elif len(parts) == 1:
            h = 0; m = 0; s = parts[0]
        else:
            raise ValueError("Bad timestamp format.")
        return h*3600 + m*60 + s
    pattern = (
        r'(?:(?P<h>\d+)h)?'
        r'(?:(?P<m>\d+)m)?'
        r'(?:(?P<s>\d+(?:\.\d+)?)s)?'
    )
    m = re.fullmatch(pattern, ts)
    if m:
        h = float(m.group('h') or 0)
        m_ = float(m.group('m') or 0)
        s = float(m.group('s') or 0)
        if h==m_==s==0:
            return float(ts)
        return h*3600 + m_*60 + s
    return float(ts)


def format_timestamp(seconds: float) -> str:
    """
    Formats a float number of seconds into HH:MM:SS string.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Detect variable BPM sections in a video.")
    parser.add_argument("video_path", help="Input video file path")
    parser.add_argument("--resize", nargs=2, type=int, default=[64,64],
                        help="Resize dimensions (width height)")
    parser.add_argument("--start", type=str, default=None, help="Start time (e.g. '00:10')")
    parser.add_argument("--end",   type=str, default=None, help="End time (e.g. '01:00')")
    parser.add_argument("--window", type=str, default="5s",
                        help="Window length for BPM estimation (default: 5s)")
    parser.add_argument("--hop",    type=str, default="1s",
                        help="Hop length between windows (default: 1s)")
    parser.add_argument("--tol",    type=float, default=2.0,
                        help="Merge tolerance in BPM (default: 2.0)")
    parser.add_argument("--min-bpm", type=float, default=None,
                        help="Minimum BPM to include in output")
    parser.add_argument("--max-bpm", type=float, default=None,
                        help="Maximum BPM to include in output")
    parser.add_argument("--plot", action="store_true", help="Show intermediate plots")
    args = parser.parse_args()

    start = parse_timestamp(args.start) if args.start else None
    end   = parse_timestamp(args.end)   if args.end   else None

    print("Extracting frames...")
    features, fps = extract_frame_features(args.video_path, tuple(args.resize), start, end)
    n_frames = features.shape[0]
    if n_frames < 2:
        print("Not enough frames.")
        return
    print(f"Got {n_frames} frames @ {fps:.2f} FPS")

    w_sec = parse_timestamp(args.window)
    h_sec = parse_timestamp(args.hop)
    print(f"Sliding-window: {w_sec}s window, {h_sec}s hop, tol={args.tol} BPM")
    segments = sliding_window_bpm(features, fps, w_sec, h_sec, tolerance=args.tol)

    # Filter by BPM range if specified
    if args.min_bpm is not None or args.max_bpm is not None:
        filtered = []
        for s, e, bpm in segments:
            if args.min_bpm is not None and bpm < args.min_bpm:
                continue
            if args.max_bpm is not None and bpm > args.max_bpm:
                continue
            filtered.append((s, e, bpm))
        segments = filtered

    # Sort by segment duration descending
    segments.sort(key=lambda x: (x[1] - x[0]), reverse=True)

    print("Detected segments:")
    for s, e, bpm in segments:
        start_str = format_timestamp(s / fps)
        end_str = format_timestamp(e / fps)
        duration = e - s
        print(f"{start_str} - {end_str} ({duration/fps:.2f}s): {bpm:.1f} BPM")

if __name__ == "__main__":
    main()

