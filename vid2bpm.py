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
    resizes and flattens them.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    features = []
    
    # If start_time is specified, jump to that position (in milliseconds)
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
    features = np.array(features)
    return features, fps


def compute_autocorrelation(signal):
    """
    Computes the normalized autocorrelation of a 1D signal.
    """
    signal = signal - np.mean(signal)
    autocorr_full = np.correlate(signal, signal, mode='full')
    autocorr = autocorr_full[autocorr_full.size // 2:]
    if np.max(autocorr) != 0:
        autocorr /= np.max(autocorr)
    return autocorr


def estimate_period(T, smooth_sigma=2, min_lag=1, max_lag=None, plot=False):
    """
    Smooths the time series, computes its autocorrelation, and finds the lag with
    maximum autocorrelation (ignoring lag 0) as an estimate of the period.
    """
    T_smoothed = gaussian_filter1d(T, sigma=smooth_sigma)
    autocorr = compute_autocorrelation(T_smoothed)

    if max_lag is None:
        max_lag = len(autocorr) // 2
    search_range = autocorr[min_lag:max_lag]
    period = np.argmax(search_range) + min_lag

    if plot:
        lags = np.arange(len(autocorr))
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(T_smoothed)
        plt.title("Smoothed Time Series")
        plt.xlabel("Frame index")
        plt.ylabel("Difference magnitude")
        plt.subplot(1, 2, 2)
        plt.plot(lags, autocorr)
        plt.axvline(x=period, color='r', linestyle='--', label=f'Estimated period: {period}')
        plt.title("Autocorrelation")
        plt.xlabel("Lag (frames)")
        plt.ylabel("Normalized autocorrelation")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return period, autocorr


def parse_timestamp(timestamp: str) -> float:
    """
    Parse a video timestamp in various formats and return the total seconds.
    """
    if not timestamp:
        raise ValueError("Empty timestamp string.")
    timestamp = timestamp.strip()

    # Colon-separated format
    if ':' in timestamp:
        parts = [float(p) for p in timestamp.split(':')]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h = 0
            m, s = parts
        elif len(parts) == 1:
            h = 0
            m = 0
            s = parts[0]
        else:
            raise ValueError("Unsupported colon-separated timestamp format.")
        return h * 3600 + m * 60 + s

    # Suffix-based format (e.g., "1h2m3s")
    pattern = (
        r'(?:(?P<hours>\d+)\s*h)?\s*'
        r'(?:(?P<minutes>\d+)\s*m)?\s*'
        r'(?:(?P<seconds>\d+(?:\.\d+)?)\s*s)?'
    )
    match = re.fullmatch(pattern, timestamp, re.IGNORECASE)
    if match:
        h = float(match.group('hours') or 0)
        m = float(match.group('minutes') or 0)
        s = float(match.group('seconds') or 0)
        if h == 0 and m == 0 and s == 0:
            return float(timestamp)
        return h * 3600 + m * 60 + s

    # Plain numeric
    return float(timestamp)


def main():
    parser = argparse.ArgumentParser(description="Detect repetitive motion period in a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--resize", type=int, nargs=2, default=[64, 64],
                        help="Resize dimensions for processing (width height)")
    parser.add_argument("--start", type=str, default=None,
                        help="Start time (e.g., '00:10' or '10s')")
    parser.add_argument("--end", type=str, default=None,
                        help="End time (e.g., '01:00' or '60s')")
    parser.add_argument("--plot", action="store_true", help="Plot time series and autocorrelation")
    args = parser.parse_args()

    # Fix: only parse when provided
    start = parse_timestamp(args.start) if args.start else None
    end   = parse_timestamp(args.end)   if args.end   else None

    print("Extracting frame features...")
    features, fps = extract_frame_features(args.video_path, tuple(args.resize), start, end)
    if features.shape[0] < 2:
        print("Not enough frames to analyze. Check your timestamp range.")
        return
    print(f"Total frames extracted: {features.shape[0]}, FPS: {fps}")

    # Optimization: compute frame-to-frame feature differences directly
    print("Computing frame-to-frame differences...")
    T = np.linalg.norm(features[1:] - features[:-1], axis=1)

    print("Estimating period via autocorrelation...")
    period_frames, autocorr = estimate_period(T, smooth_sigma=2, min_lag=5, plot=args.plot)

    period_seconds = period_frames / fps
    frequency = fps / period_frames
    print(f"Estimated period: {period_frames} frames ({period_seconds:.2f} s)")
    print(f"Estimated frequency: {frequency:.2f} Hz ({frequency*60:.2f} BPM)")

if __name__ == "__main__":
    main()

