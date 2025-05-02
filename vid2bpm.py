#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import re

def extract_frame_features(video_path, resize_dim=(64, 64), start_time=None, end_time=None):
    """
    Reads video frames from a specified timestamp range, converts to grayscale,
    resizes and flattens them.
    
    Parameters:
    - video_path: path to the input video file.
    - resize_dim: tuple (width, height) to resize frames.
    - start_time: start time in seconds (optional).
    - end_time: end time in seconds (optional).
    
    Returns:
    - features: array of flattened frame features.
    - fps: frames per second of the video.
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
        # Get the current timestamp (in seconds)
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if end_time is not None and current_time > end_time:
            break
        # Process frame: convert to grayscale, resize, flatten, and normalize.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, resize_dim)
        features.append(gray.flatten().astype(np.float32) / 255.0)
    cap.release()
    features = np.array(features)
    return features, fps

def compute_distance_matrix(features):
    """
    Compute the full cosine distance matrix between frame features.
    """
    M = pairwise_distances(features, metric='cosine')
    return M

def time_series_from_distance_matrix(M):
    """
    Create a 1D time series from the distance matrix by computing the Euclidean 
    distance between consecutive rows.
    """
    N = M.shape[0]
    T = np.zeros(N - 1)
    for i in range(N - 1):
        T[i] = np.linalg.norm(M[i, :] - M[i+1, :])
    return T

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
    
    Supported formats:
      - Colon-separated: "HH:MM:SS", "MM:SS", or even "SS"
      - Suffix-based: "1h 2m 3s", "2m3s", "45s", etc.
      - Plain number: "90" (interpreted as seconds)
    
    Args:
        timestamp (str): The timestamp string.
    
    Returns:
        float: Total seconds.
    
    Raises:
        ValueError: If the timestamp format is unsupported.
    """
    timestamp = timestamp.strip()
    if not timestamp:
        raise ValueError("Empty timestamp string.")
    
    # Case 1: Colon-separated format.
    if ':' in timestamp:
        parts = timestamp.split(':')
        try:
            # Convert each part to a float (to support fractional seconds)
            parts = [float(part) for part in parts]
        except ValueError:
            raise ValueError("Invalid number in timestamp.")
        
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = parts[0]
        else:
            raise ValueError("Unsupported colon-separated timestamp format.")
        return hours * 3600 + minutes * 60 + seconds

    # Case 2: Suffix-based format (e.g., "1h 2m 3s")
    # This regex captures optional hours, minutes, and seconds.
    pattern = (
        r'(?:(?P<hours>\d+)\s*h(?:ours?)?)?\s*'
        r'(?:(?P<minutes>\d+)\s*m(?:in(?:utes?)?)?)?\s*'
        r'(?:(?P<seconds>\d+(?:\.\d+)?)\s*s(?:ec(?:onds?)?)?)?'
    )
    match = re.fullmatch(pattern, timestamp, re.IGNORECASE)
    if match:
        hours = float(match.group("hours")) if match.group("hours") else 0.0
        minutes = float(match.group("minutes")) if match.group("minutes") else 0.0
        seconds = float(match.group("seconds")) if match.group("seconds") else 0.0
        # If nothing was captured, fall back to try a plain float conversion.
        if hours == 0 and minutes == 0 and seconds == 0:
            try:
                return float(timestamp)
            except ValueError:
                pass
        return hours * 3600 + minutes * 60 + seconds

    # Case 3: Fallback to plain numeric conversion.
    try:
        return float(timestamp)
    except ValueError:
        raise ValueError("Unsupported timestamp format.")


def main():
    parser = argparse.ArgumentParser(description="Detect repetitive motion period in a video.")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--resize", type=int, nargs=2, default=[64, 64],
                        help="Resize dimensions for processing (width height), default is 64x64")
    parser.add_argument("--start", type=str, default=None,
                        help="Start time in seconds from which to extract frames")
    parser.add_argument("--end", type=str, default=None,
                        help="End time in seconds to stop extracting frames")
    parser.add_argument("--plot", action="store_true", help="Plot time series and autocorrelation")
    args = parser.parse_args()
    
    start = parse_timestamp(args.start)
    end = parse_timestamp(args.end)
    # Extract frame features from the specified timestamp range
    print("Extracting frame features...")
    features, fps = extract_frame_features(args.video_path, tuple(args.resize), start, end)
    if features.shape[0] < 2:
        print("Not enough frames to analyze. Check your timestamp range.")
        return
    print(f"Total frames extracted: {features.shape[0]}, FPS: {fps}")
    
    # Compute the distance matrix
    print("Computing distance matrix...")
    M = compute_distance_matrix(features)
    
    # Create the 1D time series from the distance matrix
    print("Creating time series from distance matrix...")
    T = time_series_from_distance_matrix(M)
    
    # Estimate the period using autocorrelation
    print("Computing autocorrelation to estimate period...")
    period_frames, autocorr = estimate_period(T, smooth_sigma=2, plot=args.plot, min_lag=5)
    
    period_seconds = period_frames / fps
    frequency = fps / period_frames  # repetitions per second
    print(f"Estimated period: {period_frames} frames")
    print(f"Estimated period: {period_seconds:.2f} seconds")
    print(f"Estimated frequency: {frequency:.2f} Hz")
    print(f"Estimated BPM: {frequency*60:.2f} BPM")
    
if __name__ == "__main__":
    main()

