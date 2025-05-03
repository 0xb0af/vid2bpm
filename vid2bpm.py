#!/usr/bin/env python3
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
import argparse
import re
import os
import matplotlib.pyplot as plt


def extract_frame_features(video_path, resize_dim=(64, 64), start_time=None, end_time=None):
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
    signal = signal - np.mean(signal)
    full = np.correlate(signal, signal, mode='full')
    ac = full[full.size // 2:]
    return ac / np.max(ac) if np.max(ac) != 0 else ac


def estimate_period(T, smooth_sigma=2, min_lag=1, max_lag=None, plot=False):
    T_smooth = gaussian_filter1d(T, sigma=smooth_sigma)
    ac = compute_autocorrelation(T_smooth)
    max_lag = len(ac) // 2 if max_lag is None else max_lag
    search = ac[min_lag:max_lag]
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
    merged = []
    for s, e, bpm in raw:
        if not merged:
            merged.append([s, e, bpm])
        else:
            ps, pe, pb = merged[-1]
            if abs(bpm - pb) <= tolerance:
                w1, w2 = pe - ps, e - s
                merged[-1][2] = (pb * w1 + bpm * w2) / (w1 + w2)
                merged[-1][1] = e
            else:
                merged.append([s, e, bpm])
    return merged


def parse_timestamp(timestamp: str) -> float:
    if not timestamp:
        raise ValueError("Empty timestamp.")
    ts = timestamp.strip()
    if ':' in ts:
        parts = [float(p) for p in ts.split(':')]
        if len(parts) == 3:
            h, m, s = parts
        elif len(parts) == 2:
            h, m, s = 0, parts[0], parts[1]
        else:
            h, m, s = 0, 0, parts[0]
        return h*3600 + m*60 + s
    pattern = r'(?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>\d+(?:\.\d+)?)s)?'
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
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    rem = seconds % 60
    s = int(rem)
    ms = int((rem - s) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def convert_video_bpm_fps(input_path: str,
                          output_path: str,
                          orig_bpm: float,
                          target_bpm: float,
                          target_fps: float):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video {input_path}")
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    speed_factor = orig_bpm / target_bpm
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    total_out_frames = int(len(frames) * speed_factor * (target_fps / orig_fps))
    for i in range(total_out_frames):
        t_out = i / target_fps
        t_in = t_out / speed_factor
        idx_f = t_in * orig_fps
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, len(frames) - 1)
        alpha = idx_f - i0
        if i0 == i1 or alpha == 0:
            writer.write(frames[i0])
            continue
        flow = cv2.calcOpticalFlowFarneback(
            gray_frames[i0], gray_frames[i1], None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = gray_frames[i0].shape
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x0 = (grid_x + flow[...,0] * alpha).astype(np.float32)
        map_y0 = (grid_y + flow[...,1] * alpha).astype(np.float32)
        map_x1 = (grid_x - flow[...,0] * (1-alpha)).astype(np.float32)
        map_y1 = (grid_y - flow[...,1] * (1-alpha)).astype(np.float32)
        warp0 = cv2.remap(frames[i0], map_x0, map_y0, interpolation=cv2.INTER_LINEAR)
        warp1 = cv2.remap(frames[i1], map_x1, map_y1, interpolation=cv2.INTER_LINEAR)
        out_frame = cv2.addWeighted(warp0, 1-alpha, warp1, alpha, 0)
        writer.write(out_frame)
    writer.release()
    print(f"Written re-sped & resampled video to {output_path}")


def slice_video_opencv(video_path, segments, output_dir, target_bpm=None, target_fps=None):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for idx, (s, e, bpm) in enumerate(segments, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, s)
        slice_path = os.path.join(output_dir, f"segment_{idx:02d}_{int(bpm)}bpm.mp4")
        writer = cv2.VideoWriter(slice_path, fourcc, fps, (width, height))
        for f in range(s, e):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
        writer.release()
        print(f"Written slice: {slice_path}")

        if target_bpm is not None:
            out_conv = os.path.join(
                output_dir,
                f"segment_{idx:02d}_{int(bpm)}bpm_to_{int(target_bpm)}bpm.mp4"
            )
            tgt_fps = target_fps or fps
            convert_video_bpm_fps(slice_path, out_conv, orig_bpm=bpm, target_bpm=target_bpm, target_fps=tgt_fps)

    cap.release()
    print(f"All slices (and conversions) written to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Detect & slice BPM sections in a video.")
    parser.add_argument("video_path")
    parser.add_argument("--resize", nargs=2, type=int, default=[64,64])
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end",   type=str, default=None)
    parser.add_argument("--window", type=str, default="5s")
    parser.add_argument("--hop",    type=str, default="1s")
    parser.add_argument("--tol",    type=float, default=2.0)
    parser.add_argument("--min-bpm", type=float, default=None)
    parser.add_argument("--max-bpm", type=float, default=None)
    parser.add_argument("--subtitles", type=str, help="Output SRT path")
    parser.add_argument("--slice-dir", type=str, help="Directory to output sliced videos with OpenCV")
    parser.add_argument("--target-bpm", type=float, help="If set, rescale each slice to this BPM")
    parser.add_argument("--target-fps", type=float, default=None, help="FPS for resampled output; defaults to original video FPS")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    start = parse_timestamp(args.start) if args.start else None
    end   = parse_timestamp(args.end)   if args.end   else None

    # Extract features only within the given time window
    features, fps = extract_frame_features(args.video_path, tuple(args.resize), start, end)
    if features.shape[0] < 2:
        print("Not enough frames in the specified range.")
        return

    # Compute relative segments within the extracted range
    w_sec = parse_timestamp(args.window)
    h_sec = parse_timestamp(args.hop)
    raw_segments = sliding_window_bpm(features, fps, w_sec, h_sec, tolerance=args.tol)

    # Offset segments back to absolute video frame indices
    frame_offset = int(start * fps) if start is not None else 0
    segments = [(s + frame_offset, e + frame_offset, bpm) for s, e, bpm in raw_segments]

    # Apply min/max BPM filters
    if args.min_bpm or args.max_bpm:
        segments = [seg for seg in segments if
                    (args.min_bpm is None or seg[2] >= args.min_bpm) and
                    (args.max_bpm is None or seg[2] <= args.max_bpm)]
    segments.sort(key=lambda x: x[1] - x[0], reverse=True)

    # Print segment times in absolute video time
    for idx, (s, e, bpm) in enumerate(segments, 1):
        start_str = format_timestamp(s / fps)
        end_str   = format_timestamp(e / fps)
        print(f"{idx}. {start_str} - {end_str} : {bpm:.1f} BPM")

    # Write subtitles if requested
    if args.subtitles:
        with open(args.subtitles, 'w') as f:
            for idx, (s, e, bpm) in enumerate(segments, 1):
                f.write(f"{idx}\n")
                f.write(f"{format_timestamp_srt(s / fps)} --> {format_timestamp_srt(e / fps)}\n")
                f.write(f"{bpm:.1f} BPM\n\n")
        print(f"Subtitles written to {args.subtitles}")

    # Slice (and optionally convert) segments
    if args.slice_dir:
        slice_video_opencv(
            args.video_path,
            segments,
            args.slice_dir,
            target_bpm=args.target_bpm,
            target_fps=args.target_fps
        )

if __name__ == "__main__":
    main()

