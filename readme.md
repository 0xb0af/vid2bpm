# BPM Slicer

A command-line tool written in python to read a local video file and analyze per-frame differences to estimate local periodic motion using autocorrelation. The app detects constant-bpm (beats per minute) sections in a video and can optionally slice and time-scale them too. 

This program _does not_ read or process audio in any way. The "beat" is determined by looking for repetitive periodic motion in the video.

This app was written almost entirely using vibe coding with chatGPT, based on this research paper. There will likely be bugs.

Adhikari, Slesa, "Detecting periodic action patterns in videos" (2020). Theses. 323.
https://louis.uah.edu/uah-theses/323 

I have no affiliation with the author.

## Features

* **Slice Segments**: Extract detected BPM-consistent segments into separate video files with `--slice-dir`.
* **Time-Scaling (Respeed)**: Rescale each slice from its detected BPM to a **target BPM** using optical-flow frame interpolation. Adjust output framerate with `--target-fps`.
* **Subtitle Export**: Export BPM annotations as an SRT subtitle file with `--subtitles`.

## Installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
./bpm_slicer.py <video_path> [OPTIONS]
```

### Options

* `--start TIME`        : Begin BPM detection at this timestamp (e.g. `00:01:30`, `90s`, `1m30s`).
* `--end TIME`          : End BPM detection at this timestamp.
* `--min-bpm FLOAT`     : Discard any segments slower than this BPM.
* `--max-bpm FLOAT`     : Discard any segments faster than this BPM.
* `--slice-dir DIR`     : Directory to write out each detected segment as its own MP4.
* `--target-bpm FLOAT`  : If set with `--slice-dir`, rescale slices from their detected BPM to this target BPM.
* `--target-fps FLOAT`  : Frame rate for rescaled slices (default: original video FPS).
* `--window TIME`       : (Advanced) Length of sliding window for local BPM analysis (default: `5s`).
* `--hop TIME`          : (Advanced) Step size between windows (default: `1s`).
* `--plot`              : Show diagnostic plots of the autocorrelation and smoothed signal.
* `--subtitles FILE`    : Write an SRT subtitle file of BPM labels for each segment.
* `--resize W H`        : Resize each frame to `W×H` before motion analysis (default: `64 64`). **This affects only analysis**, not output dimensions.

## Examples

1. **Basic BPM Detection**

   ```bash
   ./bpm_slicer.py myvideo.mp4
   ```

2. **Restrict to a Specific Time Range**

   ```bash
   ./bpm_slicer.py myvideo.mp4 --start 30s --end 2m
   ```
   
3. **Extract and Slice Segments**
   **What happens:** Detects constant-BPM sections and writes each one as a separate MP4 file in `segments/`.

   ```bash
   ./bpm_slicer.py myvideo.mp4 --slice-dir segments
   ```
   
4. **Smooth and Fine-Tune Analysis**
   **What happens:** Uses a 10‑second sliding window that hops every 2 seconds. This trades off time resolution versus BPM stability: longer windows give a more reliable BPM estimate, shorter hops catch tempo changes more precisely.

   ```bash
   ./bpm_slicer.py myvideo.mp4 --slice-dir segments --start 30s --end 2m --window 10s --hop 2s
   ```

5. **Rescale Slices to 90 BPM**
   **What happens:** After slicing, each segment at its detected BPM is time-scaled faster or slower so its playback matches exactly 90 BPM. Video framerate remains constant. Optical-flow interpolation preserves smooth motion when speeding up or slowing down.

   ```bash
   ./bpm_slicer.py myvideo.mp4 --slice-dir segments --start 30s --end 2m --window 10s --hop 2s --target-bpm 90
   ```

6. **Adjust Output Frame Rate**
   **What happens:** Same as above, but forces the output slices to a different FPS.

   ```bash
   ./bpm_slicer.py myvideo.mp4 --slice-dir segments --start 30s --end 2m --window 10s --hop 2s --target-bpm 90 --target-fps 30
   ```

## How the algorithm works

1. **Motion Analysis**: The script converts each frame to grayscale and downsamples it (via `--resize`) to focus on motion, not detail.
2. **Windowed BPM Estimation**: It measures frame‑to‑frame difference magnitude within short windows. Autocorrelation of this signal reveals the dominant repetition period (beat).
3. **Merging Nearby Beats**: Adjacent windows with very similar BPM (within `--tol`) are merged into longer segments.
4. **Slice Output**: Each merged segment is extracted from the original video and saved as its own MP4. If `--target-bpm` is specified, those clips are then time-scaled to match the desired tempo.

*BPM Slicer* combines these steps into a single CLI for rapid rhythm-based video editing.

