# vid2bpm

- slapped-together vibe coding
- may contain bugs and rough edges

## overview

* analyzes a video file frame by frame (no audio analysis)
* uses simple autocorrelation on grayscale pixel features to guess the bpm
* outputs a list of time segments with estimated constant bpm

## features

* sliding-window segmentation (`--window`, `--hop`, `--tol`) for varying bpm
* timestamp parsing in hh\:mm\:ss, mm\:ss, 1h2m3s, or plain seconds
* filter by min/max bpm (`--min-bpm`, `--max-bpm`)
* prints segments sorted by longest duration first
* optional srt subtitle export (`--subtitles`) showing bpm on screen
* optional video slicing with audio (`--slice-dir`) via moviepy
* debug plots (`--plot`) of smoothed signal and autocorrelation


## usage examples

steady music track between 80–120 BPM, full analysis:

```bash
./vid2bpm.py path/to/video.mp4
```

(defaults: window=5s, hop=1s, tol=2.0)

custom range and subtitles:

```bash
./vid2bpm.py music.mp4 --min-bpm 80 --max-bpm 120 --subtitles bpm.srt
```

slice into segments with audio for a dance clip:

```bash
./vid2bpm.py dance.mp4 \
  --window 3s --hop 0.5s --tol 4.0 \
  --slice-dir dance_segments/
```

full example combining filters, export, and plotting:

```bash
./vid2bpm.py highlights.mp4 \
  --window 8s --hop 2s --tol 5.0 \
  --min-bpm 60 --max-bpm 200 \
  --subtitles highlights.srt --slice-dir highlights/ \
  --plot
```

## choosing window/hop/tolerance

acceptable values depend on content and expected tempo:

* **steady performance (e.g., workout tutorial):**

  * window: 10s, hop: 2s, tol: 3.0 BPM
  * large window smooths over slight frame noise, small hop keeps updates frequent

* **music video with occasional breaks:**

  * window: 5s, hop: 1s, tol: 2.0 BPM
  * shorter window catches tempo changes at verse/chorus boundaries

* **dance performance (variable tempo):**

  * window: 3s, hop: 0.5s, tol: 4.0 BPM
  * smaller window/hop to track quick shifts, higher tol to merge similar bursts

* **sports or fast-cut montage:**

  * window: 8s, hop: 2s, tol: 5.0 BPM
  * moderate window handles rapid scene changes; wider tol accounts for jitter

start with these presets and tweak if output is too noisy (fragmented) or too coarse (misses changes).

## limitations

* bpm detection based solely on video frames; no audio considered
* may fail on shaky, low-contrast, or variable lighting footage
* uses naive autocorrelation, not advanced beat-tracking models
* timestamp parsing and slicing logic can break on edge cases
* performance will suffer on long or high-resolution videos
* minimal error handling and user feedback

