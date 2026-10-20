# SwimVision

**Turn ordinary swim race footage into measurable performance data.**

> Built for the 2026 Congressional App Challenge.
> **Status: Phase 1 done; Phase 2 uses a hybrid persistent tracker and is being validated on real footage.** No stroke or race metrics exist yet. See the [Roadmap](#roadmap).

## Problem

Competitive swimmers often have videos of their races, but a normal video only shows *what* happened. To understand *why* time was gained or lost, a swimmer or coach has to replay footage by hand and manually measure strokes, breakouts, turns and splits. Professional sports-analysis systems that automate this are expensive or inaccessible.

## Solution

SwimVision lets a swimmer upload an ordinary race video and uses computer vision to measure the race. Every number shown comes from the video, from user input, or from a calculation the app performs. **Nothing is hard-coded or simulated.**

## Features

| Feature | Status |
|---|---|
| Race setup: every individual pool event - free, back, breast, fly, IM - in SCY / SCM / LCM, with lengths, turns and IM legs | ✅ Phase 1 (metadata only) |
| Video upload (MP4 / MOV / AVI / M4V / MKV), real FPS / frames / duration / resolution | ✅ Phase 1 |
| Choose the swimmer visually with one rectangle—no frame numbers or coordinates | ✅ Phase 2 |
| Hybrid YOLO + swimmer U-Net detection with persistent identity and gap prediction | ✅ Phase 2 (validating) |
| Annotated tracking video + per-frame CSV | ✅ Phase 2 |
| Pose, strokes, events, dashboard, calibration, comparison, AI coach, validation | Phases 3-10 |

## How Phase 2 works

1. **Choose.** Move a time slider until the swimmer is visible, drag the yellow box around them, and click **Use this swimmer**. The app reads the time, frame and pixel coordinates itself. This is the only manual selection.
2. **Track.** Click **Track my swimmer**. YOLO finds people on the block and outside the water; the swimmer U-Net specializes in the in-water part. The tracker remembers the selected swimmer's visual appearance and automatically creates a hard corridor around that lane. It combines those identity signals with position, motion, detector source and confidence.
3. **Bridge gaps.** During a short splash or underwater gap, the tracker uses a slowing motion prediction rather than switching to the highest-confidence person. After a longer gap it requires the same candidate in consecutive frames before reacquiring. If identity is uncertain, it reports searching instead of jumping to another swimmer.
4. **Report.** The result shows followed/lost frames, every lost interval, an annotated video and a per-frame CSV. Green boxes are detector observations; yellow boxes are short predictions through a gap.

Timing always comes from frame index ÷ the video's real FPS. Videos above 30 fps are analysed every *n*-th frame (60 fps -> every 2nd); timestamps stay exact.

## Architecture

```mermaid
flowchart LR
    U[User] --> UP[ui/upload.py<br/>race details, upload, metadata]
    UP --> V[core/video.py<br/>metadata, frames, H.264 writer]
    U --> SEL[ui/selection.py<br/>time slider + one visual rectangle]
    SEL --> G[core/geometry.py<br/>display px -> original px]
    SEL --> TR[ui/tracking.py<br/>run + results]
    TR --> H[core/hybrid_tracking.py<br/>YOLO + swimmer U-Net candidates]
    H --> P[core/swimmer_tracker.py<br/>identity + motion + lane scoring]
    P --> O[annotated video]
    P --> ST[(storage/analyses.py<br/>local JSON + CSV)]
    ST --> TR
```

**Design rules**

- `core/` has no Streamlit code, so it can be tested and run from scripts.
- All timing comes from frame index and real FPS.
- Expected failures raise `SwimVisionError` subclasses with a readable message, including the underlying detector/PyTorch error when tracking fails.
- Developer mode: start with `SWIMVISION_DEBUG=1` to see full tracebacks in the page; they are also written to `outputs/logs/swimvision.log`. Without it, the error type and message are shown and the traceback goes to the terminal.

## Technology stack

Python 3.12, Streamlit, streamlit-cropper, OpenCV, Ultralytics YOLO, PyTorch, NumPy, Pandas, FFmpeg (bundled via `imageio-ffmpeg`), pytest.

**Python version:** use **3.12**. Python 3.14 is not recommended because PyTorch/Ultralytics wheels lag behind.

## Installation

Windows PowerShell:

```powershell
cd SwimVision
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

macOS / Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

No separate FFmpeg install is needed.

### GPU acceleration (strongly recommended with an NVIDIA GPU)

`pip install` gives the CPU build of PyTorch, which works but is slower. With an NVIDIA GPU, install the CUDA build *before* the requirements:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-dev.txt
```

The device used is shown under *Run details* after tracking.

### First run needs internet (once)

The small YOLO person-detector weights are downloaded by Ultralytics the first time tracking runs. The swimmer U-Net weights are included in `models/`. **Your video is never uploaded.**

## Running locally

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open <http://localhost:8501>. Developer mode (PowerShell): `$env:SWIMVISION_DEBUG="1"` before the command.

In the app: upload the video, move the time slider, draw one box around the swimmer, click **Use this swimmer**, then click **Track my swimmer**. There are no frame numbers, pixel coordinates or lane boundaries to enter.

Tests: `.\.venv\Scripts\python.exe -m pytest`

## Current limitations

- **No swimming metrics yet.** Phase 2 only locates the swimmer.
- Hybrid tracking is still being validated across different pool angles, lighting and camera motion.
- Long camera cutaways can outlast identity memory. The tracker reports those frames as lost instead of silently choosing a different person.
- Watch the annotated video once to confirm that the box stayed on the intended swimmer.
- Positions are in **pixels**, not metres/yards (needs calibration, later phase).
- Non-MP4/H.264 uploads (MOV, AVI, HEVC) are shown through a re-encoded preview; the original file is always analysed.
- Variable-frame-rate videos use the container's average FPS. Upload limit is 1 GB (`.streamlit/config.toml`).

## Privacy

- **All video processing happens locally on your computer.**
- Videos are stored in `uploads/`, which is git-ignored. Unsaved uploads older than one hour are deleted automatically at start-up.
- The only network use is the one-time download of the model file.
- A later optional AI Coach will receive only calculated numbers, never video or frames.
- Never commit swim videos (especially of children). `.gitignore` blocks common video extensions and the `uploads/`, `outputs/`, `data/` and `models/` contents.

## AI use disclosure

Generative AI was used to help write this project. See [AI_USAGE.md](AI_USAGE.md) for specifics.

## Roadmap

1. **Phase 1 ✅** Setup, upload, metadata, playback
2. **Phase 2 (validating)** One visual selection, hybrid persistent tracking, annotated video, honest loss reporting
3. Phase 3: Pose estimation
4. Phase 4: Stroke signal, stroke count, stroke rate
5. Phase 5: Start / breakout / finish (hybrid auto + manual)
6. Phase 6: Dashboard, clips, overlay video
7. Phase 7: Turns, distance calibration
8. Phase 8: Race comparison
9. Phase 9: AI Coach
10. Phase 10: Validation, performance tuning, documentation
