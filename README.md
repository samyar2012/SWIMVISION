# SwimVision

## Congressional App Challenge judge guide

SwimVision turns an ordinary swimming-race video into a reviewable performance report. A swimmer uploads a video, identifies themselves with one click, and the app follows that athlete through the race. It then presents race events, length splits when the walls are visible, stroke-rhythm estimates, visible technique signals, practice ideas, and a race-specific coaching chat.

The project is a working local prototype. It does not upload race footage to a cloud service, require an account, or require an API key.

## What judges can evaluate

- A React interface designed for swimmers and families rather than developers.
- One-click swimmer selection with no frame numbers, coordinates, or manually entered tracking intervals.
- Persistent tracking through the block, dive, splash, underwater phase, surface swimming, turn, and finish.
- Support for valid 25, 50, 100, and 200 yard or metre events.
- Evidence-labelled takeoff, water-entry, turn, breakout, and finish estimates.
- Per-length splits when every required wall is visible.
- Stroke-cycle rhythm and visible-technique analysis with confidence and limitations.
- Plain-language strengths, improvement priorities, training-set examples, and general recovery guidance.
- A conversational coach that answers questions using the measurements from the analyzed race.
- Local processing and private storage of uploaded videos.

## Fastest judge setup

### Requirements

- Windows 10 or 11, macOS, or Linux
- Python 3.12
- Node.js 20.19 or newer
- Git
- Approximately 4 GB of free disk space for dependencies and generated files
- An NVIDIA GPU is optional but makes analysis much faster

Python 3.14 is not recommended because PyTorch and Ultralytics packages may not provide compatible wheels.

### Windows PowerShell

Run these commands from the folder where you want the project:

```powershell
git clone https://github.com/samyar2012/SWIMVISION.git
cd SWIMVISION

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

cd frontend
npm ci
npm run build
cd ..

.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep that terminal open and visit [http://127.0.0.1:8000](http://127.0.0.1:8000).

### macOS or Linux

```bash
git clone https://github.com/samyar2012/SWIMVISION.git
cd SWIMVISION

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

cd frontend
npm ci
npm run build
cd ..

.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Keep that terminal open and visit [http://127.0.0.1:8000](http://127.0.0.1:8000).

No `.env` file or API key is needed for the judge demonstration.

## Recommended judge demonstration

The repository includes the complete 54-second race used by the full tracking regression:

```text
tests/fixtures/swimmer_video2_full.mp4
```

Use these settings:

| Field | Value |
|---|---|
| Video | `tests/fixtures/swimmer_video2_full.mp4` |
| Stroke | Freestyle |
| Distance | 50 |
| Pool | SCY |
| Official time | Leave blank, or enter the known race time if desired |

Then complete the following steps:

1. Select the video or drag it into the upload card.
2. Choose **Freestyle**, **50**, and **SCY**.
3. Click **Continue**.
4. On **Which swimmer are you?**, click the swimmer in the upper middle of the opening frame, directly behind the center starting block and wearing a dark cap.
5. Click **That’s me** once.
6. Leave the browser open while SwimVision follows the race and builds the report.
7. Review the five result tabs described below.

The analysis runs neural-network inference over the video. Runtime depends heavily on the computer. A CUDA-capable NVIDIA GPU is much faster than CPU-only processing.

## Five-minute judge walkthrough

### 1 Overview

Start with the headline and the clearest takeaway. The page explains what went well, what the strongest improvement opportunity is, and what evidence supports each statement. The next-practice card turns one measured finding into an actionable set.

### 2 Technique

Open **Technique** to review what the camera can support:

- stroke-timing consistency;
- vertical body-path consistency;
- first-to-final-length cycle economy;
- turn-to-breakout consistency; and
- finish execution.

The app deliberately marks catch angle, elbow position, breathing mechanics, and kick mechanics as not measured when the video does not contain validated anatomical keypoints. This prevents a polished interface from presenting guesses as biomechanics.

### 3 Race

Open **Race** to compare length splits, pace notes, and provisional stroke cycles. SwimVision expects the correct number of turns for the selected course and distance. If every wall is not visible, it withholds the affected splits instead of dividing the race evenly and inventing data.

### 4 Practice plan

Open **Practice plan** for training examples tied to the report. These are practice ideas, not medical prescriptions. Food and recovery suggestions are general and food-first; they do not prescribe calories, weight change, supplements, or a medical diet.

### 5 Ask coach

Open **Ask coach** and choose a suggested question or ask one of these:

```text
What should I focus on at my next practice?
Where did I lose the most time?
How consistent was my stroke rhythm?
What can this video not measure reliably?
```

The coach uses the current race report. When the race does not contain enough evidence, it is designed to say that rather than invent an answer.

## How the app works

```text
Upload race video
        ↓
Choose race details
        ↓
Click the swimmer once
        ↓
Track the same visual identity
        ↓
Hand off from deck-person evidence to in-water swimmer evidence
        ↓
Build an evidence-labelled race timeline
        ↓
Present findings, technique signals, splits, practice ideas, and coach chat
```

### Tracking

Before water entry, a YOLO person detector and OpenCV ViTTrack preserve the athlete selected by the user. In the water, a custom swimmer U-Net heatmap supplies swimming-specific observations. Motion continuity, lane consistency, camera-motion compensation, confidence gates, and persistent identity rules help bridge splash and partial submersion. Brief predictions are labelled as estimates. The tracker must not silently replace the selected athlete with a person in another lane.

### Race analysis

All timing comes from frame index divided by the video's measured frame rate. Event metadata determines the required number of lengths and turns. The analysis uses the verified track and swimmer-only crops to estimate:

- takeoff and water entry;
- alternating wall turns;
- first-stroke breakout after the dive and turns;
- a sustained-stop finish near the final wall;
- per-length splits when supported;
- provisional stroke cycles and cycle rate; and
- within-race temporal and path-based technique signals.

The user may enter an official time. It is shown separately and never changes the tracking or the independent video estimate.

## Supported race formats

- Individual freestyle, backstroke, breaststroke, and butterfly events through 200 yards or metres.
- Individual medley events through 200 yards or metres when the distance and course form a valid event.
- Short-course yards, short-course metres, and long-course metres.
- MP4, MOV, AVI, M4V, and MKV video containers up to 2 GB, subject to the codecs installed on the computer.

For the cleanest result, use a landscape video that begins with the selected swimmer visible, keeps the full lane in view, and clearly shows the walls.

## Reading confidence correctly

| Output | Meaning |
|---|---|
| Official time | Entered by the user and displayed separately |
| Video estimate | Calculated from visible race evidence |
| Observed frame | Supported by a detector or visual-tracker observation |
| Estimated frame | Brief motion prediction during a visibility gap |
| Takeoff | Visible deck-to-water or movement transition, not buzzer reaction time |
| Breakout | First plausible periodic stroke after entry or a turn |
| Finish | Onset of sustained low movement near the final wall |
| Not measured | The current camera evidence cannot support the requested conclusion |

## Privacy

- Video processing occurs on the computer running SwimVision.
- The application does not require an account.
- Uploaded videos are stored only in the local `uploads/` folder.
- Generated videos and logs stay in the local `outputs/` folder.
- Analysis records stay in the local `data/` folder.
- These folders and common video formats are blocked by `.gitignore` so private footage is not accidentally committed.
- No API key is required for the app or coaching questions.

## Validation

The automated suite includes unit, integration, API, race-configuration, tracker identity, and real-video regression tests.

Run the standard suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The full `swimmer_video2_full.mp4` regression processes all 1,635 frames and checks hand-reviewed identity regions at the block, dive, both lengths, turn, and finish. The test requires zero unlocated frames on that video and rejects a false pass in which the tracker merely follows a different lane.

This proves repeatable behavior on the included race. It is not a universal accuracy percentage across every pool, athlete, stroke, camera, or lighting condition.

## Project status and honest limitations

Ready to demonstrate:

- upload and real video metadata;
- one-click selection;
- full-race identity tracking on the included regression video;
- distance-aware analysis through 200 yards or metres;
- a plain-language React report;
- technique, race, practice-plan, and coach views; and
- confidence-aware behavior when evidence is missing.

Not yet production-validated:

- audio-buzzer reaction timing;
- exact wall-contact timing against an electronic touchpad;
- moving-camera pool calibration for physical speed;
- head-passage breakout distance;
- anatomical joint-angle, breathing, or kick analysis;
- a large labelled accuracy study across pools, strokes, ages, and camera positions; and
- long camera cutaways that completely remove the selected swimmer.

## Troubleshooting

### The page says Failed to fetch

The API is not reachable. For the single-server setup above, confirm that the Uvicorn terminal is still running and open the exact address printed in that terminal.

### FastAPI is missing

The wrong Python interpreter is running. From the repository root, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8000
```

### Port 8000 is unavailable

Use another port:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --host 127.0.0.1 --port 8001
```

Then open [http://127.0.0.1:8001](http://127.0.0.1:8001).

### PowerShell blocks virtual-environment activation

Use this for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

You may also run `.venv\Scripts\python.exe` directly without activating the environment.

### Analysis is slow

CPU inference works but can take several minutes. A CUDA-enabled PyTorch installation and NVIDIA GPU greatly reduce runtime. Close other GPU-heavy applications and keep the browser tab open.

### A split is missing

This is expected when the required wall reversal is not visible clearly enough. SwimVision withholds unsupported splits instead of manufacturing them.

### The wrong swimmer was selected

Click **Analyze another swim** and repeat the selection. Click the center of the athlete's body, not the lane line, block, or a spectator.

## Optional development setup

To work on the React interface with live reloading, run the API and frontend in separate terminals.

Terminal 1:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api:app --reload --port 8000
```

Terminal 2:

```powershell
cd frontend
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Vite proxies `/api` requests to port 8000.

## Technology

- React 19 and Vite
- FastAPI and Uvicorn
- Python 3.12
- OpenCV
- Ultralytics YOLO and BoT-SORT components
- PyTorch and a custom swimmer U-Net
- OpenCV ViTTrack
- NumPy and Pandas
- FFmpeg through `imageio-ffmpeg`
- Pytest

## Repository map

```text
api.py                    FastAPI service and React integration
config.py                 paths, model settings, and supported formats
core/                     tracking, video, race analysis, and coaching logic
frontend/                 React user interface
models/                   bundled swimmer and visual-tracker models
storage/                  local analysis persistence
tests/                    unit, integration, API, and real-video regressions
tests/fixtures/           included regression videos
docs/ANALYSIS_METHOD.md   research-to-feature map and validation boundaries
docs/SUBMISSION_STATUS.md exact ready and not-ready checklist
scripts/                  command-line development and diagnostic tools
```

## Research basis

The system architecture is informed by published swimming-video research including:

- [SwimTrack](https://ceur-ws.org/Vol-3583/paper38.pdf), which separates swimmer position, stroke rate, camera registration, scoreboard recognition, and start-audio detection;
- [DeepDASH](https://ashwhall.github.io/download/Hall2020_Article_TheDetectionTrackingAndTempora.pdf), which combines swimmer-specific temporal detection, identity tracking, stroke localization, and camera calibration;
- [Swimmer Stroke Rate Estimation From Overhead Race Video](https://arxiv.org/abs/2104.12056), which supports phase-aware, smoothed temporal signals; and
- [swimming pose research](https://openaccess.thecvf.com/content_CVPRW_2019/html/CVSports/Zecha_Refining_Joint_Locations_for_Human_Pose_Tracking_in_Sports_Videos_CVPRW_2019_paper.html), which explains why splash, self-occlusion, refraction, and joint swaps require swimming-specific temporal correction.

The complete source-to-feature mapping and the evidence required for stronger claims are documented in [`docs/ANALYSIS_METHOD.md`](docs/ANALYSIS_METHOD.md).

