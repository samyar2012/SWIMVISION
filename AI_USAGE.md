# AI Usage Disclosure

This file records how generative AI was used on SwimVision, for honest disclosure to the Congressional App Challenge. **Keep it updated every time AI helps with something new.**

## Tools

- Cursor IDE agent (large language model coding assistant)

## Development assistance log

| Date | Phase | What the AI did | What the student did |
|---|---|---|---|
| 2026-10-04 | Phase 1 | Proposed the project architecture and folder layout; wrote the initial code for configuration, video metadata reading, JSON storage, the Streamlit upload page and styling, unit tests, README and this file; installed dependencies; ran the app and tests; fixed errors found during testing. | Defined the product idea, requirements, and development plan; directed and reviewed the work. *(Edit this column to describe exactly what you did.)* |
| 2026-10-05 | Phase 2 | Wrote the person-detection, swimmer-tracking (target-following logic, visual-tracker fallback, camera-cut detection), overlay/annotated-video, storage-extension and Streamlit selection/results code, plus unit and integration tests and a command-line dev tool. Chose pretrained models (YOLO11, ByteTrack, OpenCV ViT tracker). Found and fixed problems by testing on real footage (e.g. the visual tracker following across camera cuts; boxes collapsing to a dot). | Directed the work and the phase plan; reviewed results. *(Edit this column to describe exactly what you did.)* |
| 2026-10-05 | Phase 2 fix | After the user reported 'swimmer found in less than half the frames' on their own pool videos, AI diagnosed it by experiment: COCO YOLO found 0 swimmers in 16 tests (only spectators), a too-strict box-size rule wrongly stopped the visual tracker (fixed, regression test), and the tracker can latch onto splash. Added stop-reason messages and a drag-to-draw box with zoom preview. | Reported the problem with real footage. *(Edit this column.)* |
| 2026-10-05 | Tracking redesign | After the user reported that the box does not follow the swimmer into the dive, AI researched published swim-analysis work (SwimTrack benchmark, DeepDASH, a 2026 paper using SAM 3 per lane), tested SAM 2 on the user's videos, found it can latch onto a spectator, and built: SAM 2 tracking from user checkpoints, a camera-motion-compensated 'not moving' check (measured thresholds on real footage), plain-language statuses, and tests. | Reported the dive problem and asked for research. *(Edit this column.)* |
| 2026-10-07 | Phase 2 repair | The student judged the system over-engineered and unreliable and ordered a simplification. AI audited the code, verified the real APIs from installed package source (Ultralytics `SAM2VideoPredictor`, `streamlit-image-coordinates`, `streamlit-cropper`), replaced the selection tool with a visible rectangle (`streamlit-cropper`), reduced tracking to one SAM 2 pipeline with honest lost-frame reporting, removed YOLO/ViT/camera-motion/checkpoints/background jobs from the app (moved to `legacy/`), added `SWIMVISION_DEBUG=1`, wrote a standalone check script and ran it on the student's real race video, and rewrote tests and docs. | Wrote the repair specification and acceptance criteria. *(Edit this column.)* |
| 2026-10-07 | Dive-entry loss | The student reported that tracking always lost the swimmer right after the dive. AI ran experiments on the student's video (block-only start, SAM 2.1 large, later starts) showing SAM 2 never re-finds the swimmer after entry, then built multi-mark tracking (forward + backward SAM 2 passes between user marks, conflicts counted as lost) and measured it: 14% -> 58% mask coverage over 15 s, entry gap 0.5 s. | Identified the failure and demanded the fix. *(Edit this column.)* |
| 2026-10-24 | First race-analysis system | AI implemented a distance-aware 25/50/100/200 analysis pipeline, temporal swimmer-crop motion signal, expected-turn and length-split validation, plain-language evidence dashboard, training-set examples, nutrition safety boundaries, persistence and tests. It mapped the implementation to the research sources in `docs/ANALYSIS_METHOD.md`, tested all valid through-200 event configurations, and added start/turn/two-length checks to the complete 1,635-frame `Video(2).mp4` regression. | Required a family-friendly report, research-based implementation, full-video testing, and support through 200 yards/metres. *(Edit this column to describe your review.)* |

## Rules for this project

- AI-generated code is **not** presented as written entirely by the student.
- The student must understand and be able to explain every part of the submitted code.
- Computer-vision results are never invented by AI; every displayed metric comes from video processing, user input, or an explicit calculation in the code.

## AI / machine-learning models used inside the running application

- **Pretrained/custom computer-vision models used by the current hybrid tracker:**
  - Ultralytics YOLO person detection before water entry.
  - The project's swimmer U-Net heatmap model after entry.
  - OpenCV ViTTrack to preserve the one user-selected visual identity and bridge detector gaps.
- **Race analysis:** deterministic NumPy/OpenCV signal processing turns the saved track and swimmer crops into
  provisional events, turns, splits and stroke cycles. It does not call a generative model and does not claim the
  published papers' accuracy.
- **Future optional text generation:** if added, it will receive calculated numbers/text, never raw video, and must
  not invent measurements. The current user-friendly coaching text is deterministic and works offline.
- No generative AI is used for the video analysis itself.

## Test data note

- Phase 1 tests use small synthetic videos generated by the test code (and FFmpeg's built-in test pattern) so frame rate, frame count and resolution have known true values.
- Unit tests use synthetic videos and paths for exact timing, distance, expected-turn and report-format checks.
- The complete 1,635-frame `Video(2).mp4` is a permanent real-footage regression. It checks identity regions across
  the whole race, zero unlocated frames, the dive-window start anchor, one turn and two complete lengths. This is one
  video regression, not a general accuracy claim.
- During Phase 2 development, two **public-domain** historical open-water swimming films from Wikimedia Commons ("2 KM ZWEMWEDSTRIJDEN" and "ZWEMWEDSTRIJDEN OM DE SCHELDEBEKER") were downloaded to a temporary folder **outside the project** and used for manual testing only. They are not committed to the repository. No accuracy figure is claimed from them; they are not modern pool footage.
