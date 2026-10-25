# SwimVision submission status — October 25, 2026

## Ready to demonstrate

- Local video upload and real metadata inspection.
- One visual swimmer selection; no coordinates or start/end intervals.
- Persistent identity tracking across the complete 1,635-frame `Video(2).mp4` regression with no unlocated frame.
- Evidence-labelled automatic analysis for valid 25, 50, 100 and 200 yard/metre events.
- Required-turn validation for races from one through eight lengths. Missing turns cause splits to be withheld, not invented.
- Ordered takeoff, water-entry, turn, estimated-breakout and finish timeline.
- Per-length provisional splits, speed, stroke cycles, cycle rate and distance per cycle.
- Plain-language strengths, improvement priorities, example training sets, general food-first recovery guidance,
  confidence labels and downloadable JSON/TXT/CSV reports.
- Automated tests for every supported distance/stroke/pool configuration and a full real-video regression.

## Ready with a visible qualification

- **Takeoff:** inferred from the tracker handoff/movement, not an audio buzzer; do not call it official reaction time.
- **Water entry:** estimated from the frame where the persistent tracker first verifies an in-water swimmer.
- **Turn:** a prominent alternating direction reversal; should be reviewed in the evidence table.
- **Breakout:** first detected periodic stroke after entry/turn, not calibrated head passage through a distance mark.
- **Finish:** final tracked frame and therefore low confidence until wall contact is detected.
- **Stroke cycles:** deterministic crop-motion periodicity, not a validated pose classifier.
- **Distance per cycle:** pool length divided by detected cycles; includes underwater distance and is not biomechanical stroke length.
- **Coaching:** evidence-linked general suggestions, not a replacement for a qualified coach, clinician or dietitian.

## Not production-validated

- No accuracy study across a large labelled set of swimmers, strokes, pools, ages and camera positions.
- No audio-buzzer detector, automatic moving-camera pool calibration, head-pose breakout distance, wall-contact finish,
  swim-specific keypoint model or injury/medical assessment.
- `Video(2).mp4` proves a repeatable regression on one complete race; it is not a universal accuracy percentage.

## Honest submission description

> SwimVision is a working local prototype that persistently follows one selected swimmer and converts the tracked
> race into reviewable events, length metrics, stroke-rhythm estimates and an evidence-linked improvement report.
> Tracking is validated on the project's complete regression video. Analysis values are labelled by confidence and
> experimental measurements are never presented as official race timing or medical advice.
