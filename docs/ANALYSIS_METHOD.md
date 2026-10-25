# SwimVision analysis method

This document records which published ideas SwimVision uses and, just as importantly, what the app does **not**
yet claim to measure. A paper's reported accuracy is not treated as SwimVision's accuracy: the cameras, datasets
and trained models differ.

## Implemented now

1. The persistent track is converted to a sorted, interpolated temporal path. Estimated tracker frames remain
   labelled; they are never counted as direct observations.
2. The tracker's explicit transition from an on-deck person to an in-water swimmer anchors the automatic takeoff
   estimate. It is not labelled as buzzer reaction time.
3. The race metadata determines the required number of lengths and turns. Direction-reversal candidates are
   smoothed, scored for prominence and selected in alternating order. If the correct number is missing, the app
   withholds per-length splits instead of inventing them.
4. Each tracked swimmer crop is resized to a fixed temporal input. Frame-to-frame crop motion is smoothed, its
   dominant period is estimated by autocorrelation, and peaks are separated by that period. For freestyle and
   backstroke the UI distinguishes same-arm cycles from individual arm strokes.
5. The first verified in-water tracker observation becomes an estimated water-entry event. The first plausible
   periodic stroke after entry and after each turn becomes a low-confidence breakout estimate. Counts and rates
   begin at that breakout so dive/turn splash is not counted as swimming rhythm.
6. Findings compare the swimmer with their own race (for example, first versus final length), not with an
   unexplained universal target. Every event retains a source frame, timestamp, confidence and evidence sentence.
7. Training sets are examples tied to a measured finding. Nutrition is food-first, general guidance only; the app
   does not prescribe calories, weight change, supplements or medical diets.

## Research mapping

| Source | Idea used in SwimVision |
|---|---|
| [SwimTrack benchmark](https://ceur-ws.org/Vol-3583/paper38.pdf) | Separate swimmer position, stroke-rate, camera-registration and start-audio tasks; keep stroke/event definitions explicit. |
| [DeepDASH](https://ashwhall.github.io/download/Hall2020_Article_TheDetectionTrackingAndTempora.pdf) | Analyse short temporal sequences of swimmer-only crops and turn a per-frame signal into stroke events. |
| [Woinoski & Bajić](https://arxiv.org/abs/2104.12056) | Represent on-block, dive, underwater, swimming, turn and finish as temporal states; smooth a stroke signal before peak detection. |
| [Central Lyon swimming research](https://github.com/centralelyon/swimming) | Keep tracking, cycle counting and sports-field registration as separate replaceable components. |
| [Swimming Stroke Rate Analysis](https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis) | Treat strokes as a cyclic sequence and use temporal consistency rather than independent frame guesses. |
| [In-field swimming-analysis validation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11598412/) | Use frame-level operational definitions for takeoff, water entry, breakout, turn and split validation. |
| [Dronaquatics](https://openaccess.thecvf.com/content/WACV2026/papers/Tran_Dronaquatics_Real-time_Swimming_Analytics_Using_Drone_Captured_Imagery_WACV_2026_paper.pdf) | Keep pose/stroke classification, optical-flow speed and lap timing as complementary signals. |
| [SAM 3](https://proceedings.iclr.cc/paper_files/paper/2026/hash/e0982cbc81401df3430ee1ff780dc7a2-Abstract-Conference.html) | Potential future promptable segmentation/tracking component only. It is not a swimming-analysis paper and is not presented as one. |

## Validation gates before stronger claims

- **Start:** add audio extraction and buzzer detection, then validate takeoff and reaction-time frame tolerances.
- **Finish:** train or implement visible wall-contact detection rather than using the final tracked frame.
- **Breakout/underwater:** add water-entry, head-breakout and first-stroke labels following the validation study.
- **Stroke count:** create reviewed stroke-event annotations for every stroke and camera style, then measure precision,
  recall and timing error. The current crop-motion number stays visibly provisional.
- **Distance and speed:** register pool landmarks across camera movement before calling pixel motion physical velocity.
- **Coaching:** validate that every recommendation is traceable to a measured result and let a qualified coach review
  age-appropriate volume and rest.

## Nutrition sources

The UI wording follows the food-first approach and individualized guidance in
[USA Swimming nutrition and recovery](https://www.usaswimming.org/parents/popular-resources/nutrition-and-recovery),
the [USOPC Athlete Plate resources](https://www.usopc.org/nutrition), and the
[USADA Fueling for Performance guide](https://www.usada.org/athletes/substances/nutrition-guide/).
