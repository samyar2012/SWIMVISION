# Coach output for a React client

The analysis and its presentation are separate. Tracking produces a race report;
`build_coach_brief(report).to_dict()` turns that report into the small, user-facing
JSON object a React page should render.

The important fields are:

- `time_label`, `race_time_s`, and `race_time_display`
- `headline` and `overview`
- `strengths[]` and `priorities[]`, each with an explanation and action
- `next_practice`
- `suggested_questions[]`

Keep the complete `RaceAnalysis.to_dict()` response behind a “measurement details”
view. It is evidence for developers and coaches, not the first screen for families.

For chat, call `answer_coach_question(report, question)` for the dependency-free
fallback. A backend can inject a small local model as the optional `model` callable.
The model receives `build_coach_context(report)`, which explicitly limits it to
measured facts and tells it not to invent technique measurements.

The official race time is optional and should be sent as `official_time_s` to
`analyze_race`. It changes only the displayed result; it never changes tracking,
video boundaries, or detected event frames.
