"""Run tracking in a background thread so the page can never "kill" it.

Streamlit re-runs the whole script whenever the user clicks anything or
refreshes the page, and that cancels whatever the script was doing. A tracking
run takes minutes, so running it inside the script meant any click lost the
work. Instead, the run happens in a normal Python thread; the page only *shows*
its progress and can be refreshed, left and reopened at will.

The registry is shared by all browser sessions (``st.cache_resource``), so a
job started in one tab is still visible after a refresh.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import streamlit as st

import config
from core.errors import SwimVisionError
from core.overlay import write_annotated_video
from core.tracker import SwimmerSelection, summarize_tracking, track_swimmer
from storage.analyses import AnalysisRecord, save_analysis, save_tracking_table

logger = logging.getLogger(__name__)


@dataclass
class TrackingJob:
    """Progress of one background tracking run."""

    analysis_id: str
    fraction: float = 0.0
    message: str = "Starting…"
    error: str | None = None
    done: bool = False
    started_at: float = field(default_factory=time.time)

    @property
    def running(self) -> bool:
        return not self.done


@st.cache_resource
def _registry() -> dict[str, TrackingJob]:
    return {}


_lock = threading.Lock()


def get_job(analysis_id: str) -> TrackingJob | None:
    return _registry().get(analysis_id)


def clear_job(analysis_id: str) -> None:
    with _lock:
        _registry().pop(analysis_id, None)


def start_job(record: AnalysisRecord, checkpoints: list[SwimmerSelection], settings: dict,
              stride: int) -> TrackingJob:
    """Start tracking ``record`` in the background (no-op if a run is already going)."""
    with _lock:
        existing = _registry().get(record.analysis_id)
        if existing is not None and existing.running:
            return existing
        job = TrackingJob(record.analysis_id)
        _registry()[record.analysis_id] = job
    threading.Thread(target=_run, args=(job, record, checkpoints, settings, stride),
                     name=f"tracking-{record.analysis_id}", daemon=True).start()
    return job


def _run(job: TrackingJob, record: AnalysisRecord, checkpoints: list[SwimmerSelection],
         settings: dict, stride: int) -> None:
    """The actual work: track, render the annotated video, save. Never raises."""

    def progress(fraction: float, message: str) -> None:
        job.fraction, job.message = min(max(fraction, 0.0), 1.0), message

    selection, extra = checkpoints[0], checkpoints[1:]
    try:
        result = track_swimmer(record.absolute_video_path(), record.video, selection, stride=stride,
                               checkpoints=extra, progress=lambda f, m: progress(f * 0.85, m), **settings)
        table = result.to_dataframe()
        out_path = config.ANNOTATED_DIR / f"{record.analysis_id}_tracked.mp4"
        write_annotated_video(record.absolute_video_path(), record.video, table, out_path, stride,
                              progress=lambda f, m: progress(0.85 + f * 0.15, m))
        record.selection = selection
        record.checkpoints = checkpoints
        record.tracking = summarize_tracking(result)
        record.tracking_csv = save_tracking_table(record.analysis_id, table)
        record.annotated_video = out_path.relative_to(config.PROJECT_ROOT).as_posix()
        save_analysis(record)
        progress(1.0, "Finished")
    except SwimVisionError as exc:
        job.error = str(exc)
    except Exception:  # unexpected (e.g. GPU out of memory) - log details, show a calm message
        logger.exception("Tracking failed")
        job.error = ("Tracking stopped because of an unexpected problem (details are in the terminal log). "
                     "If the computer has no graphics card, try the 'Simple box tracker' in the advanced "
                     "settings.")
    finally:
        job.done = True
