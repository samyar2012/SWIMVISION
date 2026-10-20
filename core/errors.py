"""Custom exceptions.

Every exception here carries a message that is safe and understandable to show
directly to the user. The UI catches ``SwimVisionError`` and displays
``str(error)`` instead of a Python traceback.
"""

from __future__ import annotations


class SwimVisionError(Exception):
    """Base class for all expected, user-presentable errors."""


class UnsupportedVideoError(SwimVisionError):
    """The file type / container is not something SwimVision accepts."""


class VideoDecodeError(SwimVisionError):
    """The file looks like a video but OpenCV could not decode its frames."""


class ModelLoadError(SwimVisionError):
    """Tracking model weights could not be found, downloaded or loaded."""


class TrackingError(SwimVisionError):
    """Tracking could not start or produced output that fails a sanity check."""
