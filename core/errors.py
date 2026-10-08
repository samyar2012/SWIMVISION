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
    """A pretrained model (YOLO weights, tracker model) could not be loaded."""


class NoPersonDetectedError(SwimVisionError):
    """No person was found in the chosen frame."""


class TrackingError(SwimVisionError):
    """The selected swimmer could not be tracked at all."""
