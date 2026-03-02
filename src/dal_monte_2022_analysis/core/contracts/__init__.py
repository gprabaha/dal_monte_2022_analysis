"""Shared domain contracts and schema validation."""

from .gaze_events import GAZE_EVENT_REQUIRED_COLUMNS, validate_gaze_event_frame

__all__ = ["GAZE_EVENT_REQUIRED_COLUMNS", "validate_gaze_event_frame"]

