"""Runnable demonstrations with optional dependencies."""

import os

# Prevent import-time ORT telemetry from creating files outside the output tree.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
