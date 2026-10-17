"""QuantSmith behavioral and static contract tests."""

import os

# ORT initializes telemetry before fixtures can isolate runtime file writes.
os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
