"""Profile one ONNX model per fresh process using a saved representative input."""

import platform
import resource
import sys
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Annotated

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict

_ARRAY = TypeAdapter(
    NDArray[np.generic], config=ConfigDict(arbitrary_types_allowed=True)
)
_OUTPUTS = TypeAdapter(
    list[NDArray[np.generic]], config=ConfigDict(arbitrary_types_allowed=True)
)


class EventArgs(BaseModel):
    """Read the operator label while tolerating other ORT trace attributes."""

    model_config = ConfigDict(extra="ignore", strict=True)
    op_name: str = ""


class Event(BaseModel):
    """Validate the subset of an ORT trace needed for operator totals."""

    model_config = ConfigDict(extra="ignore", strict=True)
    cat: str = ""
    dur: float = 0
    args: EventArgs = Field(default_factory=EventArgs)


class Report(BaseModel):
    """Keep process high-water marks distinct from isolated tensor memory."""

    model: Path
    input: Path
    threads: int
    warmup: int
    runs: int
    runtime_version: str
    platform: str
    model_bytes: int
    median_ms: float
    p95_ms: float
    baseline_peak_mib: float
    session_peak_mib: float
    inference_peak_mib: float
    optimized_operators: dict[str, int]
    profiled_operator_us: dict[str, float]


def peak_mib() -> float:
    """Return process RSS high-water mark using platform-specific native units."""
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def benchmark(
    session: ort.InferenceSession,
    feed: Mapping[str, NDArray[np.generic]],
    warmup: int,
    runs: int,
    quiet: bool,
) -> list[float]:
    """Time warmed inference with profiling disabled and report bounded progress."""
    for _ in range(warmup):
        _OUTPUTS.validate_python(session.run(None, dict(feed)))
    timings: list[float] = []
    for index in range(runs):
        start = perf_counter()
        _OUTPUTS.validate_python(session.run(None, dict(feed)))
        timings.append((perf_counter() - start) * 1000)
        if not quiet and (index + 1) % max(1, runs // 10) == 0:
            print(f"{index + 1}/{runs} runs", file=sys.stderr)
    return timings


def operator_profile(
    model: Path,
    options: ort.SessionOptions,
    feed: Mapping[str, NDArray[np.generic]],
    runs: int,
) -> dict[str, float]:
    """Collect operator totals in a separate session after timing and RSS capture."""
    options.optimized_model_filepath = ""
    options.enable_profiling = True
    session = ort.InferenceSession(
        str(model), options, providers=["CPUExecutionProvider"]
    )
    for _ in range(runs):
        _OUTPUTS.validate_python(session.run(None, dict(feed)))
    trace = Path(session.end_profiling())
    totals: dict[str, float] = {}
    for event in TypeAdapter(list[Event]).validate_json(trace.read_text()):
        if event.cat == "Node" and event.args.op_name:
            totals[event.args.op_name] = totals.get(event.args.op_name, 0.0) + event.dur
    return totals


class Profile(BaseSettings):
    """Expose explicit paths and benchmark settings through pydantic-settings."""

    model_config = SettingsConfigDict(cli_parse_args=False, cli_kebab_case=True)
    model: Path
    input: Path
    input_name: str = "images"
    output: Path
    threads: Annotated[int, Field(gt=0)] = 1
    warmup: Annotated[int, Field(ge=0)] = 10
    runs: Annotated[int, Field(gt=0)] = 100
    quiet: CliImplicitFlag[bool] = False

    def cli_cmd(self) -> None:
        """Benchmark without profiling, then collect a separate operator trace."""
        self.output.mkdir(parents=True, exist_ok=True)
        inputs = _ARRAY.validate_python(np.load(self.input, allow_pickle=False))
        if inputs.dtype != np.float32 or not np.all(np.isfinite(inputs)):
            raise ValueError("input must be a finite FP32 .npy array")
        options = ort.SessionOptions()
        options.intra_op_num_threads = self.threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        optimized = self.output / "optimized.onnx"
        options.optimized_model_filepath = str(optimized)
        baseline = peak_mib()
        session = ort.InferenceSession(
            str(self.model), options, providers=["CPUExecutionProvider"]
        )
        session_peak = peak_mib()
        feed = {self.input_name: inputs}
        timings = benchmark(session, feed, self.warmup, self.runs, self.quiet)
        inference_peak = peak_mib()
        operators = Counter(node.op_type for node in onnx.load(optimized).graph.node)
        del session
        options.profile_file_prefix = str(self.output / "ort")
        totals = operator_profile(self.model, options, feed, self.runs)
        report = Report(
            model=self.model,
            input=self.input,
            threads=self.threads,
            warmup=self.warmup,
            runs=self.runs,
            runtime_version=ort.__version__,
            platform=platform.platform(),
            model_bytes=self.model.stat().st_size,
            median_ms=float(np.median(timings)),
            p95_ms=float(np.percentile(timings, 95)),
            baseline_peak_mib=baseline,
            session_peak_mib=session_peak,
            inference_peak_mib=inference_peak,
            optimized_operators=dict(operators),
            profiled_operator_us=dict(totals),
        )
        content = report.model_dump_json(indent=2)
        (self.output / "report.json").write_text(content + "\n")
        print(content)


if __name__ == "__main__":
    CliApp.run(Profile)
