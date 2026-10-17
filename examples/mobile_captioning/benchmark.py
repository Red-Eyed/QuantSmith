"""Fresh-process timings, RSS high-water marks, and separate raw ORT traces."""

from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numpy as np
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

from examples._shared.vision.profile_runtime import peak_mib
from examples.mobile_captioning.inference import (
    generate,
    generate_context,
    predictor,
    session,
)
from quantsmith.domain import FloatArray


class Timing(BaseModel):
    """Summarize warmed runs without profiling instrumentation."""

    median_ms: float
    p95_ms: float
    runs: int


class Measurements(BaseModel):
    """Distinguish model bytes, process memory, and actual decoding work."""

    encoder: Timing
    decoder: Timing
    generation: Timing
    fixed_steps_generation: Timing
    generated_tokens: int
    model_bytes: int
    baseline_peak_mib: float
    session_peak_mib: float
    inference_peak_mib: float
    traces: tuple[Path, Path]


def timing(call: Callable[[], None], warmup: int, runs: int) -> Timing:
    """Measure a fixed call repeatedly without retaining outputs."""
    for _ in range(warmup):
        call()
    elapsed: list[float] = []
    for _ in range(runs):
        start = perf_counter()
        call()
        elapsed.append((perf_counter() - start) * 1000)
    return Timing(
        median_ms=float(np.median(elapsed)),
        p95_ms=float(np.percentile(elapsed, 95)),
        runs=runs,
    )


def load_image(path: Path) -> FloatArray:
    """Admit only a finite batch-one RGB NCHW FP32 benchmark input."""
    pixels = np.load(path, allow_pickle=False)
    if (
        not isinstance(pixels, np.ndarray)
        or pixels.dtype != np.float32
        or pixels.ndim != 4
        or pixels.shape[:2] != (1, 3)
        or not np.all(np.isfinite(pixels))
    ):
        raise ValueError("image must be finite batch-one RGB NCHW FP32")
    return np.asarray(pixels, dtype=np.float32)


class Worker(BaseSettings):
    """Run each graph pair in an independent process with identical options."""

    model_config = SettingsConfigDict(cli_kebab_case=True)
    encoder: Path
    decoder: Path
    image: Path
    output: Path
    length: int = Field(ge=4)
    threads: int = Field(default=1, ge=1)
    warmup: int = Field(default=5, ge=0)
    runs: int = Field(default=30, ge=1)
    profile_runs: int = Field(default=5, ge=1)

    def cli_cmd(self) -> None:
        """Capture RSS before creating profile sessions, then retain both traces."""
        self.output.mkdir(parents=True, exist_ok=True)
        pixels = load_image(self.image)
        baseline = peak_mib()
        encoder = session(self.encoder, self.threads)
        decoder = session(self.decoder, self.threads)
        session_peak = peak_mib()
        predict = predictor(encoder, decoder)
        context = predict.encode(pixels)
        tokens = np.zeros((1, self.length), dtype=np.int64)
        tokens[0, 0] = 1

        def encode() -> None:
            """Time one image encoding."""
            predict.encode(pixels)

        def decode() -> None:
            """Time one decoder call with BOS and otherwise masked padding."""
            predict.decode(tokens, context)

        def generation() -> None:
            """Include EOS-dependent decoding and one image encoding."""
            generate(predict, pixels, self.length)

        def fixed() -> None:
            """Compare the same number of decoding steps even when EOS differs."""
            generate_context(
                predict, predict.encode(pixels), self.length, stop_at_eos=False
            )

        encoder_timing = timing(encode, self.warmup, self.runs)
        decoder_timing = timing(decode, self.warmup, self.runs)
        generation_timing = timing(generation, self.warmup, self.runs)
        fixed_timing = timing(fixed, self.warmup, self.runs)
        generated_tokens = len(generate(predict, pixels, self.length))
        inference_peak = peak_mib()
        profile_encoder = session(self.encoder, self.threads, profile=True)
        profile_decoder = session(self.decoder, self.threads, profile=True)
        profiled = predictor(profile_encoder, profile_decoder)
        for _ in range(self.profile_runs):
            generate(profiled, pixels, self.length)
        traces = (
            Path(profile_encoder.end_profiling()),
            Path(profile_decoder.end_profiling()),
        )
        report = Measurements(
            encoder=encoder_timing,
            decoder=decoder_timing,
            generation=generation_timing,
            fixed_steps_generation=fixed_timing,
            generated_tokens=generated_tokens,
            model_bytes=self.encoder.stat().st_size + self.decoder.stat().st_size,
            baseline_peak_mib=baseline,
            session_peak_mib=session_peak,
            inference_peak_mib=inference_peak,
            traces=traces,
        )
        (self.output / "measurements.json").write_text(
            report.model_dump_json(indent=2) + "\n"
        )


if __name__ == "__main__":
    CliApp.run(Worker)
