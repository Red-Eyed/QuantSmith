"""Compare native Stackformers to_packed conversions in dynamic-shape ORT graphs."""

from collections.abc import Callable
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Literal, override
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import stackformers
import torch
from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, CliApp, CliImplicitFlag, SettingsConfigDict
from rich.console import Console
from stackformers import (
    CrossAttention,
    CrossAttentionConfig,
    NoPosEncoding,
    SelfAttention,
    SelfAttentionConfig,
    TransformerEncoder,
    make_padded_input,
    plain_encoder_config,
)
from stackformers.feedforward.config import SwiGLUConfig
from stackformers.feedforward.swiglu import SwiGLU
from torch import Tensor, nn

from examples._shared.environment import environment
from examples._shared.schema import Environment
from examples.quantization_performance.transformer_blocks import (
    Geometry,
    fragment_output,
    synthetic_inputs,
)
from examples.quantization_performance.transformer_kernels import (
    Case,
    Measurement,
    Scheme,
    prepare_ort,
    time_probes,
)
from examples.quantization_performance.transformer_kernels import (
    Config as TimingConfig,
)

type Workload = Literal["self_attention", "cross_attention", "swiglu", "transformer"]
type Layout = Literal["unpacked", "attention_packed", "ffn_packed", "packed"]


class PackedConversion(BaseModel):
    """Validate the upgraded library's bound conversion method at its boundary."""

    model_config = ConfigDict(from_attributes=True, strict=True, frozen=True)
    to_packed: Callable[[], nn.Module]


def geometries() -> list[Geometry]:
    """Cover smaller projections and larger single-token/long-sequence regimes."""
    return [
        Geometry(tokens=tokens, context=128, width=width, heads=width // 64)
        for width in (256, 768)
        for tokens in (1, 128)
    ]


class Config(BaseSettings):
    """Keep source layout comparisons and all timing budgets explicit."""

    model_config = SettingsConfigDict(cli_kebab_case=True, cli_parse_args=False)
    output: Path = Path("work_dir/artifacts/packed_stackformers")
    workloads: list[Workload] = Field(
        default=["self_attention", "cross_attention", "swiglu", "transformer"],
        min_length=1,
    )
    geometries: list[Geometry] = Field(default_factory=geometries, min_length=1)
    opsets: list[Literal[18, 21]] = Field(default=[21], min_length=1)
    schemes: list[Scheme] = Field(
        default=[
            "fp32",
            "dynamic",
            "fixed_float_u8s8",
            "fixed_quantized_u8s8",
            "fixed_float_s8s8",
            "fixed_quantized_s8s8",
        ],
        min_length=1,
    )
    seed: int = Field(default=42, ge=0)
    activation_range: float = Field(default=4, gt=0, allow_inf_nan=False)
    warmup: int = Field(default=30, ge=1)
    runs: int = Field(default=300, ge=1)
    repetitions: int = Field(default=3, ge=1)
    profile_runs: int = Field(default=10, ge=1)
    quiet: CliImplicitFlag[bool] = False


class Variant(BaseModel):
    """Bind each layout's timings and quantization error to the common FP32 source."""

    layout: Layout
    converted_modules: tuple[str, ...]
    linear_modules: int
    fp32_maximum_absolute_error: float
    measurements: list[Measurement]


class Comparison(BaseModel):
    """Keep all layouts from one owned source together for paired timing."""

    workload: Workload
    geometry: Geometry
    source_weights_sha256: str
    input_sha256: str
    variants: list[Variant]


class Report(BaseModel):
    """Record exact installed library source, dynamic axes and every comparison."""

    config: Config
    environment: Environment
    numpy: str
    stackformers_source_sha256: str
    dynamic_sequence: bool
    comparisons: list[Comparison]


class Fragment(nn.Module):
    """Adapt both packed and unpacked native modules through the same call contract."""

    def __init__(self, workload: Workload, geometry: Geometry) -> None:
        """Use native attention and feed-forward modules with owned source weights."""
        super().__init__()
        self.workload = workload
        self.block: nn.Module
        match workload:
            case "self_attention":
                self.block = SelfAttention(
                    SelfAttentionConfig(
                        dim=geometry.width,
                        heads=geometry.heads,
                        dim_head=geometry.width // geometry.heads,
                        causal=True,
                    ),
                    NoPosEncoding(),
                )
            case "cross_attention":
                self.block = CrossAttention(
                    CrossAttentionConfig(
                        dim=geometry.width,
                        heads=geometry.heads,
                        dim_head=geometry.width // geometry.heads,
                    ),
                    NoPosEncoding(),
                )
            case "swiglu":
                self.block = SwiGLU(SwiGLUConfig(dim=geometry.width, mult=4))
            case "transformer":
                self.block = TransformerEncoder(
                    plain_encoder_config(
                        dim=geometry.width,
                        heads=geometry.heads,
                        num_layers=1,
                        causal=True,
                        ff_mult=4,
                    )
                )

    @override
    def forward(self, inputs: Tensor, memory: Tensor) -> Tensor:
        """Call the native module without branching on packed implementation classes."""
        if self.workload == "swiglu":
            result = self.block(inputs)
        else:
            query = make_padded_input(
                inputs, torch.ones(inputs.shape[:2], dtype=torch.bool)
            )
            if self.workload == "cross_attention":
                context = make_padded_input(
                    memory, torch.ones(memory.shape[:2], dtype=torch.bool)
                )
                result = self.block(query, context)
            else:
                result = self.block(query)
        if not isinstance(result, Tensor):
            raise TypeError("Stackformers fragment must return a Tensor")
        return result


def packed_copy(source: Fragment, layout: Layout) -> tuple[Fragment, tuple[str, ...]]:
    """Use only to_packed on an owned copy; keep the unpacked source unchanged."""
    target = deepcopy(source)
    converted: list[str] = []
    for name, child in tuple(target.named_modules()):
        selected = (
            isinstance(child, (SelfAttention, CrossAttention))
            and layout in ("packed", "attention_packed")
        ) or (isinstance(child, SwiGLU) and layout in ("packed", "ffn_packed"))
        if not selected:
            continue
        conversion = PackedConversion.model_validate(child)
        packed = conversion.to_packed()
        if not isinstance(packed, nn.Module):
            raise TypeError("to_packed() must return a Module")
        target.set_submodule(name, packed)
        converted.append(name)
    return target.eval(), tuple(converted)


def weight_digest(module: nn.Module) -> str:
    """Check conversion ownership against exact parameter names and byte contents."""
    digest = sha256()
    for name, parameter in module.named_parameters():
        digest.update(name.encode())
        digest.update(parameter.detach().numpy().tobytes())
    return digest.hexdigest()


def library_digest() -> str:
    """Identify the installed Python sources, including local uncommitted changes."""
    root = Path(stackformers.__file__).parent
    digest = sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_library_snapshot(output: Path) -> None:
    """Preserve the exact local library sources while its checkout keeps changing."""
    root = Path(stackformers.__file__).parent
    with ZipFile(
        output / "stackformers-source.zip", "w", compression=ZIP_DEFLATED
    ) as archive:
        for path in sorted(root.rglob("*.py")):
            archive.write(path, f"stackformers/{path.relative_to(root).as_posix()}")
        for path in sorted(root.parent.glob("stackformers-*.dist-info/METADATA")):
            archive.write(path, path.relative_to(root.parent).as_posix())


def timing_config(config: Config, geometry: Geometry) -> TimingConfig:
    """Reuse the existing ORT timing settings without changing project dependencies."""
    return TimingConfig(
        output=config.output,
        geometries=[geometry],
        opsets=config.opsets,
        opset_mode="native",
        scope="weights",
        seed=config.seed,
        activation_range=config.activation_range,
        warmup=config.warmup,
        runs=config.runs,
        repetitions=config.repetitions,
        profile_runs=config.profile_runs,
        quiet=config.quiet,
    )


def compare(
    workload: Workload, geometry: Geometry, output: Path, config: Config
) -> Comparison:
    """Verify parity and interleave all layouts and schemes in one timing sweep."""
    with torch.random.fork_rng():
        torch.manual_seed(config.seed)
        source = Fragment(workload, geometry).eval()
    inputs = synthetic_inputs("transformer", geometry, config.seed + 1)
    expected = fragment_output(source, inputs).numpy()
    source_hash = weight_digest(source)
    input_hash = sha256(
        b"".join(value.numpy().tobytes() for value in inputs)
    ).hexdigest()
    result = Comparison(
        workload=workload,
        geometry=geometry,
        source_weights_sha256=source_hash,
        input_sha256=input_hash,
        variants=[],
    )
    layouts: tuple[Layout, ...] = (
        ("unpacked", "attention_packed", "ffn_packed", "packed")
        if workload == "transformer"
        else ("unpacked", "packed")
    )
    all_probes = []
    settings = timing_config(config, geometry)
    for layout in layouts:
        candidate, converted = packed_copy(source, layout)
        if layout != "unpacked" and not converted:
            raise RuntimeError("requested packing converted no native modules")
        actual = fragment_output(candidate, inputs).numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
        assert weight_digest(source) == source_hash, "packing changed source weights"
        destination = output / layout
        destination.mkdir(parents=True, exist_ok=True)
        case = Case(
            fragment="transformer",
            geometry=geometry,
            input_shapes=tuple(tuple(value.shape) for value in inputs),
            input_strides=tuple(value.stride() for value in inputs),
            input_sha256=input_hash,
            torch_linear_modules=sum(
                isinstance(child, nn.Linear) for child in candidate.modules()
            ),
            measurements=[],
            unsupported=[],
        )
        probes = [
            probe
            for probe in prepare_ort(
                candidate, inputs, expected, case, destination, settings
            )
            if probe.measurement.scheme in config.schemes
        ]
        unsupported = [
            item for item in case.unsupported if item.scheme in config.schemes
        ]
        if unsupported:
            raise RuntimeError(f"Unsupported kernels: {unsupported}")
        result.variants.append(
            Variant(
                layout=layout,
                converted_modules=converted,
                linear_modules=case.torch_linear_modules,
                fp32_maximum_absolute_error=float(np.max(np.abs(actual - expected))),
                measurements=[probe.measurement for probe in probes],
            )
        )
        all_probes.extend(probes)
    time_probes(all_probes, settings)
    (output / "report.json").write_text(result.model_dump_json(indent=2) + "\n")
    return result


def main(config: Config) -> Report:
    """Run serial paired comparisons and checkpoint each completed workload."""
    config.output.mkdir(parents=True, exist_ok=True)
    save_library_snapshot(config.output)
    torch.set_num_threads(1)
    console = Console(stderr=True, quiet=config.quiet)
    report = Report(
        config=config,
        environment=environment(),
        numpy=np.__version__,
        stackformers_source_sha256=library_digest(),
        dynamic_sequence=True,
        comparisons=[],
    )
    with torch.inference_mode():
        for index, geometry in enumerate(config.geometries):
            for workload in config.workloads:
                console.print(f"{workload}: {geometry.model_dump()}")
                report.comparisons.append(
                    compare(
                        workload,
                        geometry,
                        config.output / f"shape_{index}_{workload}",
                        config,
                    )
                )
                (config.output / "report.json").write_text(
                    report.model_dump_json(indent=2) + "\n"
                )
    return report


if __name__ == "__main__":
    settings = CliApp.run(Config)
    print(main(settings).model_dump_json(indent=2))
