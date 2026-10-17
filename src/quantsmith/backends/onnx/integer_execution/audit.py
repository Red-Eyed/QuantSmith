"""Inspect ORT's compiled CPU graph and reject unexpected floating fallbacks."""

from pathlib import Path
from tempfile import TemporaryDirectory

import onnx
import onnxruntime as ort
from onnx import ModelProto
from onnxruntime.capi.onnxruntime_pybind11_state import (
    Fail,
    InvalidArgument,
    InvalidGraph,
    NotImplemented,
    RuntimeException,
)
from pydantic import BaseModel, ConfigDict

from quantsmith.domain import Absent
from quantsmith.plan import FloatOutput, QuantizationPlan, QuantizeOutput
from quantsmith.result import FailureKind, Ok, QuantSmithError, Result, failure


class IntegerNode(BaseModel):
    """Associate a planned source operation with its compiled integer kernel."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    node: str
    source_operator: str
    executed_operator: str
    domain: str


class UnverifiedNode(BaseModel):
    """Identify an operation intentionally outside the verified integer contract."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    node: str
    operator: str
    reason: str


class ExecutionReport(BaseModel):
    """Record compilation evidence for this ORT version and CPU target."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    runtime_version: str
    provider: str = "CPUExecutionProvider"
    integer_nodes: tuple[IntegerNode, ...]
    unverified_nodes: tuple[UnverifiedNode, ...]


_INTEGER_OPERATORS: dict[str, frozenset[str]] = {
    "Conv": frozenset({"QLinearConv"}),
    "Gemm": frozenset({"QGemm"}),
    "MatMul": frozenset({"QLinearMatMul", "MatMulInteger", "MatMulIntegerToFloat"}),
    "Add": frozenset({"QLinearAdd"}),
}


def _source_owner(name: str, originals: frozenset[str]) -> str | Absent:
    """Disambiguate source identities before accepting ORT's layout-name suffixes."""
    if name in originals:
        return name
    matches = [source for source in originals if name.startswith(source + "_token_")]
    if not matches:
        return Absent(reason="compiled node has no recognized source identity")
    return max(matches, key=len)


def compile_cpu(model: ModelProto) -> ModelProto:
    """Inspect a temporary optimized graph without exporting hardware-bound nodes."""
    with TemporaryDirectory(prefix="quantsmith-ort-") as directory:
        path = Path(directory) / "optimized.onnx"
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.optimized_model_filepath = str(path)
        options.log_severity_level = 3
        ort.InferenceSession(
            model.SerializeToString(), options, providers=["CPUExecutionProvider"]
        )
        return onnx.load(path)


def inspect_execution(
    source: ModelProto,
    compiled: ModelProto,
    plans: tuple[QuantizationPlan, ...],
    exclusions: tuple[UnverifiedNode, ...] = (),
) -> Result[ExecutionReport, QuantSmithError]:
    """Require source-name provenance rather than accepting unrelated integer nodes."""
    originals = {node.name: node for node in source.graph.node}
    original_names = frozenset(originals)
    owners = {
        candidate.name: _source_owner(candidate.name, original_names)
        for candidate in compiled.graph.node
    }
    reasons = {item.node: item.reason for item in exclusions}
    expected = {
        operation.node
        for plan in plans
        for operation in plan.operations
        if isinstance(operation, QuantizeOutput | FloatOutput)
    }
    float_outputs = {
        operation.node
        for plan in plans
        for operation in plan.operations
        if isinstance(operation, FloatOutput)
    }
    integer: list[IntegerNode] = []
    floating: list[UnverifiedNode] = []
    if not expected <= originals.keys():
        return failure(
            FailureKind.CONFLICT,
            "execution audit",
            "output plan refers to unknown source nodes",
        )
    for name, node in originals.items():
        if name not in expected:
            floating.append(
                UnverifiedNode(
                    node=name,
                    operator=node.op_type,
                    reason=reasons.get(name, "outside integer execution selection"),
                )
            )
            continue
        allowed = _INTEGER_OPERATORS.get(node.op_type, frozenset())
        if name in float_outputs:
            allowed = (
                frozenset({"MatMulIntegerToFloat"})
                if node.op_type == "MatMul"
                else frozenset()
            )
        matches = [
            candidate
            for candidate in compiled.graph.node
            if owners[candidate.name] == name
            and candidate.op_type in allowed
            and candidate.domain in {"", "com.microsoft", "com.ms.internal.nhwc"}
        ]
        if len(matches) != 1:
            return failure(
                FailureKind.EXECUTION,
                name,
                f"expected integer {node.op_type}; "
                f"found {len(matches)} matching compiled kernels",
            )
        candidate = matches[0]
        integer.append(
            IntegerNode(
                node=name,
                source_operator=node.op_type,
                executed_operator=candidate.op_type,
                domain=candidate.domain,
            )
        )
    return Ok(
        ExecutionReport(
            runtime_version=ort.__version__,
            integer_nodes=tuple(integer),
            unverified_nodes=tuple(floating),
        )
    )


def audit_execution(
    source: ModelProto,
    model: ModelProto,
    plans: tuple[QuantizationPlan, ...],
    exclusions: tuple[UnverifiedNode, ...] = (),
) -> Result[ExecutionReport, QuantSmithError]:
    """Translate documented compilation failures and preserve invariant exceptions."""
    try:
        compiled = compile_cpu(model)
    except (
        Fail,
        InvalidArgument,
        InvalidGraph,
        NotImplemented,
        RuntimeException,
    ) as error:
        return failure(FailureKind.EXECUTION, "ORT CPU compilation", str(error))
    except OSError as error:
        return failure(FailureKind.IO, "ORT CPU compilation", str(error))
    return inspect_execution(source, compiled, plans, exclusions)
