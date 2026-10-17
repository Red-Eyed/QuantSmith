"""Report quantization decisions without hiding how examples configure QuantSmith."""

import json
from pathlib import Path
from typing import Literal, TypedDict, assert_never

import numpy as np

from examples._shared.schema import Coverage
from quantsmith.backends.onnx.integer_execution.audit import ExecutionReport
from quantsmith.domain import Absent, FloatArray, Graph, IntArray, PerChannel, PerTensor
from quantsmith.plan import (
    Operation,
    QuantizationPlan,
    QuantizeConstant,
    QuantizeInput,
    QuantizeOutput,
    RescaleInput,
)


class ArrayRecord(TypedDict):
    """Keep an array's shape and dtype alongside human-readable numerical values."""

    dtype: str
    shape: tuple[int, ...]
    values: list[float]


class GranularityRecord(TypedDict):
    """Represent per-tensor granularity without an axis."""


class ChannelRecord(TypedDict):
    """Preserve the axis of a per-channel encoding."""

    axis: int


class EncodingRecord(TypedDict):
    """Serialize numerical encoding data separately from its domain model."""

    scale: ArrayRecord
    zero_point: ArrayRecord
    granularity: GranularityRecord | ChannelRecord


class QuantizeRecord(TypedDict):
    """Record the selected consumer edge and its encoding."""

    node: str
    index: int
    encoding: EncodingRecord


class RescaleRecord(TypedDict):
    """Record paired activation and weight channel rescaling."""

    node: str
    scale: ArrayRecord
    activation_axis: int
    weight_axis: int


class ConstantRecord(QuantizeRecord):
    """Preserve the exact reconstructed integer codes alongside their encoding."""

    values: ArrayRecord


class OutputRecord(QuantizeRecord):
    """Distinguish producer-output rounding from consumer-input rounding."""

    target: Literal["output"]


class PlanRecord(TypedDict):
    """Preserve stage operations and exclusions in the existing JSON format."""

    operations: list[QuantizeRecord | OutputRecord | ConstantRecord | RescaleRecord]
    excluded: tuple[str, ...]


def array_record(array: FloatArray | IntArray) -> ArrayRecord:
    """Serialize numerical plan arrays with their original dtype and shape."""
    numeric = np.asarray(array, dtype=np.float64)
    return {
        "dtype": str(array.dtype),
        "shape": tuple(int(length) for length in array.shape),
        "values": [float(number) for number in numeric.flat],
    }


def operation_record(
    operation: Operation,
) -> QuantizeRecord | OutputRecord | ConstantRecord | RescaleRecord:
    """Serialize every supported operation without an untyped JSON fallback."""
    match operation:
        case (
            QuantizeInput(node=node, index=index, encoding=encoding)
            | QuantizeOutput(node=node, index=index, encoding=encoding)
            | QuantizeConstant(node=node, index=index, encoding=encoding)
        ):
            granularity: GranularityRecord | ChannelRecord
            match encoding.granularity:
                case PerTensor():
                    granularity = GranularityRecord()
                case PerChannel(axis=axis):
                    granularity = ChannelRecord(axis=axis)
                case _:
                    assert_never(encoding.granularity)
            record: QuantizeRecord = {
                "node": node,
                "index": index,
                "encoding": {
                    "scale": array_record(encoding.scale),
                    "zero_point": array_record(encoding.zero_point),
                    "granularity": granularity,
                },
            }
            match operation:
                case QuantizeOutput():
                    return OutputRecord(
                        node=node,
                        index=index,
                        encoding=record["encoding"],
                        target="output",
                    )
                case QuantizeConstant(values=values):
                    return ConstantRecord(
                        node=record["node"],
                        index=record["index"],
                        encoding=record["encoding"],
                        values=array_record(values),
                    )
                case QuantizeInput():
                    return record
        case RescaleInput():
            return {
                "node": operation.node,
                "scale": array_record(operation.scale),
                "activation_axis": operation.activation_axis,
                "weight_axis": operation.weight_axis,
            }
        case _:
            assert_never(operation)


def save_plans(path: Path, plans: tuple[QuantizationPlan, ...]) -> None:
    """Save complete stage decisions, including scalar/channel encoding arrays."""
    records = [
        PlanRecord(
            operations=[operation_record(operation) for operation in plan.operations],
            excluded=plan.excluded,
        )
        for plan in plans
    ]
    path.write_text(json.dumps(records, indent=2))


def save_execution(path: Path, execution: ExecutionReport | Absent) -> None:
    """Persist verified CPU operators or a typed reason the audit is absent."""
    path.write_text(execution.model_dump_json(indent=2))


def coverage(graph: Graph, plans: tuple[QuantizationPlan, ...]) -> Coverage:
    """Summarize constant-weight scope and the actual operations in saved plans."""
    eligible = {
        node.name
        for node in graph.nodes
        if node.op in ("Conv", "MatMul", "Gemm")
        and len(node.inputs) >= 2
        and node.inputs[1] in graph.weights
    }
    quantized: list[str] = []
    transformed: list[str] = []
    for plan in plans:
        for operation in plan.operations:
            match operation:
                case QuantizeInput(node=name, index=0):
                    if name in eligible:
                        quantized.append(name)
                case RescaleInput(node=name):
                    transformed.append(name)
                case QuantizeInput() | QuantizeOutput() | QuantizeConstant():
                    pass
                case _:
                    assert_never(operation)
    return Coverage(
        eligible_nodes=len(eligible),
        quantized_nodes=tuple(quantized),
        transformed_nodes=tuple(transformed),
        excluded_nodes=plans[-1].excluded if plans else (),
    )
