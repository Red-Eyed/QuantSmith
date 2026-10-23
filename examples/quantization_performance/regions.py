"""Extract source operations and small regions without changing the caller graph."""

from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import onnx
from onnx import ModelProto, NodeProto, TensorProto, helper
from onnx.utils import Extractor
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class TraceArgs(BaseModel):
    """Admit only the operator identity from external ORT profiling attributes."""

    model_config = ConfigDict(extra="ignore", strict=True)
    op_name: str = ""


class TraceEvent(BaseModel):
    """Read source-mappable node names and instrumented durations."""

    model_config = ConfigDict(extra="ignore", strict=True)
    cat: str = ""
    name: str = ""
    dur: float = Field(default=0, ge=0)
    args: TraceArgs = Field(default_factory=TraceArgs)


class TensorShape(BaseModel):
    """Describe a concrete ONNX boundary tensor, including scalar shapes."""

    name: str
    dtype: int
    dimensions: tuple[int, ...]


class Region(BaseModel):
    """Identify original source nodes and concrete, independently replayable edges."""

    source: str
    operator: str
    nodes: tuple[str, ...]
    inputs: tuple[TensorShape, ...]
    outputs: tuple[TensorShape, ...]
    weight_shape: tuple[int, ...]
    groups: int
    strides: tuple[int, ...]
    dilations: tuple[int, ...]
    pads: tuple[int, ...]
    profiled_us: float


def integer_attribute(node: NodeProto, name: str, default: int) -> int:
    """Read an integer attribute without an unchecked dynamic attribute adapter."""
    return next((value.i for value in node.attribute if value.name == name), default)


def tuple_attribute(node: NodeProto, name: str) -> tuple[int, ...]:
    """Read a repeated integer attribute, leaving absent defaults explicit."""
    return next(
        (tuple(value.ints) for value in node.attribute if value.name == name), ()
    )


def profile_costs(model: ModelProto, trace: Path) -> dict[str, float]:
    """Match exact source names and ORT token suffixes; ignore unrelated operators."""
    names = {
        node.name
        for node in model.graph.node
        if node.op_type in ("Conv", "MatMul", "Gemm")
    }
    totals: dict[str, float] = defaultdict(float)
    for event in TypeAdapter(list[TraceEvent]).validate_json(trace.read_text()):
        if event.cat != "Node" or not event.name.endswith("_kernel_time"):
            continue
        name = event.name.removesuffix("_kernel_time")
        source = name.split("_token_", 1)[0]
        if source in names:
            totals[source] += event.dur
    return dict(totals)


def select_sources(
    model: ModelProto, costs: dict[str, float], count: int
) -> tuple[str, ...]:
    """Take expensive profiled operations plus the most costly matrix projection."""
    selected = sorted(costs, key=lambda name: (-costs[name], name))[:count]
    projections = [
        node.name
        for node in model.graph.node
        if node.op_type in ("MatMul", "Gemm") and node.name in costs
    ]
    if projections:
        projection = max(projections, key=lambda name: costs[name])
        if projection not in selected:
            selected.append(projection)
    return tuple(selected)


def region_nodes(model: ModelProto, source: str, hops: int) -> tuple[NodeProto, ...]:
    """Follow a private chain of activations/residual/broadcast/pooling operations."""
    nodes = {node.name: node for node in model.graph.node}
    consumers: dict[str, list[NodeProto]] = defaultdict(list)
    public = {value.name for value in model.graph.output}
    for node in model.graph.node:
        for name in node.input:
            consumers[name].append(node)
    selected = [nodes[source]]
    for _ in range(hops):
        output = selected[-1].output
        if len(output) != 1 or output[0] in public:
            break
        following = consumers[output[0]]
        if len(following) != 1:
            break
        node = following[0]
        if node.domain or node.op_type not in ("Relu", "Clip", "Add", "Mul", "MaxPool"):
            break
        selected.append(node)
    return tuple(selected)


def shape_of(model: ModelProto, name: str) -> TensorShape:
    """Require concrete floating boundary dimensions before a region is admitted."""
    values = (*model.graph.input, *model.graph.value_info, *model.graph.output)
    value = next(value for value in values if value.name == name)
    tensor = value.type.tensor_type
    if tensor.elem_type != TensorProto.FLOAT:
        raise ValueError(f"region boundary {name} must be FP32")
    if any(not dimension.HasField("dim_value") for dimension in tensor.shape.dim):
        raise ValueError(f"region boundary {name} must have concrete dimensions")
    return TensorShape(
        name=name,
        dtype=tensor.elem_type,
        dimensions=tuple(dimension.dim_value for dimension in tensor.shape.dim),
    )


def extract_region(
    model: ModelProto, source: str, hops: int, profiled_us: float
) -> tuple[ModelProto, Region]:
    """Preserve constant ancestors while cutting only dynamic external edges."""
    inferred = onnx.shape_inference.infer_shapes(deepcopy(model))
    selected = region_nodes(inferred, source, hops)
    internal = {name for node in selected for name in node.output}
    constants = {value.name for value in inferred.graph.initializer}
    constants.update(
        name
        for node in inferred.graph.node
        if node.op_type == "Constant"
        for name in node.output
    )
    inputs = tuple(
        dict.fromkeys(
            name
            for node in selected
            for name in node.input
            if name and name not in constants and name not in internal
        )
    )
    outputs = tuple(selected[-1].output)
    extracted = Extractor(inferred).extract_model(list(inputs), list(outputs))
    weight = next(
        value
        for value in inferred.graph.initializer
        if value.name == selected[0].input[1]
    )
    region = Region(
        source=source,
        operator=selected[0].op_type,
        nodes=tuple(node.name for node in selected),
        inputs=tuple(shape_of(inferred, name) for name in inputs),
        outputs=tuple(shape_of(inferred, name) for name in outputs),
        weight_shape=tuple(weight.dims),
        groups=integer_attribute(selected[0], "group", 1),
        strides=tuple_attribute(selected[0], "strides"),
        dilations=tuple_attribute(selected[0], "dilations"),
        pads=tuple_attribute(selected[0], "pads"),
        profiled_us=profiled_us,
    )
    onnx.checker.check_model(extracted)
    return extracted, region


def capture_graph(model: ModelProto, boundaries: tuple[TensorShape, ...]) -> ModelProto:
    """Expose just one region's inputs; an inference retains no other activations."""
    captured = deepcopy(model)
    del captured.graph.output[:]
    captured.graph.output.extend(
        helper.make_tensor_value_info(value.name, value.dtype, value.dimensions)
        for value in boundaries
    )
    return captured
