"""Build synthetic Transformer fragments and fixed-grid integer counterparts."""

from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Self, override

import numpy as np
import onnx
import torch
from einops import rearrange
from numpy.typing import NDArray
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator
from stackformers import (
    CrossAttention,
    CrossAttentionConfig,
    NoPosEncoding,
    TransformerEncoder,
    make_padded_input,
    plain_encoder_config,
)
from stackformers.feedforward.config import GELUConfig, SwiGLUConfig
from stackformers.feedforward.gelu import GELUFFN
from stackformers.feedforward.swiglu import SwiGLU
from torch import Tensor, nn

from quantsmith.domain import Absent

_ARRAY = TypeAdapter(
    NDArray[np.generic], config=ConfigDict(arbitrary_types_allowed=True)
)

type Fragment = Literal[
    "linear",
    "bias",
    "gelu",
    "silu",
    "residual",
    "layernorm",
    "rmsnorm",
    "scalar_add",
    "channel_add",
    "full_add",
    "strided",
    "mlp_gelu",
    "swiglu",
    "qkv_separate",
    "qkv_fused",
    "qk",
    "av",
    "attention",
    "transformer",
]
FRAGMENTS: tuple[Fragment, ...] = (
    "linear",
    "bias",
    "gelu",
    "silu",
    "residual",
    "layernorm",
    "rmsnorm",
    "scalar_add",
    "channel_add",
    "full_add",
    "strided",
    "mlp_gelu",
    "swiglu",
    "qkv_separate",
    "qkv_fused",
    "qk",
    "av",
    "attention",
    "transformer",
)


class Geometry(BaseModel, frozen=True):
    """Distinguish query tokens from the key/value context length."""

    batch: int = Field(default=1, gt=0)
    tokens: int = Field(gt=0)
    context: int = Field(gt=0)
    width: int = Field(gt=0)
    heads: int = Field(default=4, gt=0)

    @model_validator(mode="after")
    def divisible_heads(self) -> Self:
        """Require a lossless split of hidden width into heads."""
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        return self


def tensor_output(module: nn.Module, inputs: Tensor) -> Tensor:
    """Validate the untyped Module call boundary without accepting other payloads."""
    value = module(inputs)
    if not isinstance(value, Tensor):
        raise TypeError("fragment module must return a Tensor")
    return value


class Projection(nn.Module):
    """Expose the same Linear beside different normalization and epilogue patterns."""

    addition: Tensor

    def __init__(self, kind: Fragment, geometry: Geometry) -> None:
        """Keep broadcast operands fixed and distinct from runtime residual inputs."""
        super().__init__()
        self.kind = kind
        self.linear = nn.Linear(geometry.width, geometry.width, bias=kind == "bias")
        self.norm: nn.Module = nn.Identity()
        if kind == "layernorm":
            self.norm = nn.LayerNorm(geometry.width)
        if kind == "rmsnorm":
            self.norm = nn.RMSNorm(geometry.width)
        shape = {"scalar_add": (1,), "channel_add": (geometry.width,)}.get(
            kind, (geometry.batch, geometry.tokens, geometry.width)
        )
        self.register_buffer("addition", torch.randn(shape) * 0.1)

    @override
    def forward(self, inputs: Tensor, memory: Tensor) -> Tensor:
        """Retain input normalization, activation, residual and broadcast boundaries."""
        output = tensor_output(self.linear, tensor_output(self.norm, inputs))
        match self.kind:
            case "gelu":
                return torch.nn.functional.gelu(output, approximate="tanh")
            case "silu":
                return torch.nn.functional.silu(output)
            case "residual":
                return output + inputs
            case "full_add":
                return output + memory
            case "scalar_add" | "channel_add":
                return output + self.addition
            case _:
                return output


class QKV(nn.Module):
    """Compare three projections with one packed projection using identical weights."""

    def __init__(self, width: int, fused: bool) -> None:
        """Initialize the separate weights first so both representations match."""
        super().__init__()
        separate = [nn.Linear(width, width, bias=False) for _ in range(3)]
        packed = nn.Linear(width, 3 * width, bias=False)
        with torch.no_grad():
            packed.weight.copy_(torch.cat([layer.weight for layer in separate]))
        self.projections = nn.ModuleList([packed] if fused else separate)

    @override
    def forward(self, inputs: Tensor, memory: Tensor) -> Tensor:
        """Return packed Q/K/V values in both representations."""
        return torch.cat(
            [tensor_output(layer, inputs) for layer in self.projections], dim=-1
        )


class Matmul(nn.Module):
    """Isolate dynamic-operand QK or AV without implementing complete attention."""

    def __init__(self, kind: Literal["qk", "av"], geometry: Geometry) -> None:
        """Bind head geometry while leaving both operands as runtime inputs."""
        super().__init__()
        self.kind, self.heads = kind, geometry.heads

    @override
    def forward(self, inputs: Tensor, memory: Tensor) -> Tensor:
        """Preserve attention head transposes and batch broadcasting."""
        values = rearrange(memory, "b s (h d) -> b h s d", h=self.heads)
        if self.kind == "av":
            return torch.matmul(inputs, values)
        query = rearrange(inputs, "b t (h d) -> b h t d", h=self.heads)
        return torch.matmul(query, rearrange(values, "b h s d -> b h d s"))


class StackBlock(nn.Module):
    """Use Stackformers for complete attention, FFN and encoder blocks."""

    def __init__(self, kind: Fragment, geometry: Geometry) -> None:
        """Use an explicit cross-context attention case and a causal encoder case."""
        super().__init__()
        self.kind = kind
        self.block: nn.Module
        match kind:
            case "attention":
                self.block = CrossAttention(
                    CrossAttentionConfig(
                        dim=geometry.width,
                        heads=geometry.heads,
                        dim_head=geometry.width // geometry.heads,
                    ),
                    NoPosEncoding(),
                )
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
            case "mlp_gelu":
                self.block = GELUFFN(GELUConfig(dim=geometry.width, mult=4))
            case "swiglu":
                self.block = SwiGLU(SwiGLUConfig(dim=geometry.width, mult=4))
            case _:
                raise ValueError("not a Stackformers fragment")

    @override
    def forward(self, inputs: Tensor, memory: Tensor) -> Tensor:
        """Construct padded sequence contracts only for attention-bearing modules."""
        if isinstance(self.block, CrossAttention):
            query = make_padded_input(
                inputs, torch.ones(inputs.shape[:2], dtype=torch.bool)
            )
            context = make_padded_input(
                memory, torch.ones(memory.shape[:2], dtype=torch.bool)
            )
            return self.block.forward(query, context)
        if isinstance(self.block, TransformerEncoder):
            sequence = make_padded_input(
                inputs, torch.ones(inputs.shape[:2], dtype=torch.bool)
            )
            return self.block.forward(sequence)
        return tensor_output(self.block, inputs)


def build_fragment(kind: Fragment, geometry: Geometry, seed: int) -> nn.Module:
    """Initialize deterministic random weights without training or dataset access."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        match kind:
            case "qkv_separate" | "qkv_fused":
                module = QKV(geometry.width, kind == "qkv_fused")
            case "qk" | "av":
                module = Matmul(kind, geometry)
            case "mlp_gelu" | "swiglu" | "attention" | "transformer":
                module = StackBlock(kind, geometry)
            case _:
                module = Projection(kind, geometry)
    return module.eval()


def synthetic_inputs(
    kind: Fragment, geometry: Geometry, seed: int
) -> tuple[Tensor, Tensor]:
    """Generate FP32 inputs; AV uses probability-simplex attention weights."""
    generator = torch.Generator().manual_seed(seed)
    width = geometry.width * (2 if kind == "strided" else 1)
    inputs = torch.randn(geometry.batch, geometry.tokens, width, generator=generator)
    if kind == "strided":
        inputs = inputs[:, :, ::2]
    if kind == "av":
        inputs = torch.randn(
            geometry.batch,
            geometry.heads,
            geometry.tokens,
            geometry.context,
            generator=generator,
        ).softmax(dim=-1)
    memory = torch.randn(
        geometry.batch,
        geometry.tokens if kind == "full_add" else geometry.context,
        geometry.width,
        generator=generator,
    )
    return inputs, memory


def resize_axis(value: Tensor, axis: int, length: int) -> Tensor:
    """Select a representative length without tying export to a length-one example."""
    indices = torch.arange(length) % value.shape[axis]
    return value.index_select(axis, indices)


def dynamic_export_inputs(
    module: nn.Module, inputs: tuple[Tensor, Tensor]
) -> tuple[
    tuple[Tensor, Tensor],
    tuple[dict[int, torch.export.Dim], dict[int, torch.export.Dim]],
]:
    """Declare independent query/KV lengths and use nonsingleton trace examples."""
    query = torch.export.Dim("query_length", min=1)
    context = torch.export.Dim("context_length", min=1)
    first, second = inputs
    match module:
        case Matmul(kind="av"):
            first = resize_axis(resize_axis(first, 2, 3), 3, 5)
            second = resize_axis(second, 1, 5)
            return (first, second), ({2: query, 3: context}, {1: context})
        case Projection(kind="full_add"):
            first, second = resize_axis(first, 1, 3), resize_axis(second, 1, 3)
            return (first, second), ({1: query}, {1: query})
        case _:
            first, second = resize_axis(first, 1, 3), resize_axis(second, 1, 5)
            return (first, second), ({1: query}, {1: context})


def fragment_output(module: nn.Module, inputs: tuple[Tensor, Tensor]) -> Tensor:
    """Check the fragment's two-input execution boundary once."""
    result = module(*inputs)
    if not isinstance(result, Tensor):
        raise TypeError("fragment must return a Tensor")
    return result


@dataclass(frozen=True)
class OperandGrid:
    """Carry one encoded operand and the exact nodes/constants needed to produce it."""

    quantized: str
    scale: str
    zero: str
    nodes: tuple[NodeProto, ...]
    constants: tuple[TensorProto, ...]


def encode_operand(
    operand: str,
    constant: TensorProto | Absent,
    prefix: str,
    position: int,
    signed: bool,
    activation_range: float,
) -> OperandGrid:
    """Encode weights offline and activations at runtime."""
    name, scale, zero = (
        f"{prefix}_q{position}",
        f"{prefix}_s{position}",
        f"{prefix}_z{position}",
    )
    dtype = np.int8 if signed or position == 1 else np.uint8
    offset = 0 if dtype == np.int8 else 128
    step = activation_range / 127
    nodes: tuple[NodeProto, ...] = ()
    constants: list[TensorProto] = []
    match constant:
        case TensorProto():
            values = _ARRAY.validate_python(numpy_helper.to_array(constant))
            if values.dtype != np.float32 or not np.all(np.isfinite(values)):
                raise ValueError("synthetic MatMul constants must be finite FP32")
            weights = np.asarray(values, dtype=np.float32)
            step = max(float(np.abs(weights).max()) / 127, 1e-8)
            encoded = np.clip(
                np.rint(weights / step) + offset,
                np.iinfo(dtype).min,
                np.iinfo(dtype).max,
            ).astype(dtype)
            constants.append(numpy_helper.from_array(encoded, name))
        case Absent():
            nodes = (
                onnx.helper.make_node("QuantizeLinear", [operand, scale, zero], [name]),
            )
    constants.extend(
        [
            numpy_helper.from_array(np.array(step, dtype=np.float32), scale),
            numpy_helper.from_array(np.array(offset, dtype=dtype), zero),
        ]
    )
    return OperandGrid(name, scale, zero, nodes, tuple(constants))


def integer_nodes(
    node: NodeProto,
    left: OperandGrid,
    right: OperandGrid,
    prefix: str,
    activation_range: float,
    output: Literal["float", "quantized"],
    signed: bool,
) -> tuple[tuple[NodeProto, ...], tuple[TensorProto, ...]]:
    """Keep float outputs and output requantization as distinct kernel contracts."""
    if output == "float":
        kernel = onnx.helper.make_node(
            "MatMulIntegerToFloat",
            [
                left.quantized,
                right.quantized,
                left.scale,
                right.scale,
                left.zero,
                right.zero,
            ],
            list(node.output),
            domain="com.microsoft",
            name=prefix,
        )
        return (kernel,), ()
    scale, zero = f"{prefix}_sy", f"{prefix}_zy"
    constants = (
        numpy_helper.from_array(
            np.array(activation_range / 127, dtype=np.float32), scale
        ),
        numpy_helper.from_array(
            np.array(0 if signed else 128, dtype=np.int8 if signed else np.uint8), zero
        ),
    )
    nodes = (
        onnx.helper.make_node(
            "QLinearMatMul",
            [
                left.quantized,
                left.scale,
                left.zero,
                right.quantized,
                right.scale,
                right.zero,
                scale,
                zero,
            ],
            [f"{prefix}_out"],
            name=prefix,
        ),
        onnx.helper.make_node(
            "DequantizeLinear", [f"{prefix}_out", scale, zero], list(node.output)
        ),
    )
    return nodes, constants


def fixed_onnx(
    source: ModelProto,
    activation_range: float,
    output: Literal["float", "quantized"],
    signed: bool,
    scope: Literal["weights", "all"],
) -> ModelProto:
    """Replace MatMul with fixed-grid integer kernels in an owned graph snapshot."""
    result = deepcopy(source)
    constants = {value.name: value for value in result.graph.initializer}
    nodes: list[NodeProto] = []
    for index, node in enumerate(result.graph.node):
        if node.op_type != "MatMul" or (
            scope == "weights" and node.input[1] not in constants
        ):
            nodes.append(deepcopy(node))
            continue
        prefix = f"synthetic_int8_{index}"
        operands = [
            encode_operand(
                name,
                constants.get(name, Absent(reason="runtime operand")),
                prefix,
                position,
                signed,
                activation_range,
            )
            for position, name in enumerate(node.input)
        ]
        left, right = operands
        kernels, extra_constants = integer_nodes(
            node, left, right, prefix, activation_range, output, signed
        )
        nodes.extend((*left.nodes, *right.nodes, *kernels))
        result.graph.initializer.extend(
            (*left.constants, *right.constants, *extra_constants)
        )
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    if output == "float" and not any(
        item.domain == "com.microsoft" for item in result.opset_import
    ):
        result.opset_import.append(onnx.helper.make_opsetid("com.microsoft", 1))
    onnx.checker.check_model(result)
    return result
