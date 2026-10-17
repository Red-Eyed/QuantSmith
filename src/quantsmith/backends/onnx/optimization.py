"""Conservative, owned ONNX rewrites before the next calibration revision."""

from collections import Counter

import numpy as np
import onnx
from onnx import ModelProto, NodeProto, TensorProto, numpy_helper

from quantsmith.backends.onnx import Names, admit, normalize, remove_unused_initializers
from quantsmith.domain import FloatArray
from quantsmith.result import QuantSmithError, Result


def _constant(model: ModelProto, name: str) -> FloatArray | None:
    """Borrow a finite FP32 initializer or decline the rewrite."""
    for tensor in model.graph.initializer:
        if tensor.name == name and tensor.data_type == TensorProto.FLOAT:
            value = np.asarray(numpy_helper.to_array(tensor), dtype=np.float32)
            if np.all(np.isfinite(value)):
                return value
    return None


def _parameters(
    model: ModelProto, producer: NodeProto, scale: FloatArray
) -> list[tuple[int, FloatArray]]:
    """Calculate replacements only for exact NCHW output-channel broadcasts."""
    if scale.ndim < 3 or scale.shape[:1] != (1,):
        return []
    if any(size != 1 for size in scale.shape[2:]) or np.any(scale <= 0):
        return []
    channels = scale.reshape(-1)
    if producer.op_type == "Conv":
        weight = _constant(model, producer.input[1])
        if weight is None or weight.ndim != scale.ndim:
            return []
        if weight.shape[0] != channels.size:
            return []
        replacements = [
            (1, weight / channels.reshape((-1,) + (1,) * (weight.ndim - 1)))
        ]
        bias_index = 2
    elif producer.op_type == "BatchNormalization":
        if len(producer.input) != 5 or any(
            attr.name == "training_mode" and attr.i != 0 for attr in producer.attribute
        ):
            return []
        gain = _constant(model, producer.input[1])
        if gain is None or gain.shape != channels.shape:
            return []
        replacements = [(1, gain / channels)]
        bias_index = 2
    else:
        return []
    if len(producer.input) > bias_index and producer.input[bias_index]:
        bias = _constant(model, producer.input[bias_index])
        if bias is None or bias.shape != channels.shape:
            return []
        replacements.append((bias_index, bias / channels))
    if any(not np.all(np.isfinite(value)) for _, value in replacements):
        return []
    return replacements


def _optimize(model: ModelProto) -> ModelProto:
    """Fold single-consumer channel divisions without modifying caller storage."""
    result = normalize(model)
    names = Names(result)
    producers = {output: node for node in result.graph.node for output in node.output}
    uses = Counter(name for node in result.graph.node for name in node.input)
    outputs = {value.name for value in result.graph.output}
    ranks = {
        value.name: len(value.type.tensor_type.shape.dim)
        for value in (*result.graph.input, *result.graph.value_info)
    }
    removed: set[str] = set()
    removed_values: set[str] = set()
    for division in result.graph.node:
        if division.domain or division.op_type != "Div" or len(division.input) != 2:
            continue
        source = division.input[0]
        producer = producers.get(source)
        if producer is None or producer.domain or len(producer.output) != 1:
            continue
        if uses[source] != 1 or source in outputs:
            continue
        scale = _constant(result, division.input[1])
        if scale is None:
            continue
        if producer.op_type == "BatchNormalization" and ranks.get(source) != scale.ndim:
            continue
        replacements = _parameters(result, producer, scale)
        if not replacements:
            continue
        for index, value in replacements:
            name = names.new(producer.name + "_folded_scale")
            result.graph.initializer.append(numpy_helper.from_array(value, name))
            producer.input[index] = name
        producer.output[0] = division.output[0]
        producers[division.output[0]] = producer
        removed.add(division.name)
        removed_values.add(source)
    retained = [node for node in result.graph.node if node.name not in removed]
    del result.graph.node[:]
    result.graph.node.extend(retained)
    metadata = [
        value for value in result.graph.value_info if value.name not in removed_values
    ]
    del result.graph.value_info[:]
    result.graph.value_info.extend(metadata)
    remove_unused_initializers(result)
    onnx.checker.check_model(result)
    return onnx.shape_inference.infer_shapes(result, strict_mode=True)


def optimize(model: ModelProto) -> Result[ModelProto, QuantSmithError]:
    """Return an owned graph with eligible Conv/BatchNorm divisions folded.

    Shared producer outputs, custom operators, and unsupported broadcasts remain
    unchanged. Rewrites preserve real arithmetic, with FP32 rounding differences;
    calibration must use this returned revision rather than prior statistics.
    """
    return admit("ONNX optimization", lambda: _optimize(model))
