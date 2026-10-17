"""Complete built-in static encodings with calibrated producer boundaries."""

from dataclasses import dataclass

import numpy as np
from onnx import ModelProto

from quantsmith.algorithms.contracts import Algorithm, Needs, Statistics
from quantsmith.algorithms.encoding import encode
from quantsmith.algorithms.static import StaticW8A8
from quantsmith.backends.onnx.integer_execution.activation_function_ranges import (
    ActivationFunctionRanges,
    activation_function_outputs,
)
from quantsmith.backends.onnx.integer_execution.support import restriction
from quantsmith.backends.onnx.ort_scaling.boundaries import BoundaryNames
from quantsmith.calibration import Requirement
from quantsmith.domain import Absent, Encoding, Graph, Node, PerTensor
from quantsmith.pipeline.policies import (
    FloatMatMulOutputs,
    MatMulOutputs,
    QuantizedMatMulOutputs,
)
from quantsmith.plan import QuantizationPlan, QuantizeInput, QuantizeOutput
from quantsmith.result import (
    Err,
    FailureKind,
    Ok,
    QuantSmithError,
    Result,
    failure,
    validate,
)
from quantsmith.rules import Exclude, Rule, Rules

_QUANTIZED_OUTPUTS = QuantizedMatMulOutputs()


def accumulator_safe_plan(
    node: Node, graph: Graph, plan: QuantizationPlan
) -> QuantizationPlan:
    """Widen weight scales where constant biases would overflow INT32 accumulators."""
    if node.op not in {"Conv", "Gemm"} or len(node.inputs) < 3 or not node.inputs[2]:
        return plan
    inputs = {
        operation.index: operation
        for operation in plan.operations
        if isinstance(operation, QuantizeInput)
    }
    activation, weight = inputs[0].encoding, inputs[1].encoding
    bias = graph.weights[node.inputs[2]]
    if not np.all(np.isfinite(bias)):
        raise ValueError("bias must be finite")
    minimum = (
        np.abs(bias.astype(np.float64))
        * 1.0001
        / np.iinfo(np.int32).max
        / activation.scale.astype(np.float64)
    )
    wide = np.maximum(weight.scale.astype(np.float64), minimum)
    if np.any(wide > np.finfo(np.float32).max):
        raise ValueError("bias requires unrepresentable weight scales")
    scales = wide.astype(np.float32)
    encoding = Encoding(
        scale=scales, zero_point=weight.zero_point, granularity=weight.granularity
    )
    updated = QuantizeInput(node=node.name, index=1, encoding=encoding)
    return QuantizationPlan(
        operations=tuple(
            updated if operation is inputs[1] else operation
            for operation in plan.operations
        ),
        excluded=plan.excluded,
    )


@dataclass(frozen=True)
class ExecutionQuantizer:
    """Adapt static planning without changing the standalone algorithm contract."""

    algorithm: StaticW8A8

    def requirements(self, node: Node, graph: Graph) -> Result[Needs, QuantSmithError]:
        """Observe outputs on the same graph revision as activation inputs."""
        match self.algorithm.requirements(node, graph):
            case Err() as error:
                return error
            case Ok(needs):
                output = Requirement(tensor=node.outputs[0])
                return Ok(
                    Needs(
                        ranges=needs.ranges + (output,),
                        histograms=needs.histograms
                        + (
                            (output,)
                            if self.algorithm.calibration.needs_histogram
                            else ()
                        ),
                    )
                )

    def plan(
        self, node: Node, graph: Graph, stats: Statistics
    ) -> Result[QuantizationPlan, QuantSmithError]:
        """Preserve algorithm encodings and expose additional output rounding."""
        match self.algorithm.plan(node, graph, stats):
            case Err() as error:
                return error
            case Ok(plan):
                pass
        match validate(node.name, lambda: accumulator_safe_plan(node, graph, plan)):
            case Err() as error:
                return error
            case Ok(plan):
                pass
        request = Requirement(tensor=node.outputs[0])
        if request not in stats.ranges:
            return failure(
                FailureKind.INVALID_DATA,
                node.name,
                "output statistics were not collected",
            )
        histogram = stats.histograms.get(
            request, Absent(reason="histogram was not collected")
        )
        match self.algorithm.calibration.interval(stats.ranges[request], histogram):
            case Err() as error:
                return error
            case Ok(interval):
                encoding = encode(
                    interval,
                    self.algorithm.activation_type,
                    self.algorithm.activation_symmetric,
                    PerTensor(),
                )
        return plan.then(
            QuantizationPlan(
                operations=(QuantizeOutput(node=node.name, index=0, encoding=encoding),)
            )
        )


def _adapt(
    model: ModelProto,
    graph: Graph,
    node: Node,
    decision: Algorithm | Exclude,
    matmul_outputs: MatMulOutputs,
) -> Algorithm | Exclude:
    """Complete built-in static methods while preserving custom implementations."""
    match decision:
        case StaticW8A8():
            match restriction(model, graph, node, decision):
                case str() as reason:
                    return Exclude(reason=reason)
                case Absent():
                    if node.op == "MatMul" and isinstance(
                        matmul_outputs, FloatMatMulOutputs
                    ):
                        from quantsmith.backends.onnx.integer_execution import (
                            float_output,
                        )

                        return float_output.float_matmul_algorithm(decision)
                    return ExecutionQuantizer(decision)
        case _:
            return decision


def execution_rules(
    model: ModelProto,
    graph: Graph,
    rules: Rules[Algorithm],
    *,
    activation_function_aware: bool = False,
    matmul_outputs: MatMulOutputs = _QUANTIZED_OUTPUTS,
) -> Rules[Algorithm]:
    """Own execution policy in the ORT shell, outside numerical algorithms."""
    configured = Rules[Algorithm](
        default=Exclude(reason="outside the current graph revision"),
        overrides=tuple(
            Rule(
                selector=BoundaryNames(frozenset({node.name})),
                decision=_adapt(
                    model, graph, node, rules.resolve(node, graph), matmul_outputs
                ),
            )
            for node in graph.nodes
        ),
    )
    if not activation_function_aware:
        return configured
    outputs = activation_function_outputs(model)

    def observe(node: Node) -> Algorithm | Exclude:
        """Keep custom algorithms and exclusions outside the built-in policy."""
        decision = configured.resolve(node, graph)
        match decision:
            case ExecutionQuantizer() if node.name in outputs:
                return ActivationFunctionRanges(decision, outputs[node.name])
            case _:
                return decision

    return Rules(
        default=configured.default,
        overrides=tuple(
            Rule(selector=BoundaryNames(frozenset({node.name})), decision=observe(node))
            for node in graph.nodes
        ),
    )
