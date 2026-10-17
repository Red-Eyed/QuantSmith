"""Concrete ORT stage implementations for the declarative pipeline shell."""

from dataclasses import dataclass, field, replace
from typing import assert_never

from quantsmith.algorithms.contracts import Algorithm
from quantsmith.algorithms.shell import build_plan
from quantsmith.backends.onnx import describe, lower
from quantsmith.backends.onnx.integer_execution.audit import (
    UnverifiedNode,
    audit_execution,
)
from quantsmith.backends.onnx.integer_execution.planning import execution_rules
from quantsmith.backends.onnx.optimization import optimize
from quantsmith.backends.onnx.ort_scaling import lower_integer_scaling
from quantsmith.backends.onnx.ort_scaling.boundaries import with_boundaries
from quantsmith.calibration import Calibration, MinMax
from quantsmith.domain import Absent
from quantsmith.pipeline.contracts import CalibrationContext
from quantsmith.pipeline.policies import (
    Folding,
    FoldOnly,
    FunctionOutputs,
    JointFoldOnly,
    Outputs,
    ProducerOutputs,
    RuntimeScaling,
)
from quantsmith.pipeline.state import FloatGraph, QuantizationResult, QuantizedGraph
from quantsmith.plan import QuantizationPlan, RescaleInput
from quantsmith.result import Err, FailureKind, Ok, QuantSmithError, Result, failure
from quantsmith.rules import Exclude, Rules
from quantsmith.runtime import OnnxEvaluator


def _runtime_transform(
    value: FloatGraph, plan: QuantizationPlan
) -> Result[FloatGraph, QuantSmithError]:
    """Fold what is supported and carry explicitly permitted runtime scale debt."""
    match lower(value.revision.model, plan):
        case Err() as error:
            return error
        case Ok(model):
            pass
    revision = replace(
        value.revision, model=model, plans=value.revision.plans + (plan,)
    )
    nodes = {operation.node for operation in plan.operations}
    if not nodes:
        return Ok(replace(value, revision=revision))
    introduced = frozenset(
        node.input[0] for node in model.graph.node if node.name in nodes
    )
    pending = value.scaling | introduced
    match optimize(model):
        case Err() as error:
            return error
        case Ok(model):
            remaining = frozenset(
                node.output[0]
                for node in model.graph.node
                if node.op_type == "Div" and node.output[0] in pending
            )
            return Ok(FloatGraph(replace(revision, model=model), remaining))


@dataclass(frozen=True)
class TransformStage:
    """Declare a floating transformation before subsequent fresh calibration."""

    rules: Rules[Algorithm]
    folding: Folding = field(default_factory=RuntimeScaling)

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[FloatGraph, QuantSmithError]:
        """Reject quantization operations rather than inferring the stage's purpose."""
        match describe(value.revision.model):
            case Err() as error:
                return error
            case Ok(graph):
                pass
        match build_plan(
            graph,
            OnnxEvaluator(value.revision.model),
            context.samples,
            self.rules,
            histogram_bins=context.histogram_bins,
            max_channel_histogram_bytes=context.max_channel_histogram_bytes,
        ):
            case Err() as error:
                return error
            case Ok(plan):
                pass
        if any(
            not isinstance(operation, RescaleInput) for operation in plan.operations
        ):
            return failure(
                FailureKind.CONFLICT,
                "transform stage",
                "transformation stage requires a rescaling-only plan",
            )
        match self.folding:
            case RuntimeScaling():
                return _runtime_transform(value, plan)
            case FoldOnly():
                from quantsmith.backends.onnx.folding import fold_only

                folding = fold_only
            case JointFoldOnly():
                from quantsmith.backends.onnx.group_folding import fold_groups

                folding = fold_groups
            case _:
                assert_never(self.folding)
        match folding(value.revision.model, plan):
            case Err() as error:
                return error
            case Ok(payload):
                model, accepted, outcomes = payload
                revision = replace(
                    value.revision,
                    model=model,
                    plans=value.revision.plans + (accepted,),
                    folding=value.revision.folding + outcomes,
                )
                return Ok(replace(value, revision=revision))


@dataclass(frozen=True)
class QuantizationStage:
    """Declare static calibration and integer lowering of a floating graph."""

    rules: Rules[Algorithm]
    outputs: Outputs = field(default_factory=ProducerOutputs)
    boundary_calibration: Calibration = field(default_factory=MinMax)

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[QuantizedGraph, QuantSmithError]:
        """Recollect all statistics after transforms, preserving source exclusions."""
        match describe(value.revision.model):
            case Err() as error:
                return error
            case Ok(graph):
                pass
        configured = execution_rules(
            value.revision.model,
            graph,
            with_boundaries(
                value.revision.model,
                graph,
                self.rules,
                value.scaling,
                self.boundary_calibration,
            ),
            activation_function_aware=isinstance(self.outputs, FunctionOutputs),
        )
        match build_plan(
            graph,
            OnnxEvaluator(value.revision.model),
            context.samples,
            configured,
            histogram_bins=context.histogram_bins,
            max_channel_histogram_bytes=context.max_channel_histogram_bytes,
        ):
            case Err() as error:
                return error
            case Ok(plan):
                pass
        if any(isinstance(operation, RescaleInput) for operation in plan.operations):
            return failure(
                FailureKind.CONFLICT,
                "quantization stage",
                "quantization stage cannot contain rescaling operations",
            )
        match lower(value.revision.model, plan):
            case Err() as error:
                return error
            case Ok(model):
                pass
        if value.scaling:
            match lower_integer_scaling(model, value.scaling):
                case Err() as error:
                    return error
                case Ok(model):
                    pass
        exclusions: list[UnverifiedNode] = []
        for node in graph.nodes:
            match configured.resolve(node, graph):
                case Exclude(reason=reason):
                    exclusions.append(
                        UnverifiedNode(node=node.name, operator=node.op, reason=reason)
                    )
                case _:
                    pass
        revision = replace(
            value.revision, model=model, plans=value.revision.plans + (plan,)
        )
        return Ok(QuantizedGraph(revision, tuple(exclusions)))


@dataclass(frozen=True)
class ReuseQuantizersStage:
    """Reuse identical private activation quantizers before compiling and auditing."""

    def run(
        self, value: QuantizedGraph, context: CalibrationContext
    ) -> Result[QuantizedGraph, QuantSmithError]:
        """Preserve numerical plans and leave verification to the following audit."""
        from quantsmith.backends.onnx.quantizer_reuse import reuse_quantizers

        match reuse_quantizers(value.revision.model):
            case Err() as error:
                return error
            case Ok(model):
                return Ok(replace(value, revision=replace(value.revision, model=model)))


@dataclass(frozen=True)
class VerifyStage:
    """Require selected source operations to compile to integer CPU kernels."""

    def run(
        self, value: QuantizedGraph, context: CalibrationContext
    ) -> Result[QuantizationResult, QuantSmithError]:
        """Return compiled execution evidence independently of task quality."""
        revision = value.revision
        match audit_execution(
            revision.source, revision.model, revision.plans, value.exclusions
        ):
            case Err() as error:
                return error
            case Ok(execution):
                return Ok(
                    QuantizationResult(
                        model=revision.model,
                        plans=revision.plans,
                        execution=execution,
                        folding=revision.folding,
                        selection=value.selection,
                    )
                )


@dataclass(frozen=True)
class FinishTransform:
    """Inspect a transform-only program without claiming integer execution."""

    def run(
        self, value: FloatGraph, context: CalibrationContext
    ) -> Result[QuantizationResult, QuantSmithError]:
        """Expose a floating result and the reason no integer audit was performed."""
        revision = value.revision
        return Ok(
            QuantizationResult(
                model=revision.model,
                plans=revision.plans,
                execution=Absent(
                    reason="transform-only pipeline does not claim integer execution"
                ),
                folding=revision.folding,
            )
        )
