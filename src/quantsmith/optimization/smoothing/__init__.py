"""Output-error selection independent of graph formats and runtime execution."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from quantsmith.result import FailureKind, Ok, QuantSmithError, Result, failure


class ContractionOutputs(BaseModel):
    """Observe baseline-quantized Conv, MatMul and Gemm output tensors."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["contractions"] = Field(default="contractions")


class ModelOutputs(BaseModel):
    """Score the model's public floating outputs in their original coordinates."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["model"] = Field(default="model")


class ObservedTensors(BaseModel):
    """Choose explicit source tensor identities for reconstruction scoring."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: Literal["tensors"] = Field(default="tensors")
    names: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_names(self) -> Self:
        """Reject empty identities and repeated objective terms."""
        if any(not name for name in self.names) or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError("observed tensor names must be nonempty and unique")
        return self


type Observations = ModelOutputs | ContractionOutputs | ObservedTensors


class CandidateScore(BaseModel):
    """Summarize streamed elementwise output error for one audited candidate."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    candidate: str = Field(min_length=1)
    mean_squared_error: float = Field(ge=0, allow_inf_nan=False)
    maximum_absolute_error: float = Field(ge=0, allow_inf_nan=False)
    batches: int = Field(gt=0)
    integer_nodes: int = Field(ge=0)
    folded_nodes: tuple[str, ...] = Field(default=())


class SelectionReport(BaseModel):
    """Record a reproducible decision without claiming task or latency improvement."""

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")
    selected: str = Field(min_length=1)
    outputs: tuple[str, ...] = Field(min_length=1)
    candidates: tuple[CandidateScore, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent(self) -> Self:
        """Require unique candidates and a selected identity present in the report."""
        names = tuple(score.candidate for score in self.candidates)
        if len(set(names)) != len(names) or self.selected not in names:
            raise ValueError(
                "selection requires unique candidates and a present winner"
            )
        if len(set(self.outputs)) != len(self.outputs) or any(
            not name for name in self.outputs
        ):
            raise ValueError("selection output names must be nonempty and unique")
        return self


def prefer(candidate: CandidateScore, incumbent: CandidateScore) -> bool:
    """Minimize element-weighted MSE; retain declaration order on exact ties."""
    return candidate.mean_squared_error < incumbent.mean_squared_error


def select(
    scores: tuple[CandidateScore, ...], outputs: tuple[str, ...]
) -> Result[SelectionReport, QuantSmithError]:
    """Choose a declared candidate with no execution, mutation or hidden tolerance."""
    if not scores or not outputs:
        return failure(
            FailureKind.EMPTY, "smoothing selection", "scores and outputs are required"
        )
    names = tuple(score.candidate for score in scores)
    if (
        len(set(names)) != len(names)
        or len(set(outputs)) != len(outputs)
        or any(not name for name in outputs)
    ):
        return failure(
            FailureKind.INVALID_DATA,
            "smoothing selection",
            "candidate and output identities must be unique and nonempty",
        )
    winner = scores[0]
    for score in scores[1:]:
        if prefer(score, winner):
            winner = score
    return Ok(
        SelectionReport(selected=winner.candidate, outputs=outputs, candidates=scores)
    )
