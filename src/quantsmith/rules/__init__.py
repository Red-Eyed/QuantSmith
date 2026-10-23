"""Ordered rules with open selectors and explicit exclusion decisions."""

from fnmatch import fnmatchcase
from typing import Generic, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypeVar

from quantsmith.domain import Graph, Node

T = TypeVar("T", infer_variance=True)


@runtime_checkable
class Selector(Protocol):
    """Select graph nodes without mutation or backend dependencies."""

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Return whether this node belongs to the selected graph region."""
        ...


class ByOperator(BaseModel):
    """Select any of the supplied operator names."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    operators: frozenset[str] = Field()

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Match operator identity."""
        return node.op in self.operators


class ByName(BaseModel):
    """Select node names using shell-style patterns."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    pattern: str = Field()

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Match the full name case-sensitively."""
        return fnmatchcase(node.name, self.pattern)


class ByTensor(BaseModel):
    """Select nodes touching a named input or output tensor."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    name: str = Field()

    def matches(self, node: Node, graph: Graph, /) -> bool:
        """Match connectivity on either side of the node."""
        return self.name in node.inputs or self.name in node.outputs


class Exclude(BaseModel):
    """Keep a node floating point and preserve the caller's reason."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    reason: str = Field()


class Rule(BaseModel, Generic[T]):
    """Associate a selector with a configured implementation or exclusion."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    selector: Selector = Field()
    decision: T | Exclude = Field()


class Rules(BaseModel, Generic[T]):
    """Resolve rules in declaration order; the last matching rule wins."""

    model_config = ConfigDict(
        frozen=True, strict=True, extra="forbid", arbitrary_types_allowed=True
    )
    default: T | Exclude = Field()
    overrides: tuple[Rule[T], ...] = Field(default=())

    def resolve(self, node: Node, graph: Graph) -> T | Exclude:
        """Return the complete decision without modifying rule or graph state."""
        decision = self.default
        for rule in self.overrides:
            if rule.selector.matches(node, graph):
                decision = rule.decision
        return decision
