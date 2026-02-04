from typing import List, Optional, Dict, Any

from state import GlobalState
from entities.base import EntityBase


class ExecutionDecision:
    """
    Immutable execution decision returned by a policy.
    """

    def __init__(
        self,
        entity: EntityBase,
        device: str = "cpu",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.entity = entity
        self.device = device
        self.config = config or {}


class ExecutionPolicy:
    """
    Base class for all execution policies.

    A policy decides WHICH entity to run next and HOW.
    """

    def decide(
        self,
        state: GlobalState,
        entities: List[EntityBase],
    ) -> Optional[ExecutionDecision]:
        raise NotImplementedError