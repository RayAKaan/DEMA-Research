from abc import ABC, abstractmethod
from typing import List, Iterable

from state import GlobalState


class EntityBase(ABC):
    """
    Base class for all Entities in DEMA.

    Design guarantees:
    - Operates ONLY on GlobalState
    - Explicit capability contracts
    - In-place mutation
    - Deterministic by default
    - Backward-compatible batching
    """

    # --------------------------------------------------
    # Capability declarations (override in subclasses)
    # --------------------------------------------------
    requires: List[str] = []
    produces: List[str] = []
    invariants: List[str] = []

    # --------------------------------------------------
    # Execution metadata (optional, profiler-filled)
    # --------------------------------------------------
    param_count: int = 0
    supports_cpu: bool = True
    supports_gpu: bool = False
    deterministic: bool = True
    avg_latency_ms: float | None = None

    name: str = "EntityBase"

    # --------------------------------------------------
    # Init
    # --------------------------------------------------
    def __init__(self):
        self._validate_contract()

    # --------------------------------------------------
    # Static contract validation
    # --------------------------------------------------
    def _validate_contract(self):
        overlap = set(self.produces) & set(self.invariants)
        if overlap:
            raise ValueError(
                f"{self.name}: produces and invariants overlap: {overlap}"
            )

    # --------------------------------------------------
    # Applicability check (router-safe)
    # --------------------------------------------------
    def is_applicable(self, state: GlobalState) -> bool:
        for path in self.requires:
            if self._get_field(state, path) is None:
                return False
        return True

    # --------------------------------------------------
    # Single-state execution (unchanged)
    # --------------------------------------------------
    def run(self, state: GlobalState) -> GlobalState:
        """
        Execute entity on a single GlobalState with invariant enforcement.
        """
        self._snapshot_and_run([state])
        return state

    # --------------------------------------------------
    # Batched execution (NEW, SAFE)
    # --------------------------------------------------
    def run_batch(self, states: Iterable[GlobalState]) -> Iterable[GlobalState]:
        """
        Execute entity on a batch of GlobalState objects.

        Default behavior:
        - Calls forward_batch if implemented
        - Otherwise falls back to per-state forward
        """
        states = list(states)
        self._snapshot_and_run(states)
        return states

    # --------------------------------------------------
    # Core execution with invariant enforcement
    # --------------------------------------------------
    def _snapshot_and_run(self, states: List[GlobalState]) -> None:
        # Snapshot invariants
        invariant_snapshots = []
        for state in states:
            invariant_snapshots.append({
                path: self._get_field(state, path)
                for path in self.invariants
            })

        # Execute
        if hasattr(self, "forward_batch"):
            self.forward_batch(states)
        else:
            for state in states:
                self.forward(state)

        # Enforce invariants
        for state, snapshot in zip(states, invariant_snapshots):
            for path, old_value in snapshot.items():
                new_value = self._get_field(state, path)
                if new_value != old_value:
                    raise RuntimeError(
                        f"{self.name} violated invariant '{path}'"
                    )

            # Trace logging
            state.trace.entities_used.append(self.name)
            state.control.step += 1

    # --------------------------------------------------
    # Abstract execution (required)
    # --------------------------------------------------
    @abstractmethod
    def forward(self, state: GlobalState) -> None:
        """
        Mutate a single GlobalState in-place.
        """
        pass

    # --------------------------------------------------
    # Optional batched execution (override when needed)
    # --------------------------------------------------
    def forward_batch(self, states: List[GlobalState]) -> None:
        """
        Optional batched execution.
        Override in neural-heavy entities.

        Default behavior = safe fallback.
        """
        for state in states:
            self.forward(state)

    # --------------------------------------------------
    # Utility: dotted-path access
    # --------------------------------------------------
    def _get_field(self, state: GlobalState, path: str):
        obj = state
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj