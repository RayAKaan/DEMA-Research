from controller.policies.base import ExecutionPolicy, ExecutionDecision
from state import GlobalState
from entities.base import EntityBase


class RuleBasedPolicy(ExecutionPolicy):
    """
    Deterministic, interpretable execution policy.

    Strategy:
    - Select the first applicable entity
    - Choose device based on entity metadata
    """

    def decide(self, state: GlobalState, entities):
        for entity in entities:
            if entity.is_applicable(state):
                device = self._choose_device(entity)
                return ExecutionDecision(
                    entity=entity,
                    device=device,
                )
        return None

    # -------------------------------------------------
    # Device selection heuristic
    # -------------------------------------------------
    def _choose_device(self, entity: EntityBase) -> str:
        """
        Simple, safe heuristic.
        """
        if entity.supports_gpu and entity.param_count > 50_000:
            return "cuda"
        return "cpu"