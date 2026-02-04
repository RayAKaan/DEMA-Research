from collections import defaultdict
import math


class AttentionLens:
    """
    Observer-only attention lens for arithmetic execution.

    PURPOSE:
    - Reveal which entities influenced the result
    - Measure interaction frequency and temporal locality
    - Expose structural relationships WITHOUT control

    GUARANTEES:
    - Read-only
    - No mutation of state
    - No routing
    - No execution decisions
    """

    # --------------------------------------------------
    # Build attention map
    # --------------------------------------------------
    @staticmethod
    def build(state):
        """
        Builds a symbolic attention map from execution trace.
        """

        attention = defaultdict(float)
        timeline = state.trace.decisions

        for t, decision in enumerate(timeline):
            entity = AttentionLens._extract_entity(decision)
            if entity is None:
                continue

            # Temporal decay (recent steps matter more)
            weight = AttentionLens._time_weight(t, len(timeline))
            attention[entity] += weight

        return dict(attention)

    # --------------------------------------------------
    # Normalize attention
    # --------------------------------------------------
    @staticmethod
    def normalize(attention_map):
        total = sum(attention_map.values())
        if total == 0:
            return attention_map

        return {
            k: v / total
            for k, v in attention_map.items()
        }

    # --------------------------------------------------
    # Pretty print
    # --------------------------------------------------
    @staticmethod
    def report(state, top_k=None):
        """
        Prints attention distribution.
        """

        raw = AttentionLens.build(state)
        attn = AttentionLens.normalize(raw)

        print("\n=== Attention Lens (Observer) ===")

        items = sorted(
            attn.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if top_k:
            items = items[:top_k]

        for entity, score in items:
            print(f"{entity:<30} : {score:.3f}")

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    @staticmethod
    def _extract_entity(decision: str):
        """
        Extracts entity name from trace string.
        """
        if not isinstance(decision, str):
            return None

        for token in decision.split():
            if token.endswith("Entity") or token.endswith("Controller"):
                return token
        return None

    @staticmethod
    def _time_weight(step, total_steps):
        """
        Later steps get slightly higher weight.
        """
        return math.exp(step / max(1, total_steps))