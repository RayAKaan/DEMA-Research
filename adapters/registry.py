class AdapterRegistry:
    """
    Selects the most appropriate input adapter using
    capability detection + priority resolution.
    """

    def __init__(self, adapters):
        self.adapters = adapters

    def adapt(self, raw_input: str) -> dict:
        candidates = [
            adapter for adapter in self.adapters
            if adapter.supports(raw_input)
        ]

        if not candidates:
            raise ValueError(f"No input adapter supports: {raw_input}")

        # Choose the most specific adapter
        adapter = max(candidates, key=lambda a: a.priority)
        return adapter.adapt(raw_input)
