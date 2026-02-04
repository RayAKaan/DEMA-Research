from abc import ABC, abstractmethod


class InputAdapter(ABC):
    """
    Base class for all input adapters.

    Adapters deterministically convert raw input strings
    into a canonical global state.
    """

    # Higher priority = more specific adapter
    priority: int = 0

    @abstractmethod
    def supports(self, raw_input: str) -> bool:
        """
        Returns True if this adapter can handle the input.
        Must be cheap and deterministic.
        """
        pass

    @abstractmethod
    def adapt(self, raw_input: str) -> dict:
        """
        Converts raw input into a canonical global state.
        Must not perform reasoning or learning.
        """
        pass
