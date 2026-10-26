"""Privacy-safe rolling conversational memory for the active session."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class ConversationMessage:
    role: str
    message: str
    timestamp: str


class RollingConversationBuffer:
    """Keep recent user/assistant messages within turn and approximate character limits."""

    def __init__(self, max_turns: int = 4, max_characters: int = 6000) -> None:
        if max_turns < 1 or max_characters < 1:
            raise ValueError("Memory limits must be positive")
        self.max_turns = max_turns
        self.max_characters = max_characters
        self._messages: deque[ConversationMessage] = deque()

    def add(self, role: str, message: str, timestamp: str | None = None) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        item = ConversationMessage(
            role=role,
            message=str(message).strip(),
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        )
        self._messages.append(item)
        self._truncate()

    def _truncate(self) -> None:
        max_messages = self.max_turns * 2
        while len(self._messages) > max_messages:
            self._messages.popleft()
        while self.character_count > self.max_characters and len(self._messages) > 1:
            self._messages.popleft()

    @property
    def character_count(self) -> int:
        return sum(len(item.message) for item in self._messages)

    def messages(self) -> list[dict[str, str]]:
        return [asdict(item) for item in self._messages]

    def as_prompt_text(self) -> str:
        return "\n".join(f"{item.role.upper()}: {item.message}" for item in self._messages)

    def clear(self) -> None:
        self._messages.clear()

