from __future__ import annotations

import asyncio

from filingdelta.schemas.chat import (
    ChatConversationMessage,
    ChatSessionState,
    ConversationSummary,
)


_RECENT_MESSAGE_LIMIT = 8


class ChatMemoryStore:
    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str], ChatSessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, *, document_id: str, session_id: str) -> ChatSessionState:
        key = (document_id, session_id)
        async with self._lock:
            state = self._sessions.get(key)
            if state is None:
                state = ChatSessionState(
                    session_id=session_id,
                    document_id=document_id,
                    conversation_summary=ConversationSummary(),
                )
                self._sessions[key] = state
            return state.model_copy(deep=True)

    async def replace_summary(
        self,
        *,
        document_id: str,
        session_id: str,
        summary: ConversationSummary,
    ) -> ChatSessionState:
        key = (document_id, session_id)
        async with self._lock:
            state = self._sessions.setdefault(
                key,
                ChatSessionState(
                    session_id=session_id,
                    document_id=document_id,
                    conversation_summary=ConversationSummary(),
                ),
            )
            state.conversation_summary = summary
            self._sessions[key] = state
            return state.model_copy(deep=True)

    async def append_turn(
        self,
        *,
        document_id: str,
        session_id: str,
        user_message: str,
        assistant_message: str,
    ) -> ChatSessionState:
        key = (document_id, session_id)
        async with self._lock:
            state = self._sessions.setdefault(
                key,
                ChatSessionState(
                    session_id=session_id,
                    document_id=document_id,
                    conversation_summary=ConversationSummary(),
                ),
            )
            state.recent_messages.extend(
                [
                    ChatConversationMessage(role="user", content=user_message),
                    ChatConversationMessage(role="assistant", content=assistant_message),
                ]
            )
            state.recent_messages = state.recent_messages[-_RECENT_MESSAGE_LIMIT:]
            self._sessions[key] = state
            return state.model_copy(deep=True)
