from typing import TypedDict, List, Optional

class SupportState(TypedDict, total=False):
    query: str
    intent: str
    retrieved_chunks: List[str]
    confidence: float
    answer: str
    escalated: bool
    human_response: Optional[str]
    citations: List[str]
