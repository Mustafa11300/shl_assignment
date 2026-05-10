"""
Pydantic request/response models for the SHL Assessment Recommender API.

These models match the exact schema specified in the assignment.
The schema is non-negotiable — deviating breaks the automated evaluator.
"""

from pydantic import BaseModel, Field
from typing import Literal


class Message(BaseModel):
    """A single message in the conversation history."""
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat.
    
    The API is stateless — every call carries the full conversation history.
    """
    messages: list[Message]


class Recommendation(BaseModel):
    """A single assessment recommendation with catalog reference."""
    name: str = Field(..., description="Assessment name from the SHL catalog")
    url: str = Field(..., description="Full URL to the product catalog page")
    test_type: str = Field(..., description="Letter codes: A, B, C, D, E, K, P, S (comma-separated for multi-type)")


class ChatResponse(BaseModel):
    """Response body for POST /chat.
    
    - `recommendations` is EMPTY [] when the agent is clarifying, comparing, or refusing.
    - `recommendations` has 1-10 items when the agent commits to a shortlist.
    - `end_of_conversation` is true only when the agent considers the task complete.
    """
    reply: str = Field(..., description="The agent's natural language response")
    recommendations: list[Recommendation] = Field(
        default_factory=list,
        description="Empty when gathering context or refusing. 1-10 items when recommending."
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only when the agent considers the task complete"
    )
