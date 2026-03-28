from __future__ import annotations

from typing import Any, Dict

from pydantic import BaseModel, Field, model_validator


class RewardModel(BaseModel):
    """
    Typed reward model returned after each step() in CleanFlowEnv.

    Encapsulates the full reward signal: net reward, quality improvement,
    penalties, budget cost, and episode termination status.
    """

    reward: float = Field(
        ..., description="Net reward for this step: quality_delta - penalty - budget_cost."
    )
    quality_delta: float = Field(
        ..., description="Improvement over best_quality_so_far (floored at 0)."
    )
    penalty: float = Field(
        ..., description="Total penalties incurred this step (0 if valid action)."
    )
    budget_cost: float = Field(
        ..., description="Action credits consumed this step."
    )
    cumulative_quality: float = Field(
        ..., description="Overall quality of current_table vs ground_truth after this step."
    )
    done: bool = Field(
        ..., description="Whether the episode has ended."
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata (e.g. action applied, termination reason).",
    )

    @model_validator(mode="after")
    def check_reward_consistency(self) -> "RewardModel":
        expected = self.quality_delta - self.penalty - self.budget_cost
        if abs(self.reward - expected) > 1e-6:
            import logging

            logging.getLogger("cleanflow").warning(
                f"Reward inconsistency: reward={self.reward:.6f}, "
                f"expected={expected:.6f} (delta={self.quality_delta}, "
                f"penalty={self.penalty}, cost={self.budget_cost})"
            )
        return self

    @classmethod
    def from_step(
        cls,
        quality_delta: float,
        penalty: float,
        budget_cost: float,
        cumulative_quality: float,
        done: bool,
        info: Dict[str, Any] | None = None,
    ) -> "RewardModel":
        """Construct a RewardModel from step-level components."""
        return cls(
            reward=quality_delta - penalty - budget_cost,
            quality_delta=quality_delta,
            penalty=penalty,
            budget_cost=budget_cost,
            cumulative_quality=cumulative_quality,
            done=done,
            info=info or {},
        )
