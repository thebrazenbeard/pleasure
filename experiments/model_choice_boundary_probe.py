"""Non-explicit real-model appraisal-to-choice boundary probe (source-only)."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Case:
    name: str
    a_score: float
    b_score: float
    a_cost: float
    b_cost: float
    activity_a: str
    activity_b: str

CASES = (
    Case("ON", .85, .25, 0, 0, "catalog a museum archive", "walk a garden"),
    Case("INVERTED", .25, .85, 0, 0, "catalog a museum archive", "walk a garden"),
    Case("LABEL_SWAP", .25, .85, 0, 0, "walk a garden", "catalog a museum archive"),
    Case("COST", .85, .60, .65, .05, "catalog a museum archive", "walk a garden"),
    Case("ABSTAIN", .30, .20, 0, 0, "catalog a museum archive", "walk a garden"),
)

def render_prompt(case: Case) -> str:
    raise NotImplementedError("render_prompt")

def extract_generated_choice(raw: str) -> str | None:
    raise NotImplementedError("extract_generated_choice")

def ideal_choice(case: Case) -> str:
    raise NotImplementedError("ideal_choice")
