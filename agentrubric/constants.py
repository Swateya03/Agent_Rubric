"""
constants.py — Single source of truth for all tuneable values in AgentRubric.

All magic numbers, thresholds, and display lengths are defined here.
Import from this module instead of hardcoding values anywhere else.

To tune the system, change values here and the change propagates everywhere.
"""

# ── Hack detection ──────────────────────────────────────────────────────────
DEFAULT_HACK_THRESHOLD = 0.25
# Divergence above this triggers a hack flag and retry loop.
# Based on Gao et al. 2022 proxy/gold divergence research.

DEFAULT_MAX_ITERATIONS = 3
# Maximum number of hack-detection retry loops per sample.
# Prevents infinite loops when a rubric cannot be made robust.

# ── Divergence classification thresholds ────────────────────────────────────
DIVERGENCE_ALIGNED        = 0.10
# abs(proxy - gold) below this → "aligned". Rubric is reliable.

DIVERGENCE_MINOR_DRIFT    = 0.25
# abs(proxy - gold) below this → "minor_drift". Watch but do not retry.

DIVERGENCE_SIGNIFICANT    = 0.40
# abs(proxy - gold) below this → "significant_drift". Likely rubric weakness.
# Above this → "likely_hack". Triggers retry if iterations remain.

# ── Scoring ──────────────────────────────────────────────────────────────────
PASS_THRESHOLD = 0.6
# overall_score >= PASS_THRESHOLD → RubricResult.passed = True

# ── Display truncation lengths ───────────────────────────────────────────────
MAX_TASK_DISPLAY      = 60
# Characters shown for task text in terminal output and run_history messages.

MAX_RESPONSE_DISPLAY  = 300
# Characters shown for response text in LLM prompts (critic, gold judge).

MAX_RUBRIC_DISPLAY    = 150
# Characters shown for rubric text in smoke test / debug output.

MAX_REASONING_DISPLAY = 40
# Characters shown for criterion reasoning in print_result().

MAX_HISTORY_PREVIEW   = 120
# Characters shown for retry_context preview in smoke tests.

# ── Transcript quality ───────────────────────────────────────────────────────
MIN_RESPONSE_WORDS     = 20
# Responses shorter than this get a quality penalty of -0.5.

MIN_TASK_WORDS         = 5
# Tasks shorter than this get a quality penalty of -0.3.

MIN_REPETITION_NGRAM   = 5
# N-gram size for repetition detection.

REPETITION_COUNT_LIMIT = 3
# If an N-gram appears >= this many times, response is flagged as repetitive.

ALPHA_RATIO_MIN        = 0.5
# If ratio of alpha chars to total chars is below this, flag as non_text_response.

QUALITY_FLAG_THRESHOLD = 0.5
# transcript_quality_score below this → transcript_flagged = True
