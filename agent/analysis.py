"""Qwen3.6-Plus advisory — veto/shrink ONLY, fail-safe.

Hard asymmetry (the safety story): the advisor may only make accumulation MORE
cautious. It returns a `size_multiplier` in [min_fraction, 1.0]; the gate clamps
it so values > 1 are ignored and the advisor can never increase a buy, buy above
the ladder, deploy faster, sell, or block the deploy-by-deadline guard. It is
read-only with respect to orders.

Fail-safe: a missing / malformed / STALE verdict (``as_of_bar`` older than the
current bar) is treated as ``proceed`` at the *spine's* size (multiplier 1.0) —
a broken advisor can never change deployment in either direction beyond the
deterministic ladder.

Inputs are computed NUMBERS only (never chart screenshots). The OpenAI-compatible
client (OpenRouter/DashScope) is imported lazily, so dry-run and tests need no SDK.
"""
from __future__ import annotations

import abc
import json
from dataclasses import dataclass, field

from .config import AnalysisConfig
from .ladder import clamp_multiplier

_SYSTEM = (
    "You are a cautious risk advisor for a deterministic BTC accumulation ladder. "
    "You may ONLY make buying more cautious. You CANNOT increase a buy, buy above "
    "the ladder, deploy faster, or sell. Given computed numbers, decide whether the "
    "next tranche should proceed at full size or be shrunk/deferred to a lower level. "
    "Respond with STRICT JSON only:\n"
    '{"stance":"proceed|shrink|defer","size_multiplier":<float 0..1>,'
    '"note":"<short>","sources":[],"as_of_bar":"<ISO8601 of the input bar>"}'
)


@dataclass
class Verdict:
    stance: str = "proceed"
    size_multiplier: float = 1.0
    note: str = ""
    sources: list[str] = field(default_factory=list)
    as_of_bar: str = ""

    @staticmethod
    def safe(as_of_bar: str, note: str = "fail-safe: spine size") -> "Verdict":
        return Verdict("proceed", 1.0, note, [], as_of_bar)


class Advisor(abc.ABC):
    @abc.abstractmethod
    def advise(self, features: dict, *, as_of_bar: str) -> Verdict: ...


class MockAdvisor(Advisor):
    """Deterministic advisor for dry-run/tests. Defaults to proceed; can be told
    to shrink so the gate path is exercised without a network call."""

    def __init__(self, stance: str = "proceed", multiplier: float = 1.0,
                 note: str = "mock") -> None:
        self._stance, self._mult, self._note = stance, multiplier, note

    def advise(self, features: dict, *, as_of_bar: str) -> Verdict:
        return Verdict(self._stance, self._mult, self._note, [], as_of_bar)


class QwenAdvisor(Advisor):
    """Calls Qwen3.6-Plus via an OpenAI-compatible endpoint. Fail-safe on any error."""

    def __init__(self, cfg: AnalysisConfig, api_key: str) -> None:
        self.cfg = cfg
        self._api_key = api_key

    def advise(self, features: dict, *, as_of_bar: str) -> Verdict:
        if not self._api_key:
            return Verdict.safe(as_of_bar, "no LLM key: spine size")
        try:
            from openai import OpenAI  # lazy import
            client = OpenAI(base_url=self.cfg.base_url, api_key=self._api_key)
            user = json.dumps({**features, "as_of_bar": as_of_bar})
            resp = client.chat.completions.create(
                model=self.cfg.model,
                messages=[{"role": "system", "content": _SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0,
                timeout=self.cfg.request_timeout_s,
                max_tokens=self.cfg.max_tokens,
                response_format={"type": "json_object"},
            )
            return _parse(resp.choices[0].message.content or "", as_of_bar)
        except Exception:  # noqa: BLE001 - never let a broken advisor change sizing
            return Verdict.safe(as_of_bar, "LLM error: spine size")


def _parse(raw: str, as_of_bar: str) -> Verdict:
    try:
        d = json.loads(raw)
        return Verdict(
            stance=str(d.get("stance", "proceed")),
            size_multiplier=float(d.get("size_multiplier", 1.0)),
            note=str(d.get("note", ""))[:200],
            sources=[str(s) for s in d.get("sources", [])][:5],
            as_of_bar=str(d.get("as_of_bar", as_of_bar)),
        )
    except Exception:  # noqa: BLE001
        return Verdict.safe(as_of_bar, "malformed verdict: spine size")


def gate_multiplier(verdict: Verdict | None, *, current_bar: str,
                    min_fraction: float) -> float:
    """Convert a verdict into the deterministic size multiplier the gate applies.

    Fail-safe rules:
      * ``None`` / missing            -> 1.0 (spine size)
      * ``as_of_bar`` older than now  -> 1.0 (stale advisor cannot act)
      * otherwise                     -> clamp to [min_fraction, 1.0]; >1 ignored
    """
    if verdict is None:
        return 1.0
    # staleness: lexicographic ISO8601 compare is correct for same-format UTC stamps
    if verdict.as_of_bar and current_bar and verdict.as_of_bar < current_bar:
        return 1.0
    return clamp_multiplier(verdict.size_multiplier, min_fraction)


def build_features(*, price: float, anchor: float, gen_floor: float,
                   floor_price: float, deployed: float, budget: float,
                   days_left: float, resting_levels: list[float]) -> dict:
    """Computed numbers fed to the advisor (NEVER chart images)."""
    return {
        "price": round(price, 2),
        "anchor": round(anchor, 2),
        "generation_floor": round(gen_floor, 2),
        "target_floor": round(floor_price, 2),
        "pct_to_target_floor": round(100 * (price - floor_price) / max(price, 1), 2),
        "usdc_deployed": round(deployed, 2),
        "usdc_budget": round(budget, 2),
        "pct_deployed": round(100 * deployed / max(budget, 1), 1),
        "days_left_in_window": round(days_left, 1),
        "resting_levels": [round(x, 2) for x in resting_levels],
    }
