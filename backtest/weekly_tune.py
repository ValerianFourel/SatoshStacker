"""Weekly self-tuner — the meta-layer above the 4h tactical trader.

Once per week it (1) backtests which technicals best predict BTC returns across
timeframes (indicators.py), then (2) asks the BEST Qwen model to choose the LEADING
timeframe + primary indicator + parameters (RSI period, forward lag, momentum-vs-
reversion regime) + confirming indicators for the coming week, and writes a CONTEXT
note that the downstream 4h trader injects into its prompt. Output: agent/technicals.json.

Run weekly (cron / systemd timer — see DEPLOY.md). One smart-Qwen call per week.

    python3 backtest/weekly_tune.py --days 60 --tfs 1h,4h
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backtest.indicators import OUT, load_tf, rank_tf  # noqa: E402

META_SYSTEM = (
    "You are a quant research lead tuning a BTC trader. You are given, per candle "
    "timeframe, technical indicators ranked by Information Coefficient (IC = predictive "
    "correlation with forward returns; POSITIVE = momentum/trend persists, NEGATIVE = "
    "mean-reversion / extremes revert). Choose, for the COMING WEEK: the single leading "
    "timeframe, the primary indicator + its parameters, the forward lag (horizon in bars), "
    "whether the current regime is MOMENTUM or REVERSION, the RSI period to use, and 1-2 "
    "confirming indicators. Then write a concise CONTEXT note (<=80 words) telling the "
    "downstream 4h tactical trader how to read these signals this week (e.g. 'regime is "
    "momentum: favor BTC when RSI28 is high & rising; only de-risk on trend breaks'). "
    "Respond STRICT JSON only:\n"
    '{"timeframe":"4h","primary_indicator":"rsi_28","rsi_period":28,"lag":6,'
    '"regime":"momentum","confirming":["ema_cross_20_50"],"context_note":"...",'
    '"rationale":"..."}'
)


def smart_decision(payload: dict) -> dict:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=True)
    from agent.secrets import clean_secret
    key = clean_secret(os.getenv("LLM_API_KEY"))
    model = os.getenv("WEEKLY_TUNE_MODEL", "qwen/qwen3.6-plus")  # best-of-best, weekly
    if not key:
        return {"error": "no LLM key"}
    try:
        from openai import OpenAI
        c = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=key)
        resp = c.chat.completions.create(
            model=model, temperature=0.2, timeout=90, max_tokens=700,
            messages=[{"role": "system", "content": META_SYSTEM},
                      {"role": "user", "content": json.dumps(payload)}])
        txt = resp.choices[0].message.content or ""
        i, j = txt.find("{"), txt.rfind("}")
        return json.loads(txt[i:j + 1]) if 0 <= i < j else {"error": "unparseable"}
    except Exception as e:  # noqa: BLE001
        return {"error": type(e).__name__}


def main() -> None:
    ap = argparse.ArgumentParser("weekly_tune")
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--tfs", default="1h,4h")
    ap.add_argument("--lags", default="1,2,3,6")
    args = ap.parse_args()
    lags = [int(x) for x in args.lags.split(",")]
    tfs = args.tfs.split(",")

    print(f"\n=== WEEKLY SELF-TUNE — backtest {args.days}d, timeframes {tfs} ===")
    ranked = {tf: rank_tf(load_tf(tf, args.days), lags) for tf in tfs}
    payload = {tf: [{k: r[k] for k in ("indicator", "lag", "ic", "direction")}
                    for r in ranked[tf][:6]] for tf in tfs}
    for tf in tfs:
        print(f"  [{tf}] top: " + ", ".join(
            f"{r['indicator']}(IC{r['ic']:+.3f})" for r in ranked[tf][:3]))

    print("  asking best-of-best Qwen to choose this week's leading technicals…")
    decision = smart_decision(payload)

    leader = sorted([{**ranked[tf][0], "timeframe": tf} for tf in tfs],
                    key=lambda r: -r["abs_ic"])[0]
    # Qwen decision drives config; fall back to the raw backtest leader if it failed
    sug = {
        "timeframe": decision.get("timeframe", leader["timeframe"]),
        "primary_indicator": decision.get("primary_indicator", leader["indicator"]),
        "rsi_period": int(decision.get("rsi_period", 14)),
        "lag": int(decision.get("lag", leader["lag"])),
        "regime": decision.get("regime", leader["direction"]),
        "confirming": decision.get("confirming", []),
    }
    cfg = {
        "as_of": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lookback_days": args.days,
        "backtest_leader": leader,
        "ranked": ranked,
        "qwen_decision": decision,
        "context_note": decision.get("context_note", ""),
        "suggested": sug,
    }
    OUT.write_text(json.dumps(cfg, indent=2))

    if decision.get("error"):
        print(f"  (Qwen meta-call failed: {decision['error']} — used raw backtest leader)")
    else:
        print(f"\n  QWEN CHOSE: {sug['primary_indicator']} on {sug['timeframe']} "
              f"(rsi_period={sug['rsi_period']}, lag={sug['lag']}, regime={sug['regime']})")
        print(f"  confirming: {sug['confirming']}")
        print(f"  context for the trader: {cfg['context_note']}")
    print(f"\n  -> wrote {OUT.relative_to(ROOT)} (the 4h trader reads `suggested` + `context_note`)")


if __name__ == "__main__":
    main()
