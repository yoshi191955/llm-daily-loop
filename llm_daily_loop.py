"""
llm_daily_loop.py — LLM日足ループ（クラウド版・GitHub Actions）

設計:
  * 米国市場の【寄り付き前】(08:30 ET) に実行 → 後知恵バイアス防止
  * 52銘柄すべてにカタリスト駆動の予測を生成（Claude API）
      - lean  : 全銘柄に必ず付ける方向の傾き（クロスセクション検定用）
      - score : 確信度 0-1
      - direction: 実際に賭けるか（up/down/flat）＝選別。閾値未満は flat
  * 1予測 → 2行（horizon=1d / horizon=3d）を出力（別行で記録）
  * 満期の来た行を採点:
      1d … その日の 寄り→引け
      3d … その日の 寄り → 3営業日目の引け
  * コスト: 往復0.5% + 利益のみ25%課税

秘密情報: ANTHROPIC_API_KEY のみ（環境変数。コードに書かない）
"""
from __future__ import annotations

import csv, json, os, sys, datetime as dt
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "docs", "predictions_log.csv")
WATCH = os.path.join(HERE, "watchlist.csv")

ET = ZoneInfo("America/New_York")
JST = ZoneInfo("Asia/Tokyo")

ROUND_TRIP_COST = 0.5      # 往復%
TAX = 0.25                 # 利益のみ課税
SCORE_THRESHOLD = 0.60     # これ未満は flat（見送り）。選別を維持する
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

COLS = ["pred_id","predict_date_jst","session_date","market","ticker","corr_bucket",
        "strategy_tag","lean","score","direction","entry_ref","target_pct","stop_pct",
        "horizon","confidence","catalyst_type","regime","thesis","status",
        "actual_open","actual_close","return_pct","net_return_pct","hit",
        "exit_reason","notes"]


def cost_net(r: float) -> float:
    a = r - ROUND_TRIP_COST
    return a * (1 - TAX) if a > 0 else a


def load_watchlist():
    with open(WATCH, encoding="utf-8-sig") as f:
        return [(r["ticker"], r.get("corr_bucket", "")) for r in csv.DictReader(f)]


def read_log():
    if not os.path.exists(LOG):
        return []
    with open(LOG, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in COLS})


# ─────────────────────────── データ ───────────────────────────

def fetch_daily(tickers):
    """日足（直近60日）。index は ET の日付。"""
    data = yf.download(tickers, period="60d", interval="1d", progress=False,
                       auto_adjust=False, group_by="ticker", threads=True)
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(subset=["Open", "Close"]).copy()
            df.index = pd.to_datetime(df.index).date
            if len(df) >= 5:
                out[t] = df
        except Exception:
            continue
    return out


def fetch_news(ticker, k=4):
    """直近ニュース見出し（無料・キー不要）。取れなければ空。"""
    try:
        items = yf.Ticker(ticker).news or []
        titles = []
        for it in items[:k]:
            c = it.get("content") or it
            ttl = c.get("title") or it.get("title")
            if ttl:
                titles.append(str(ttl)[:160])
        return titles
    except Exception:
        return []


def build_context(tickers_buckets, daily):
    ctx = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        news_map = dict(zip([t for t, _ in tickers_buckets],
                            ex.map(lambda tb: fetch_news(tb[0]), tickers_buckets)))
    for t, bucket in tickers_buckets:
        df = daily.get(t)
        if df is None or len(df) < 21:
            continue
        c = df["Close"]
        ctx[t] = {
            "bucket": bucket,
            "chg_1d": round((c.iloc[-1] / c.iloc[-2] - 1) * 100, 2),
            "chg_5d": round((c.iloc[-1] / c.iloc[-6] - 1) * 100, 2),
            "chg_20d": round((c.iloc[-1] / c.iloc[-21] - 1) * 100, 2),
            "last_close": round(float(c.iloc[-1]), 2),
            "news": news_map.get(t, []),
        }
    return ctx


# ─────────────────────────── 予測（Claude API） ───────────────────────────

SYSTEM = """あなたは米国株のデイトレード予測を行う定量アナリストです。

【最重要の前提】
- 価格だけから作るテクニカル指標（移動平均・RSI等）には予測力が無いことが、
  12,492トレードの検証で確認されています（方向的中率50.4%＝コイン投げ）。
  よってテクニカルをシグナルにしないでください。
- あなたの比較優位は「カタリスト（材料）の解釈」です。ニュース・イベント・
  セクター波及・地合いの因果を読んでください。
- 往復0.5%のコストと利益への25%課税があります。値幅が小さい予測は必ず損になります。
  「動く理由がある」銘柄だけを賭けの対象にしてください。

【出力ルール】
各銘柄について必ず次を出力:
- lean: "up" か "down"（必ずどちらか。賭けるかどうかとは無関係な"傾き"）
- score: 0.00〜1.00 の確信度
- direction: 実際に賭けるなら "up"/"down"、見送るなら "flat"
  ※ 明確なカタリストが無い銘柄は迷わず "flat" にしてください。
     見送りは失点ではありません。大半が "flat" で構いません。
- catalyst_type: 材料の種類（例: earnings, oil_supply_shock, guidance_cut, none）
- thesis: 根拠を1〜2文（日本語）

必ず JSON配列のみを出力し、説明文は付けないでください。"""


def predict(ctx, session_date, regime_hint=""):
    from anthropic import Anthropic
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY が未設定です（GitHub Secrets に登録してください）")
    client = Anthropic(api_key=key)

    lines = []
    for t, d in ctx.items():
        news = " | ".join(d["news"]) if d["news"] else "(ニュース無し)"
        lines.append(f'{t} [{d["bucket"]}] 終値{d["last_close"]} '
                     f'1d{d["chg_1d"]:+.1f}% 5d{d["chg_5d"]:+.1f}% 20d{d["chg_20d"]:+.1f}% :: {news}')
    user = (f"対象セッション: {session_date}（米国市場・寄り付き前）\n"
            f"{regime_hint}\n\n以下の{len(ctx)}銘柄すべてについて予測してください。\n\n"
            + "\n".join(lines) +
            '\n\n出力形式（JSON配列のみ）:\n'
            '[{"ticker":"NVDA","lean":"up","score":0.72,"direction":"up",'
            '"catalyst_type":"earnings","thesis":"..."}]')

    msg = client.messages.create(model=MODEL, max_tokens=8000,
                                 system=SYSTEM,
                                 messages=[{"role": "user", "content": user}])
    txt = msg.content[0].text.strip()
    if txt.startswith("```"):
        txt = txt.split("```")[1]
        txt = txt[4:] if txt.startswith("json") else txt
    try:
        return json.loads(txt)
    except json.JSONDecodeError as e:
        print(f"[warn] JSON parse失敗: {e}\n{txt[:400]}")
        return []


# ─────────────────────────── 採点 ───────────────────────────

def fetch_intraday(tickers):
    """日中足(15分・直近60日)を ticker->DataFrame(indexはET) で返す。
    取得不能な銘柄・失敗時は空にし、採点側で日足引けにフォールバックする（捏造しない）。"""
    try:
        data = yf.download(tickers, period="60d", interval="15m", progress=False,
                           auto_adjust=False, group_by="ticker", threads=True, prepost=False)
    except Exception:
        return {}
    out = {}
    for t in tickers:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            df = df.dropna(subset=["Open", "High", "Low", "Close"]).copy()
            if len(df) == 0:
                continue
            idx = df.index
            if idx.tz is None:
                idx = idx.tz_localize("UTC")
            df.index = idx.tz_convert(ET)
            out[t] = df
        except Exception:
            continue
    return out


def managed_exit_1d(intr, session_date, direction, entry, target_pct, stop_pct):
    """1d建玉を当日の日中足で管理決済（stop/target 先着、無ければ引け）。
    戻り値 (ret_pct, exit_price, reason)。ret_pct は方向調整済み（プラス=順行）。
    日中足が無ければ None（採点側で日足引けにフォールバック）。
    同一足で stop/target 両接触は、保守的に stop 優先。"""
    if intr is None:
        return None
    try:
        day = intr[[d.date() == session_date for d in intr.index]]
        day = day.between_time("09:30", "16:00")
    except Exception:
        return None
    if len(day) == 0:
        return None
    tp = abs(target_pct) / 100.0       # 例 +1.2 -> 0.012
    sp = -abs(stop_pct) / 100.0        # 例 -0.8 -> -0.008（必ず負）
    if direction == "up":
        target = entry * (1 + tp)      # 上の利確
        stop = entry * (1 + sp)        # 下の損切り
        for _, b in day.iterrows():
            if float(b["Low"]) <= stop:
                return (sp * 100.0, stop, "stop")
            if float(b["High"]) >= target:
                return (tp * 100.0, target, "target")
    else:                              # down（ショート）
        target = entry * (1 - tp)      # 下の利確
        stop = entry * (1 - sp)        # 上の損切り（sp<0 なので entry より上）
        for _, b in day.iterrows():
            if float(b["High"]) >= stop:
                return (sp * 100.0, stop, "stop")
            if float(b["Low"]) <= target:
                return (tp * 100.0, target, "target")
    # どちらも触れず → 引け
    close = float(day["Close"].iloc[-1])
    raw = (close - entry) / entry * 100.0
    ret = raw if direction == "up" else -raw
    return (ret, close, "引け")


def score_rows(rows, daily, intraday):
    """満期の来た open 行を採点。1dは日中足で stop/target を執行（無ければ日足引けにフォールバック）、
    3dは日足の3営業日目引け（従来通り。※新規3dは flat のため実際には対象外）。
    データが無ければ open のまま（捏造しない）。"""
    scored = 0
    for r in rows:
        if r["status"] != "open" or r["direction"] not in ("up", "down"):
            continue
        df = daily.get(r["ticker"])
        if df is None:
            continue
        try:
            sd = dt.date.fromisoformat(r["session_date"])
        except ValueError:
            continue
        idx = [d for d in df.index if d >= sd]
        if not idx or idx[0] != sd:
            continue                      # その日の足がまだ無い
        i = list(df.index).index(sd)
        entry = float(df["Open"].iloc[i])

        if r["horizon"] == "1d":
            try:
                tp = float(str(r.get("target_pct", "")).replace("+", "") or 1.2)
            except (ValueError, TypeError):
                tp = 1.2
            try:
                sp = float(str(r.get("stop_pct", "")).replace("+", "") or -0.8)
            except (ValueError, TypeError):
                sp = -0.8
            res = None
            try:
                res = managed_exit_1d(intraday.get(r["ticker"]), sd,
                                      r["direction"], entry, tp, sp)
            except Exception:
                res = None
            if res is not None:
                ret, exitp, reason = res
            else:                         # フォールバック: 日足の当日引け
                exitp = float(df["Close"].iloc[i])
                raw = (exitp - entry) / entry * 100.0
                ret = raw if r["direction"] == "up" else -raw
                reason = "引け(日足)"
        else:                             # 3d = 3営業日目の引け（従来通り）
            if i + 2 >= len(df):
                continue                  # まだ満期が来ていない
            exitp = float(df["Close"].iloc[i + 2])
            raw = (exitp - entry) / entry * 100.0
            ret = raw if r["direction"] == "up" else -raw
            reason = "3営業日目の引け"

        net = cost_net(ret)
        r["actual_open"] = f"{entry:.2f}"
        r["actual_close"] = f"{exitp:.2f}"
        r["return_pct"] = f"{ret:+.2f}"
        r["net_return_pct"] = f"{net:+.2f}"
        r["hit"] = "TRUE" if ret > 0 else "FALSE"
        r["status"] = "closed"
        r["exit_reason"] = reason + ("(コスト負け)" if (ret > 0 and net < 0) else "")
        scored += 1
    return scored


# ─────────────────────────── メイン ───────────────────────────

def smoke_test():
    """APIキー・モデル名・JSONパースだけを検証する。ログには一切書き込まない。"""
    wl = load_watchlist()[:3]
    tickers = [t for t, _ in wl]
    print(f"[smoke] 対象: {tickers}（書き込みは行いません）")
    daily = fetch_daily(tickers)
    print(f"[smoke] 日足取得: {len(daily)}/{len(tickers)} 銘柄")
    ctx = build_context(wl, daily)
    if not ctx:
        print("[smoke] NG: コンテキストを構築できませんでした")
        sys.exit(1)
    preds = predict(ctx, dt.date.today().isoformat())
    if not preds:
        print("[smoke] NG: 予測が空です（JSONパース失敗の可能性）")
        sys.exit(1)
    print(f"[smoke] OK: Claude API から {len(preds)} 件の予測を取得")
    for p in preds:
        print(f"   {p.get('ticker'):<6} lean={p.get('lean'):<4} score={p.get('score')} "
              f"direction={p.get('direction'):<4} catalyst={p.get('catalyst_type')}")
        print(f"          thesis: {str(p.get('thesis',''))[:90]}")
    print("[smoke] ✓ APIキー・モデル・JSON出力すべて正常。ログは未変更。")


def main():
    if "--smoke" in sys.argv:
        smoke_test()
        return
    now = dt.datetime.now(dt.timezone.utc)
    et = now.astimezone(ET)
    session = et.date().isoformat()

    wl = load_watchlist()
    tickers = [t for t, _ in wl]
    daily = fetch_daily(tickers)
    intraday = fetch_intraday(tickers)
    print(f"[llm_loop] 日足取得 {len(daily)}/{len(tickers)} 銘柄 / "
          f"日中足取得 {len(intraday)}/{len(tickers)} 銘柄")

    rows = read_log()
    seen = {r["pred_id"] for r in rows}

    # ── 採点（先に実行）──
    n_scored = score_rows(rows, daily, intraday)

    # ── 予測（平日 かつ 寄り付き前 のみ）──
    n_new = 0
    weekday = et.weekday() < 5
    before_open = et.time() < dt.time(9, 30)
    # 後知恵バイアス防止: そのセッションの足が既に存在するなら予測しない
    already = any(session in [d.isoformat() for d in df.index] for df in daily.values())

    if weekday and before_open and not already:
        ctx = build_context(wl, daily)
        preds = predict(ctx, session)
        bucket_of = dict(wl)
        for p in preds:
            t = p.get("ticker")
            if t not in ctx:
                continue
            lean = p.get("lean", "up")
            score = float(p.get("score", 0) or 0)
            direction = p.get("direction", "flat")
            if direction in ("up", "down") and score < SCORE_THRESHOLD:
                direction = "flat"          # 閾値未満は見送り（選別を維持）
            conf = 3 if score >= 0.75 else (2 if score >= SCORE_THRESHOLD else 1)
            for h in ("1d", "3d"):
                pid = f"{session}-{t}-{h}"
                if pid in seen:
                    continue
                # 3d は「賭け」から外す（測定用に lean/score は残すが direction=flat）
                row_dir = direction if h == "1d" else "flat"
                rows.append({
                    "pred_id": pid,
                    "predict_date_jst": now.astimezone(JST).strftime("%Y-%m-%d %H:%M"),
                    "session_date": session, "market": "US", "ticker": t,
                    "corr_bucket": bucket_of.get(t, ""), "strategy_tag": "llm_catalyst",
                    "lean": lean, "score": f"{score:.2f}", "direction": row_dir,
                    "entry_ref": "session_open", "target_pct": "+1.2", "stop_pct": "-0.8",
                    "horizon": h, "confidence": conf,
                    "catalyst_type": p.get("catalyst_type", ""), "regime": "",
                    "thesis": str(p.get("thesis", ""))[:300], "status": "open",
                    "actual_open": "", "actual_close": "", "return_pct": "",
                    "net_return_pct": "", "hit": "", "exit_reason": "",
                    "notes": f"model={MODEL}",
                })
                seen.add(pid)
                n_new += 1

    write_log(rows)
    closed = sum(1 for r in rows if r["status"] == "closed")
    print(f"[llm_loop] session={session} ET={et:%H:%M} weekday={weekday} "
          f"before_open={before_open} already_open={already}")
    print(f"[llm_loop] 新規予測 {n_new} 行 / 今回採点 {n_scored} 行 / "
          f"累計 {len(rows)} 行 (closed={closed})")


if __name__ == "__main__":
    main()
