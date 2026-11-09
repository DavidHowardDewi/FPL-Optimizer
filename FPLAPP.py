"""
FPL Lineup Value Optimizer — Final submission (Checkpoint 5)
MGSC 300 · David Howard

Flow: user inputs -> live FPL API call -> guardrails -> transparent scoring
      -> linear-programming optimizer -> Top XI + KPIs + transparency box.
Endpoints:
  https://fantasy.premierleague.com/api/bootstrap-static/
  https://fantasy.premierleague.com/api/fixtures/?future=1
"""
import time
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

try:
    import pulp
    HAS_PULP = True
except Exception:  # optimizer falls back to greedy if PuLP is unavailable
    HAS_PULP = False

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/?future=1"
HEADERS = {"User-Agent": "Mozilla/5.0 (MGSC300 student project)"}
POS = {1: "GKP", 2: "DEF", 3: "MID", 4: "FWD"}
FORMATIONS = {
    "3-4-3": {1: 1, 2: 3, 3: 4, 4: 3},
    "3-5-2": {1: 1, 2: 3, 3: 5, 4: 2},
    "4-4-2": {1: 1, 2: 4, 3: 4, 4: 2},
    "4-3-3": {1: 1, 2: 4, 3: 3, 4: 3},
    "4-5-1": {1: 1, 2: 4, 3: 5, 4: 1},
    "5-3-2": {1: 1, 2: 5, 3: 3, 4: 2},
    "5-4-1": {1: 1, 2: 5, 3: 4, 4: 1},
}
HORIZON = 3  # gameweeks


# ---------- 1. API CALL (cached, with last-good fallback) ----------
@st.cache_data(ttl=600, show_spinner=False)
def _fetch_live():
    log, out = [], {}
    for name, url in [("bootstrap-static", BOOTSTRAP_URL), ("fixtures?future=1", FIXTURES_URL)]:
        t0 = time.time()
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        out[name] = r.json()
        log.append({"endpoint": name, "status": r.status_code,
                    "KB": round(len(r.content) / 1024, 1),
                    "ms": int((time.time() - t0) * 1000)})
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return out["bootstrap-static"], out["fixtures?future=1"], log, fetched_at


def fetch_fpl():
    """Live call; if it fails, reuse the last good response from this session."""
    try:
        data = _fetch_live()
        st.session_state["last_good"] = data
        return (*data, "live")
    except Exception as e:
        if "last_good" in st.session_state:
            return (*st.session_state["last_good"], f"fallback (live call failed: {e})")
        raise


# ---------- 2. TRANSFORM ----------
def build_pool(bootstrap, fixtures):
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    df["price"] = df["now_cost"] / 10.0
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0.0)
    df["pos"] = df["element_type"].map(POS)
    df["club"] = df["team"].map(teams)
    df["pts_per_m"] = df["total_points"] / df["price"]

    # Fixture ease = 6 - FDR, averaged over each team's next HORIZON fixtures
    fx = pd.DataFrame(fixtures)
    rows = []
    if not fx.empty:
        fx = fx[fx["event"].notna()]
        if "finished" in fx:
            fx = fx[~fx["finished"].astype(bool)]
        for _, f in fx.iterrows():
            rows.append({"team": f["team_h"], "event": f["event"], "ease": 6 - f["team_h_difficulty"]})
            rows.append({"team": f["team_a"], "event": f["event"], "ease": 6 - f["team_a_difficulty"]})
    ease = pd.DataFrame(rows)
    if ease.empty:
        df["fix_ease"] = 3.0
    else:
        ease = ease.sort_values("event").groupby("team").head(HORIZON)
        df["fix_ease"] = df["team"].map(ease.groupby("team")["ease"].mean()).fillna(3.0)
    return df


# ---------- 3. GUARDRAILS ----------
def apply_guardrails(df, min_minutes):
    reason = pd.Series("", index=df.index)
    reason[df["status"] != "a"] = "Not available (injured / doubtful / suspended)"
    reason[(reason == "") & (df["minutes"] < min_minutes)] = f"Under {min_minutes} minutes played"
    excluded = df[reason != ""].assign(reason=reason[reason != ""])
    return df[reason == ""].copy(), excluded


# ---------- 4. TRANSPARENT SCORE ----------
def _scale(s):
    lo, hi = s.min(), s.max()
    return (s - lo) / (hi - lo) if hi > lo else s * 0 + 0.5


def score(df, w_form, w_value, w_fix):
    total = max(w_form + w_value + w_fix, 1)
    df = df.copy()
    df["c_form"] = 100 * (w_form / total) * _scale(df["form"])
    df["c_value"] = 100 * (w_value / total) * _scale(df["pts_per_m"])
    df["c_fix"] = 100 * (w_fix / total) * _scale(df["fix_ease"])
    df["score"] = df["c_form"] + df["c_value"] + df["c_fix"]
    return df.sort_values("score", ascending=False)


# ---------- 5. OPTIMIZER (LP, greedy fallback) ----------
def pick_lp(df, budget, formation, max_club):
    need = FORMATIONS[formation]
    prob = pulp.LpProblem("fpl_xi", pulp.LpMaximize)
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in df.index}
    prob += pulp.lpSum(df.at[i, "score"] * x[i] for i in df.index)
    prob += pulp.lpSum(df.at[i, "price"] * x[i] for i in df.index) <= budget
    for p, n in need.items():
        prob += pulp.lpSum(x[i] for i in df.index if df.at[i, "element_type"] == p) == n
    for t in df["team"].unique():
        prob += pulp.lpSum(x[i] for i in df.index if df.at[i, "team"] == t) <= max_club
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=20))
    if pulp.LpStatus[prob.status] != "Optimal":
        return pd.DataFrame()
    return df.loc[[i for i in df.index if x[i].value() and x[i].value() > 0.5]]


def pick_greedy(df, budget, formation, max_club):
    need = dict(FORMATIONS[formation])
    picked, spent, clubs = [], 0.0, {}

    def min_cost_remaining(need_now, taken_ids):
        cost = 0.0
        for p, n in need_now.items():
            prices = sorted(df[(df["element_type"] == p) & (~df["id"].isin(taken_ids))]["price"])
            if len(prices) < n:
                return float("inf")
            cost += sum(prices[:n])
        return cost

    for _, r in df.iterrows():
        p = r["element_type"]
        if need.get(p, 0) == 0 or clubs.get(r["team"], 0) >= max_club:
            continue
        need[p] -= 1
        taken = [x["id"] for x in picked] + [r["id"]]
        if spent + r["price"] + min_cost_remaining(need, taken) <= budget + 1e-9:
            picked.append(r)
            spent += r["price"]
            clubs[r["team"]] = clubs.get(r["team"], 0) + 1
        else:
            need[p] += 1
        if sum(need.values()) == 0:
            break
    return pd.DataFrame(picked)


def pick_lineup(df, budget, formation, max_club):
    if HAS_PULP:
        try:
            lineup = pick_lp(df, budget, formation, max_club)
            if not lineup.empty:
                return lineup, "Linear programming (optimal)"
        except Exception:
            pass
    return pick_greedy(df, budget, formation, max_club), "Greedy fallback"


# ---------- UI ----------
def main():
    st.set_page_config(page_title="FPL Lineup Value Optimizer", page_icon="⚽", layout="wide")
    st.title("⚽ FPL Lineup Value Optimizer")
    st.caption("Final app · Live data: Fantasy Premier League API (bootstrap-static, fixtures?future=1)")
    st.markdown("**Decision question:** Which 11 players give me the most value for my budget "
                f"over the next {HORIZON} gameweeks?")

    with st.sidebar:
        st.header("1 · Inputs")
        budget = st.number_input("Budget (£m)", 50.0, 120.0, 83.0, 0.5)
        formation = st.selectbox("Formation", list(FORMATIONS), index=0)
        min_minutes = st.number_input("Min. minutes played", 0, 3000, 180, 30)
        max_club = st.number_input("Max players per club", 1, 3, 3)
        st.caption(f"Fixture horizon: next {HORIZON} gameweeks (fixed)")
        st.subheader("Weights (0–10)")
        w_form = st.slider("Form", 0, 10, 6)
        w_value = st.slider("Value (pts/£m)", 0, 10, 8)
        w_fix = st.slider("Fixture ease", 0, 10, 4)
        if st.button("▶ Run / Recommend lineup", type="primary", use_container_width=True):
            st.session_state["ran"] = True

    if not st.session_state.get("ran"):
        st.info("Set inputs in the sidebar and press **Run / Recommend lineup** to call the FPL API.")
        return

    try:
        with st.spinner("Calling FPL API…"):
            bootstrap, fixtures, log, fetched_at, source = fetch_fpl()
    except Exception as e:
        st.error(f"FPL API call failed: {e}. The API can be down between gameweeks — try again in a minute.")
        return

    if source == "live":
        st.success(f"API call succeeded · data fetched {fetched_at}")
    else:
        st.warning(f"Using cached data from {fetched_at} — {source}")
    with st.expander("API call log"):
        st.dataframe(pd.DataFrame(log), hide_index=True)

    pool = build_pool(bootstrap, fixtures)
    eligible, excluded = apply_guardrails(pool, min_minutes)
    ranked = score(eligible, w_form, w_value, w_fix)
    lineup, method = pick_lineup(ranked, budget, formation, max_club)

    if lineup.empty:
        st.warning("No lineup fits these constraints — raise the budget or lower the minutes threshold.")
        return
    lineup = lineup.sort_values(["element_type", "score"], ascending=[True, False])

    # ----- Results + KPIs -----
    st.header("2 · Results & KPIs")
    cost = lineup["price"].sum()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Lineup cost / budget", f"£{cost:.1f}m", f"£{budget - cost:.1f}m left")
    k2.metric("Avg points per £m", f"{lineup['pts_per_m'].mean():.2f}",
              f"{lineup['pts_per_m'].mean() - eligible['pts_per_m'].mean():+.2f} vs pool")
    k3.metric(f"Avg fixture ease (next {HORIZON})", f"{lineup['fix_ease'].mean():.1f} / 5")
    k4.metric("Avg form", f"{lineup['form'].mean():.1f}",
              f"{lineup['form'].mean() - eligible['form'].mean():+.1f} vs pool")
    k5.metric("Total value score", f"{lineup['score'].sum():.0f}", method, delta_color="off")

    st.subheader(f"Recommended Top XI ({formation})")
    show = lineup[["web_name", "pos", "club", "price", "pts_per_m", "form", "fix_ease", "score"]].copy()
    show.columns = ["Player", "Pos", "Club", "Price £m", "Pts/£m", "Form", "Fix. ease", "Score"]
    st.dataframe(show.round(2).reset_index(drop=True), hide_index=True, use_container_width=True)

    # ----- Transparency box -----
    st.header("3 · Transparency: why this recommendation?")
    total_w = max(w_form + w_value + w_fix, 1)
    with st.container(border=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**How each player is scored**")
            st.latex(r"\text{Score} = w_f\,\widetilde{\text{Form}} + w_v\,\widetilde{\text{Pts/£m}} + w_x\,\widetilde{\text{FixEase}}")
            st.markdown(
                f"- Weights normalised from your sliders: **Form {100*w_form/total_w:.0f}%**, "
                f"**Value {100*w_value/total_w:.0f}%**, **Fixture ease {100*w_fix/total_w:.0f}%**\n"
                "- Each metric is min–max scaled to 0–1 across eligible players, so scores run 0–100\n"
                f"- Fixture ease = 6 − FPL difficulty rating, averaged over next {HORIZON} GWs\n"
                f"- Selection: **{method}** maximises total score subject to "
                f"budget £{budget:.1f}m, formation {formation}, ≤{max_club} per club"
            )
        with c2:
            st.markdown("**Data, guardrails & limits**")
            st.markdown(
                f"- Source: official FPL API, fetched {fetched_at} ({'live' if source == 'live' else 'cached'})\n"
                f"- Player pool: {len(pool)} → **{len(eligible)} eligible**, "
                f"{len(excluded)} excluded (unavailable or < {min_minutes} min)\n"
                "- Assumes past form and points/£m carry forward; ignores bench, captaincy, "
                "transfers and chips\n"
                "- Not a points prediction — treat as a ranked, explainable shortlist"
            )

        st.markdown("**Score breakdown per player** (how many points each factor contributed)")
        bd = lineup.set_index("web_name")[["c_form", "c_value", "c_fix"]].round(1)
        bd.columns = ["Form", "Value", "Fixture ease"]
        st.bar_chart(bd, horizontal=True, height=360)

    with st.expander(f"Excluded players (guardrail) — {len(excluded)}"):
        ex = excluded[["web_name", "pos", "club", "reason"]].head(100)
        ex.columns = ["Player", "Pos", "Club", "Reason"]
        st.dataframe(ex, hide_index=True, use_container_width=True)


if __name__ == "__main__":
    main()
