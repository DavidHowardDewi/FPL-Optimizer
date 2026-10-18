"""
FPL Lineup Value Optimizer — Checkpoint 4 prototype
MGSC 300 · David Howard

End-to-end flow: user inputs -> live FPL API call -> ranked XI + KPIs.
Endpoints:
  https://fantasy.premierleague.com/api/bootstrap-static/
  https://fantasy.premierleague.com/api/fixtures/?future=1
"""
import time
import requests
import pandas as pd
import streamlit as st

BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"
FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/?future=1"
HEADERS = {"User-Agent": "Mozilla/5.0 (MGSC300 student prototype)"}
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


# ---------- 1. API CALL ----------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_fpl():
    log = []
    out = {}
    for name, url in [("bootstrap-static", BOOTSTRAP_URL), ("fixtures?future=1", FIXTURES_URL)]:
        t0 = time.time()
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        out[name] = r.json()
        log.append({
            "endpoint": name,
            "status": r.status_code,
            "KB": round(len(r.content) / 1024, 1),
            "ms": int((time.time() - t0) * 1000),
        })
    return out["bootstrap-static"], out["fixtures?future=1"], log


# ---------- 2. TRANSFORM (field mapping from CP2) ----------
def fixture_ease_by_team(fixtures, horizon=3):
    """Avg of (6 - difficulty) over each team's next `horizon` fixtures."""
    rows = []
    for f in fixtures:
        if f.get("finished") or f.get("event") is None:
            continue
        rows.append((f["team_h"], f["event"], 6 - f["team_h_difficulty"]))
        rows.append((f["team_a"], f["event"], 6 - f["team_a_difficulty"]))
    df = pd.DataFrame(rows, columns=["team", "event", "ease"])
    if df.empty:
        return {}
    df = df.sort_values("event")
    return df.groupby("team").head(horizon).groupby("team")["ease"].mean().to_dict()


def build_pool(bootstrap, fixtures):
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    ease = fixture_ease_by_team(fixtures)
    df = pd.DataFrame(bootstrap["elements"])
    df = df[["id", "web_name", "team", "element_type", "now_cost", "total_points",
             "minutes", "form", "status", "selected_by_percent"]].copy()
    df["price"] = df["now_cost"] / 10                       # tenths of £m -> £m
    df["form"] = pd.to_numeric(df["form"], errors="coerce").fillna(0)   # string -> float
    df["pts_per_m"] = (df["total_points"] / df["price"]).round(2)
    df["fix_ease"] = df["team"].map(ease).fillna(3.0).round(2)
    df["club"] = df["team"].map(teams)
    df["pos"] = df["element_type"].map(POS)
    return df


# ---------- 3. GUARDRAILS (CP2 Q5) ----------
def apply_guardrails(df, min_minutes):
    excl_min = df[(df["status"] == "a") & (df["minutes"] < min_minutes) & (df["total_points"] > 0)]
    excl_status = df[df["status"] != "a"]
    eligible = df[(df["status"] == "a") & (df["minutes"] >= min_minutes)].copy()
    excluded = pd.concat([
        excl_min.assign(reason=lambda d: "Small sample — " + d["minutes"].astype(str) + " min, excluded"),
        excl_status.assign(reason="Status ≠ available"),
    ])
    return eligible, excluded


# ---------- 4. SCORE + SELECT ----------
def minmax(s):
    rng = s.max() - s.min()
    return (s - s.min()) / rng if rng else s * 0


def score(df, w_form, w_value, w_fix):
    df = df.copy()
    total = (w_form + w_value + w_fix) or 1
    df["score"] = (100 * (w_form * minmax(df["form"]) +
                          w_value * minmax(df["pts_per_m"]) +
                          w_fix * minmax(df["fix_ease"])) / total).round(0).astype(int)
    return df.sort_values("score", ascending=False)


def pick_lineup(df, budget, formation, max_per_club):
    """Greedy: take highest score that fits slot, club cap, and leaves budget
    to fill remaining slots with the cheapest eligible players."""
    need = dict(FORMATIONS[formation])
    cheapest = {p: sorted(df[df["element_type"] == p]["price"].tolist()) for p in need}
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
        if need.get(p, 0) == 0 or clubs.get(r["team"], 0) >= max_per_club:
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
    lineup = pd.DataFrame(picked)
    if lineup.empty:
        return lineup
    return lineup.sort_values(["element_type", "score"], ascending=[True, False])


# ---------- UI ----------
def main():
    st.set_page_config(page_title="FPL Lineup Value Optimizer", layout="wide")
    st.title("FPL Lineup Value Optimizer")
    st.caption("Checkpoint 4 prototype · Live data: FPL API (bootstrap-static, fixtures?future=1)")
    st.markdown("**Decision question:** Which 11 players give me the most expected points "
                "for my budget over the next 3 gameweeks?")

    with st.sidebar:
        st.header("Inputs")
        budget = st.number_input("Budget (£m)", 50.0, 120.0, 83.0, 0.5)
        formation = st.selectbox("Formation", list(FORMATIONS), index=0)
        min_minutes = st.number_input("Min. minutes played", 0, 3000, 180, 30)
        max_club = st.number_input("Max players per club", 1, 3, 3)
        st.caption("Fixture horizon: next 3 gameweeks (fixed)")
        st.subheader("Weights (0–10)")
        w_form = st.slider("Form", 0, 10, 6)
        w_value = st.slider("Value (pts/£m)", 0, 10, 8)
        w_fix = st.slider("Fixture ease", 0, 10, 4)
        run = st.button("▶ Run / Recommend lineup", type="primary", use_container_width=True)

    if not run:
        st.info("Set inputs in the sidebar and press **Run / Recommend lineup** to call the FPL API.")
        return

    try:
        with st.spinner("Calling FPL API…"):
            bootstrap, fixtures, log = fetch_fpl()
    except Exception as e:
        st.error(f"FPL API call failed: {e}. The API may be down between gameweeks or blocking this host — try again in a minute.")
        return

    st.success("API call succeeded")
    st.dataframe(pd.DataFrame(log), hide_index=True)

    pool = build_pool(bootstrap, fixtures)
    eligible, excluded = apply_guardrails(pool, min_minutes)
    ranked = score(eligible, w_form, w_value, w_fix)
    lineup = pick_lineup(ranked, budget, formation, max_club)

    if len(lineup) < 11:
        st.warning(f"Only found {len(lineup)} players that fit — try a higher budget or lower minutes threshold.")
    if lineup.empty:
        return

    cost = lineup["price"].sum()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Lineup cost / budget", f"£{cost:.1f}m", f"£{budget - cost:.1f}m left")
    k2.metric("Avg points per £m", f"{lineup['pts_per_m'].mean():.2f}",
              f"vs {eligible['pts_per_m'].mean():.2f} pool avg")
    k3.metric("Avg fixture ease (next 3)", f"{lineup['fix_ease'].mean():.1f} / 5")
    k4.metric("Avg form (last ~5 GW)", f"{lineup['form'].mean():.1f}")

    st.subheader(f"Recommendation — Top XI ({formation})")
    show = lineup[["web_name", "pos", "club", "price", "pts_per_m", "form", "fix_ease", "score"]]
    show.columns = ["Player", "Pos", "Club", "Price £m", "Pts/£m", "Form", "Fix. ease", "Score"]
    st.dataframe(show.reset_index(drop=True), hide_index=True, use_container_width=True)

    with st.expander(f"Excluded players (guardrail) — {len(excluded)}"):
        ex = excluded[["web_name", "pos", "club", "reason"]].head(50)
        ex.columns = ["Player", "Pos", "Club", "Reason"]
        st.dataframe(ex, hide_index=True, use_container_width=True)

    st.caption("Score = weighted blend of Form, Value (pts/£m) and Fixture ease (min-max scaled), "
               "filled greedily into the formation within budget and the per-club cap. "
               "Not a points prediction — treat as a ranked shortlist.")


if __name__ == "__main__":
    main()
