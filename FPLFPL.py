"""
FPL Lineup Value Optimizer — Final submission (Checkpoint 5)
MGSC 300 · David Howard

Flow: user inputs -> live FPL API call -> guardrails -> transparent scoring
      -> linear-programming optimizer -> pitch view + KPIs + transparency + player database.
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
POS_COLOR = {"GKP": "#f59e0b", "DEF": "#38bdf8", "MID": "#22c55e", "FWD": "#f43f5e"}
HORIZON = 3  # gameweeks

# Tactical formations used in the Premier League. Each is a list of outfield rows
# from back to front: (players in row, FPL position). Attacking-mid rows count as MID;
# in 3-4-2-1 and 3-5-1-1 the two advanced rows behind/around the striker are FPL forwards,
# because FPL caps midfielders at 5. Every shape respects FPL limits (DEF 3-5, MID 2-5, FWD 1-3).
FORMATIONS = {
    "4-3-3": [(4, 2), (3, 3), (3, 4)],
    "4-2-3-1": [(4, 2), (2, 3), (3, 3), (1, 4)],
    "4-4-2": [(4, 2), (4, 3), (2, 4)],
    "4-1-4-1": [(4, 2), (1, 3), (4, 3), (1, 4)],
    "4-4-1-1": [(4, 2), (4, 3), (1, 3), (1, 4)],
    "4-3-2-1": [(4, 2), (3, 3), (2, 3), (1, 4)],
    "4-1-2-1-2 (diamond)": [(4, 2), (1, 3), (2, 3), (1, 3), (2, 4)],
    "4-3-1-2": [(4, 2), (3, 3), (1, 3), (2, 4)],
    "4-2-2-2": [(4, 2), (2, 3), (2, 3), (2, 4)],
    "4-5-1": [(4, 2), (5, 3), (1, 4)],
    "3-4-3": [(3, 2), (4, 3), (3, 4)],
    "3-4-2-1": [(3, 2), (4, 3), (2, 4), (1, 4)],
    "3-5-2": [(3, 2), (5, 3), (2, 4)],
    "3-4-1-2": [(3, 2), (4, 3), (1, 3), (2, 4)],
    "3-1-4-2": [(3, 2), (1, 3), (4, 3), (2, 4)],
    "3-5-1-1": [(3, 2), (5, 3), (1, 4), (1, 4)],
    "5-3-2": [(5, 2), (3, 3), (2, 4)],
    "5-4-1": [(5, 2), (4, 3), (1, 4)],
    "5-2-3": [(5, 2), (2, 3), (3, 4)],
    "5-2-1-2": [(5, 2), (2, 3), (1, 3), (2, 4)],
}


def slots(formation):
    need = {1: 1, 2: 0, 3: 0, 4: 0}
    for n, p in FORMATIONS[formation]:
        need[p] += n
    return need


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
NUMERIC = ["form", "points_per_game", "selected_by_percent", "ict_index", "expected_goals",
           "expected_assists", "expected_goal_involvements", "goals_scored", "assists",
           "clean_sheets", "bonus", "minutes", "total_points"]


def build_pool(bootstrap, fixtures):
    teams = {t["id"]: t["short_name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    for c in NUMERIC:
        if c not in df:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in ["news", "status"]:
        if c not in df:
            df[c] = ""
    df["price"] = df["now_cost"] / 10.0
    df["pos"] = df["element_type"].map(POS)
    df["club"] = df["team"].map(teams)
    df["pts_per_m"] = df["total_points"] / df["price"]

    # Fixture ease = 6 - FDR, averaged over each team's next HORIZON fixtures
    rows = []
    for f in fixtures:
        if f.get("event") is None or f.get("finished"):
            continue
        rows.append({"team": f["team_h"], "event": f["event"], "ease": 6 - f["team_h_difficulty"],
                     "opp": f"{teams.get(f['team_a'], '?')} (H)"})
        rows.append({"team": f["team_a"], "event": f["event"], "ease": 6 - f["team_a_difficulty"],
                     "opp": f"{teams.get(f['team_h'], '?').lower()} (A)"})
    ease = pd.DataFrame(rows)
    if ease.empty:
        df["fix_ease"], df["next3"] = 3.0, ""
    else:
        ease = ease.sort_values("event").groupby("team").head(HORIZON)
        df["fix_ease"] = df["team"].map(ease.groupby("team")["ease"].mean()).fillna(3.0)
        df["next3"] = df["team"].map(ease.groupby("team")["opp"].agg(", ".join)).fillna("")
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
    need = slots(formation)
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
    need = dict(slots(formation))
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


# ---------- UI helpers ----------
def html(s):
    """Render HTML without Markdown turning indented lines into code blocks."""
    st.markdown("\n".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@600;700&display=swap');
:root { --bg:#070b16; --panel:#0f1629; --panel2:#141d35; --line:rgba(148,163,184,.14);
        --text:#e8edf7; --muted:#8b97b0; --accent:#22c55e; --accent2:#a78bfa; }
html, body, [class*="css"], .stApp { font-family:'Inter',sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, rgba(167,139,250,.12), transparent 60%),
         radial-gradient(900px 500px at 100% 0%, rgba(34,197,94,.10), transparent 60%), var(--bg); color:var(--text); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0c1324,#080d1a); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color: var(--text); }
.block-container { padding-top: 1.6rem; max-width: 1280px; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.01em; }
.stTabs [data-baseweb="tab-list"] { gap:6px; background:var(--panel); padding:6px; border-radius:14px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { border-radius:10px; padding:8px 18px; color:var(--muted); font-weight:600; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,rgba(34,197,94,.22),rgba(167,139,250,.22)); color:var(--text)!important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
[data-testid="stTabs"] [role="tablist"] { gap:6px; background:var(--panel); padding:6px; border-radius:14px; border:1px solid var(--line); }
[data-testid="stTabs"] button[role="tab"] { border-radius:10px; padding:8px 18px; font-weight:600; border:0; }
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] { background:linear-gradient(135deg,rgba(34,197,94,.25),rgba(167,139,250,.25)); }
div.stButton > button[kind="primary"] { background:linear-gradient(135deg,#22c55e,#16a34a); border:0; border-radius:12px;
  font-weight:700; padding:.7rem 1rem; box-shadow:0 8px 24px rgba(34,197,94,.25); transition:transform .15s; }
div.stButton > button[kind="primary"]:hover { transform:translateY(-1px); }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:14px; overflow:hidden; }

/* hero */
.hero { display:flex; align-items:center; gap:22px; padding:22px 26px; border-radius:20px;
  background:linear-gradient(135deg,rgba(20,29,53,.95),rgba(12,19,36,.95)); border:1px solid var(--line);
  box-shadow:0 20px 60px rgba(0,0,0,.35); margin-bottom:18px; overflow:hidden; position:relative; }
.hero::after { content:""; position:absolute; inset:0; background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.05) 50%,transparent 70%);
  transform:translateX(-100%); animation:sweep 3.5s ease-in-out 0.6s infinite; }
.hero h1 { margin:0; font-size:2.1rem; line-height:1.1; background:linear-gradient(90deg,#fff,#c4b5fd 60%,#86efac);
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:rise .9s ease-out both .3s; }
.hero p { margin:.35rem 0 0; color:var(--muted); font-size:.98rem; animation:rise .9s ease-out both .5s; }
.crest { flex:0 0 auto; width:86px; height:86px; animation:pop .9s cubic-bezier(.2,1.4,.4,1) both; }
.crest .ring { transform-origin:50px 50px; animation:spin 9s linear infinite; }
.crest .ball { transform-origin:50px 52px; animation:bounce 2.4s ease-in-out infinite; }
.crest .glow { animation:pulse 2.4s ease-in-out infinite; }
@keyframes spin { to { transform:rotate(360deg);} }
@keyframes bounce { 0%,100% { transform:translateY(0) rotate(0);} 50% { transform:translateY(-3px) rotate(18deg);} }
@keyframes pulse { 0%,100% { opacity:.35;} 50% { opacity:.8;} }
@keyframes pop { from { transform:scale(.3) rotate(-30deg); opacity:0;} to { transform:scale(1) rotate(0); opacity:1;} }
@keyframes rise { from { transform:translateY(10px); opacity:0;} to { transform:none; opacity:1;} }
@keyframes sweep { 0% { transform:translateX(-100%);} 60%,100% { transform:translateX(100%);} }

/* kpis */
.kpis { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; margin:6px 0 16px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:14px 16px; animation:rise .6s ease-out both; }
.kpi .l { color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.08em; font-weight:600; }
.kpi .v { font-family:'Space Grotesk',sans-serif; font-size:1.55rem; font-weight:700; margin-top:4px; }
.kpi .d { font-size:.78rem; margin-top:2px; color:var(--accent); }
.kpi .d.n { color:var(--muted); }
@media (max-width:900px) { .kpis { grid-template-columns:repeat(2,1fr);} .hero h1 { font-size:1.5rem; } }

/* status pill */
.pill { display:inline-flex; align-items:center; gap:8px; padding:6px 12px; border-radius:999px; font-size:.82rem; font-weight:600;
  background:rgba(34,197,94,.12); color:#86efac; border:1px solid rgba(34,197,94,.3); margin-bottom:10px; }
.pill.warn { background:rgba(245,158,11,.12); color:#fcd34d; border-color:rgba(245,158,11,.3); }
.pill .dot { width:8px; height:8px; border-radius:50%; background:currentColor; animation:pulse 1.6s infinite; }

/* pitch */
.pitch { position:relative; border-radius:20px; padding:22px 10px; min-height:560px; display:flex; flex-direction:column; justify-content:space-between;
  background:repeating-linear-gradient(180deg,#14532d 0 56px,#166534 56px 112px); border:2px solid rgba(255,255,255,.18);
  box-shadow:inset 0 0 80px rgba(0,0,0,.35); overflow:hidden; }
.pitch::before { content:""; position:absolute; left:6%; right:6%; top:50%; border-top:2px solid rgba(255,255,255,.22); }
.pitch::after { content:""; position:absolute; width:120px; height:120px; left:calc(50% - 60px); top:calc(50% - 60px);
  border:2px solid rgba(255,255,255,.22); border-radius:50%; }
.box { position:absolute; left:30%; right:30%; height:70px; border:2px solid rgba(255,255,255,.22); }
.box.top { top:-2px; border-top:0; } .box.bot { bottom:-2px; border-bottom:0; }
.prow { position:relative; z-index:1; display:flex; justify-content:space-evenly; align-items:center; gap:6px; }
.chip { width:118px; text-align:center; animation:rise .5s ease-out both; }
.chip .shirt { width:44px; height:44px; margin:0 auto 6px; border-radius:12px; display:flex; align-items:center; justify-content:center;
  font-weight:800; font-size:.72rem; color:#0b1020; box-shadow:0 6px 16px rgba(0,0,0,.35); }
.chip .nm { background:rgba(7,11,22,.88); color:#fff; font-weight:700; font-size:.8rem; padding:4px 6px; border-radius:8px 8px 0 0;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.chip .meta { background:rgba(255,255,255,.92); color:#0b1020; font-size:.7rem; font-weight:600; padding:3px 6px; border-radius:0 0 8px 8px; }
@media (max-width:700px) { .chip { width:74px; } .chip .nm { font-size:.66rem; } .chip .meta { font-size:.6rem; } }

/* panels */
.panel { background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:16px 18px; height:100%; }
.panel h4 { margin:0 0 8px; font-family:'Space Grotesk',sans-serif; font-size:1rem; }
.panel ul { margin:0; padding-left:18px; color:#cbd5e1; font-size:.9rem; line-height:1.55; }
.wbar { display:flex; height:12px; border-radius:999px; overflow:hidden; margin:8px 0 4px; }
.legend span { display:inline-flex; align-items:center; gap:6px; margin-right:14px; font-size:.8rem; color:var(--muted); }
.legend i { width:10px; height:10px; border-radius:3px; display:inline-block; }
</style>
"""

CREST = """
<svg class="crest" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" aria-label="App crest">
<defs>
<linearGradient id="cg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a78bfa"/><stop offset="1" stop-color="#22c55e"/></linearGradient>
<radialGradient id="gl"><stop offset="0" stop-color="#22c55e" stop-opacity=".9"/><stop offset="1" stop-color="#22c55e" stop-opacity="0"/></radialGradient>
</defs>
<circle class="glow" cx="50" cy="52" r="34" fill="url(#gl)"/>
<path d="M50 6 L86 18 V48 C86 70 70 86 50 94 C30 86 14 70 14 48 V18 Z" fill="#0f1629" stroke="url(#cg)" stroke-width="3"/>
<g class="ring"><circle cx="50" cy="50" r="40" fill="none" stroke="url(#cg)" stroke-width="1.5" stroke-dasharray="4 7" opacity=".7"/></g>
<g class="ball">
<circle cx="50" cy="52" r="18" fill="#fff"/>
<polygon points="50,44 57,49 54.5,57 45.5,57 43,49" fill="#0b1020"/>
<path d="M50 44 V36 M57 49 L65 46 M54.5 57 L59 64 M45.5 57 L41 64 M43 49 L35 46" stroke="#0b1020" stroke-width="2"/>
</g>
</svg>
"""


def pitch_html(lineup, formation):
    rows = FORMATIONS[formation]
    by_pos = {p: lineup[lineup["element_type"] == p].sort_values("score", ascending=False).to_dict("records")
              for p in (1, 2, 3, 4)}
    # fill the most advanced rows of each FPL position first
    assigned = [None] * len(rows)
    for idx in range(len(rows) - 1, -1, -1):
        n, p = rows[idx]
        assigned[idx] = [by_pos[p].pop(0) for _ in range(n) if by_pos[p]]

    def chip(r):
        c = POS_COLOR[r["pos"]]
        return (f'<div class="chip"><div class="shirt" style="background:{c}">{r["pos"]}</div>'
                f'<div class="nm">{r["web_name"]}</div>'
                f'<div class="meta">{r["club"]} · £{r["price"]:.1f}m · {r["score"]:.0f}</div></div>')

    out = ['<div class="pitch"><div class="box top"></div><div class="box bot"></div>']
    for players in reversed(assigned):  # attack at the top
        out.append('<div class="prow">' + "".join(chip(r) for r in players) + "</div>")
    out.append('<div class="prow">' + "".join(chip(r) for r in by_pos[1]) + "</div></div>")
    return "".join(out)


def kpi(label, value, delta="", neutral=False):
    d = f'<div class="d{" n" if neutral else ""}">{delta}</div>' if delta else ""
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>{d}</div>'


# ---------- UI ----------
def main():
    st.set_page_config(page_title="FPL Lineup Value Optimizer", page_icon="⚽", layout="wide")
    html(CSS)
    html(f'<div class="hero">{CREST}<div><h1>FPL Lineup Value Optimizer</h1>'
         f'<p>Which 11 players give you the most value for your budget over the next {HORIZON} gameweeks? '
         f'Live data from the official Fantasy Premier League API.</p></div></div>')

    with st.sidebar:
        st.markdown("### ⚙️ Inputs")
        budget = st.number_input("Budget (£m)", 50.0, 120.0, 83.0, 0.5)
        formation = st.selectbox("Formation", list(FORMATIONS), index=0,
                                 help="Tactical shapes used in the Premier League, mapped to FPL position slots.")
        s = slots(formation)
        st.caption(f"FPL slots: 1 GKP · {s[2]} DEF · {s[3]} MID · {s[4]} FWD")
        min_minutes = st.number_input("Min. minutes played", 0, 3000, 180, 30)
        max_club = st.number_input("Max players per club", 1, 3, 3)
        st.caption(f"Fixture horizon: next {HORIZON} gameweeks")
        st.markdown("### 🎚️ Weights")
        w_form = st.slider("Form", 0, 10, 6)
        w_value = st.slider("Value (pts/£m)", 0, 10, 8)
        w_fix = st.slider("Fixture ease", 0, 10, 4)
        if st.button("▶  Run / Recommend lineup", type="primary", width="stretch"):
            st.session_state["ran"] = True

    if not st.session_state.get("ran"):
        html('<div class="panel"><h4>👋 Ready when you are</h4><ul>'
             '<li>Set your budget, formation and weights in the sidebar</li>'
             '<li>Press <b>Run / Recommend lineup</b> to pull live FPL data</li>'
             '<li>Explore the pitch, the transparency breakdown and the full player database</li></ul></div>')
        return

    try:
        with st.spinner("Calling the FPL API…"):
            bootstrap, fixtures, log, fetched_at, source = fetch_fpl()
    except Exception as e:
        st.error(f"FPL API call failed: {e}. The API can be down between gameweeks — try again in a minute.")
        return

    pool = build_pool(bootstrap, fixtures)
    eligible, excluded = apply_guardrails(pool, min_minutes)
    ranked = score(eligible, w_form, w_value, w_fix)
    lineup, method = pick_lineup(ranked, budget, formation, max_club)

    if source == "live":
        html(f'<div class="pill"><span class="dot"></span>Live API call succeeded · {fetched_at} · '
             f'{len(pool)} players loaded</div>')
    else:
        html(f'<div class="pill warn"><span class="dot"></span>Using cached data from {fetched_at}</div>')

    tab_xi, tab_why, tab_db = st.tabs(["🏟️  Lineup & KPIs", "🔍  Transparency", "📋  All Players"])

    # ----- Tab 1: Results + KPIs -----
    with tab_xi:
        if lineup.empty:
            st.warning("No lineup fits these constraints — raise the budget or lower the minutes threshold.")
        else:
            lineup = lineup.sort_values(["element_type", "score"], ascending=[True, False])
            cost = lineup["price"].sum()
            ppm, ppm_pool = lineup["pts_per_m"].mean(), eligible["pts_per_m"].mean()
            frm, frm_pool = lineup["form"].mean(), eligible["form"].mean()
            html('<div class="kpis">'
                 + kpi("Cost / budget", f"£{cost:.1f}m", f"£{budget - cost:.1f}m in the bank")
                 + kpi("Avg pts per £m", f"{ppm:.2f}", f"{ppm - ppm_pool:+.2f} vs pool")
                 + kpi(f"Fixture ease (next {HORIZON})", f"{lineup['fix_ease'].mean():.1f}/5", "higher = easier", True)
                 + kpi("Avg form", f"{frm:.1f}", f"{frm - frm_pool:+.1f} vs pool")
                 + kpi("Total value score", f"{lineup['score'].sum():.0f}", method, True)
                 + "</div>")
            c1, c2 = st.columns([1.15, 1])
            with c1:
                st.markdown(f"#### Recommended XI · {formation}")
                html(pitch_html(lineup, formation))
            with c2:
                st.markdown("#### Lineup details")
                show = lineup[["web_name", "pos", "club", "price", "pts_per_m", "form", "fix_ease", "next3", "score"]].copy()
                st.dataframe(
                    show.reset_index(drop=True), hide_index=True, width="stretch", height=423,
                    column_config={
                        "web_name": "Player", "pos": "Pos", "club": "Club",
                        "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
                        "pts_per_m": st.column_config.NumberColumn("Pts/£m", format="%.1f"),
                        "form": st.column_config.NumberColumn("Form", format="%.1f"),
                        "fix_ease": st.column_config.NumberColumn("Fix. ease", format="%.1f"),
                        "next3": "Next 3",
                        "score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
                    })

    # ----- Tab 2: Transparency box -----
    with tab_why:
        st.markdown("#### Why this recommendation?")
        tw = max(w_form + w_value + w_fix, 1)
        pf, pv, px = 100 * w_form / tw, 100 * w_value / tw, 100 * w_fix / tw
        c1, c2 = st.columns(2)
        with c1:
            html(f'<div class="panel"><h4>🧮 How each player is scored</h4>'
                 f'<div class="wbar"><div style="width:{pf}%;background:#a78bfa"></div>'
                 f'<div style="width:{pv}%;background:#22c55e"></div><div style="width:{px}%;background:#38bdf8"></div></div>'
                 f'<div class="legend"><span><i style="background:#a78bfa"></i>Form {pf:.0f}%</span>'
                 f'<span><i style="background:#22c55e"></i>Value {pv:.0f}%</span>'
                 f'<span><i style="background:#38bdf8"></i>Fixture ease {px:.0f}%</span></div><ul>'
                 f'<li>Score (0–100) = weighted sum of Form, Points per £m and Fixture ease, each min–max scaled 0–1 '
                 f'across eligible players. Weights come from your sliders.</li>'
                 f'<li>Fixture ease = 6 − FPL difficulty rating, averaged over each club\'s next {HORIZON} fixtures.</li>'
                 f'<li><b>{method}</b> picks the XI with the highest total score while staying within '
                 f'£{budget:.1f}m, the {formation} shape ({s[2]} DEF · {s[3]} MID · {s[4]} FWD + 1 GKP) and '
                 f'≤{max_club} players per club.</li></ul></div>')
        with c2:
            html(f'<div class="panel"><h4>🛡️ Data, guardrails & limits</h4><ul>'
                 f'<li>Source: official FPL API (bootstrap-static + fixtures), fetched {fetched_at} '
                 f'({"live" if source == "live" else "cached"}).</li>'
                 f'<li>Pool: {len(pool)} players → <b>{len(eligible)} eligible</b>, {len(excluded)} excluded '
                 f'(injured/doubtful/suspended or under {min_minutes} minutes).</li>'
                 f'<li>Formations are mapped to FPL slots: attacking-mid rows count as MID; in 3-4-2-1 and 3-5-1-1 the '
                 f'advanced pair are FWD because FPL allows at most 5 midfielders.</li>'
                 f'<li>Assumes recent form and points per £m carry forward. Ignores bench, captaincy, transfers and chips. '
                 f'Not a points prediction — an explainable shortlist.</li></ul></div>')
        if not lineup.empty:
            st.markdown("#### Score breakdown per player")
            st.caption("How many of each player's score points came from each factor.")
            bd = lineup.sort_values("score", ascending=False).set_index("web_name")[["c_form", "c_value", "c_fix"]].round(1)
            bd.columns = ["Form", "Value", "Fixture ease"]
            st.bar_chart(bd, horizontal=True, height=400, color=["#a78bfa", "#22c55e", "#38bdf8"])
        with st.expander(f"Excluded players (guardrail) — {len(excluded)}"):
            ex = excluded[["web_name", "pos", "club", "reason", "news"]].copy()
            ex.columns = ["Player", "Pos", "Club", "Reason", "FPL news"]
            st.dataframe(ex, hide_index=True, width="stretch")
        with st.expander("API call log"):
            st.dataframe(pd.DataFrame(log), hide_index=True)

    # ----- Tab 3: All players -----
    with tab_db:
        st.markdown(f"#### Player database · {len(pool)} players")
        f1, f2, f3, f4 = st.columns([1.4, 1, 1, 1.2])
        q = f1.text_input("Search player", placeholder="e.g. Salah")
        pos_f = f2.multiselect("Position", ["GKP", "DEF", "MID", "FWD"])
        club_f = f3.multiselect("Club", sorted(pool["club"].dropna().unique()))
        pmin, pmax = float(pool["price"].min()), float(pool["price"].max())
        price_f = f4.slider("Price (£m)", pmin, pmax, (pmin, pmax), 0.5)
        db = pool.copy()
        db["value_score"] = db["id"].map(ranked.set_index("id")["score"])
        if q:
            db = db[db["web_name"].str.contains(q, case=False, na=False)
                    | db.get("second_name", db["web_name"]).astype(str).str.contains(q, case=False, na=False)]
        if pos_f:
            db = db[db["pos"].isin(pos_f)]
        if club_f:
            db = db[db["club"].isin(club_f)]
        db = db[db["price"].between(*price_f)]
        db["status"] = db["status"].map({"a": "✅", "d": "⚠️", "i": "🚑", "s": "⛔", "u": "—", "n": "—"}).fillna("—")
        cols = ["web_name", "pos", "club", "status", "price", "total_points", "form", "points_per_game", "pts_per_m",
                "minutes", "goals_scored", "assists", "clean_sheets", "bonus", "expected_goals", "expected_assists",
                "ict_index", "selected_by_percent", "next3", "value_score"]
        st.dataframe(
            db[cols].sort_values("total_points", ascending=False), hide_index=True, width="stretch", height=620,
            column_config={
                "web_name": "Player", "pos": "Pos", "club": "Club", "status": "Status",
                "price": st.column_config.NumberColumn("Price", format="£%.1fm"),
                "total_points": st.column_config.NumberColumn("Pts", format="%d"),
                "form": st.column_config.NumberColumn("Form", format="%.1f"),
                "points_per_game": st.column_config.NumberColumn("PPG", format="%.1f"),
                "pts_per_m": st.column_config.NumberColumn("Pts/£m", format="%.1f"),
                "minutes": st.column_config.NumberColumn("Mins", format="%d"),
                "goals_scored": st.column_config.NumberColumn("G", format="%d"),
                "assists": st.column_config.NumberColumn("A", format="%d"),
                "clean_sheets": st.column_config.NumberColumn("CS", format="%d"),
                "bonus": st.column_config.NumberColumn("Bonus", format="%d"),
                "expected_goals": st.column_config.NumberColumn("xG", format="%.2f"),
                "expected_assists": st.column_config.NumberColumn("xA", format="%.2f"),
                "ict_index": st.column_config.NumberColumn("ICT", format="%.1f"),
                "selected_by_percent": st.column_config.NumberColumn("Owned", format="%.1f%%"),
                "next3": "Next 3 fixtures",
                "value_score": st.column_config.ProgressColumn("Value score", min_value=0, max_value=100, format="%.0f",
                                                               help="Blank = excluded by guardrails"),
            })
        st.caption("✅ available · ⚠️ doubtful · 🚑 injured · ⛔ suspended. Value score uses your current weights.")


if __name__ == "__main__":
    main()
