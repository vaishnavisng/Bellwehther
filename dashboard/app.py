"""Bellwether — product-feedback early-warning dashboard.

READ-ONLY: renders precomputed DuckDB tables written by run_pipeline.py.
No TF-IDF, KMeans, regression, anomaly detection, or prediction runs here.

Three pages (sidebar navigation), one shared cached data load:
  1. Executive Overview  — KPIs + early warning
  2. Issue Analytics     — emerging table, trends, rating impact
  3. Issue Detail        — explanation + review evidence

Launch:  streamlit run dashboard/app.py   (after `python run_pipeline.py --sample`)
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make `src`/`dashboard` importable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    compute_kpis,
    db_path,
    load_all,
    representative_reviews,
    warning_ranked,
)

st.set_page_config(page_title="Bellwether", page_icon="📡", layout="wide")

RISK_COLOR = {"LOW": "#4b6b57", "MEDIUM": "#b8860b",
              "HIGH": "#c25a2b", "CRITICAL": "#b3261e"}

# Distinct accent per executive KPI card.
KPI_COLORS = ["#2563eb", "#0891b2", "#db2777", "#7c3aed", "#d97706", "#dc2626"]

# Vivid, high-contrast colorway shared by all charts.
COLORWAY = ["#2563eb", "#db2777", "#0891b2", "#7c3aed", "#d97706", "#16a34a"]

PAGES = ["Executive Overview", "Issue Analytics", "Issue Detail"]


def flashy(fig, height=360):
    """Consistent, punchy chart styling (theme-aware: transparent background)."""
    fig.update_layout(
        height=height, colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13), title_font=dict(size=17),
        margin=dict(t=52, l=10, r=10, b=10),
        hoverlabel=dict(font_size=12), bargap=0.28)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    return fig

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1300px;}
.kpi {border-radius: 12px; padding: 16px 18px; border: 1px solid rgba(128,128,128,.15);
      border-left-width: 6px; box-shadow: 0 1px 3px rgba(0,0,0,.06);}
.kpi .label {font-size: .74rem; text-transform: uppercase; letter-spacing: .05em;
      opacity: .8; font-weight: 600;}
.kpi .value {font-size: 1.85rem; font-weight: 800; line-height: 1.25;}
.kpi .sub {font-size: .8rem; opacity: .7;}
.badge {display:inline-block; padding: 2px 10px; border-radius: 999px;
      color: #fff; font-weight: 600; font-size: .8rem;}
.warn-card {border-left: 6px solid var(--accent); background: var(--secondary-background-color);
      border-radius: 8px; padding: 14px 18px; margin-bottom: 10px;}

/* Colorful gradient dashboard header */
.app-header {font-size: 2.2rem; font-weight: 800; line-height: 1.4;
      padding: .1em 0 .25em; margin: 0 0 .3rem; display: inline-block;
      color: #121A3D;}

/* Section subtitles on every page */
.block-container h2, .block-container h3 {color: #070B1D !important;}

/* Larger, clearer sidebar */
section[data-testid="stSidebar"] {min-width: 300px;}
section[data-testid="stSidebar"] h1 {font-size: 1.6rem !important;}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {font-size: 1.2rem !important;}
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] div[role="radiogroup"] label,
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] p {font-size: 1.08rem !important;}
section[data-testid="stSidebar"] div[role="radiogroup"] label {padding: 3px 0;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def get_data(_mtime: float) -> dict:
    return load_all()


@st.cache_data(show_spinner=False)
def get_reviews(issue_id: str, _mtime: float) -> pd.DataFrame:
    return representative_reviews(issue_id)


def badge(level: str) -> str:
    c = RISK_COLOR.get(level, "#666")
    return f'<span class="badge" style="background:{c}">{level}</span>'


def kpi_card(col, label, value, sub="", color="#2563eb"):
    col.markdown(
        f'<div class="kpi" style="border-left-color:{color}; background:{color}14">'
        f'<div class="label" style="color:{color}">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True)


def fmt_pct(x, d=1):
    return "—" if pd.isna(x) else f"{x*100:.{d}f}%"


def fmt_stars(x, d=2):
    return "—" if pd.isna(x) else f"{x:+.{d}f}★"


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_overview(ctx):
    kpis, ranked = ctx["kpis"], ctx["ranked"]

    st.subheader("Executive summary")
    c = st.columns(6)
    kpi_card(c[0], "Total reviews", f"{kpis['total_reviews']:,}", color=KPI_COLORS[0])
    delta = kpis["rating_trend_delta"]
    trend_sub = ("—" if delta is None else
                 f"{'▲' if delta > 0 else '▼' if delta < 0 else '■'} {delta:+.2f} vs prev month")
    kpi_card(c[1], "Average rating",
             "—" if kpis["avg_rating"] is None else f"{kpis['avg_rating']:.2f}★",
             trend_sub, color=KPI_COLORS[1])
    kpi_card(c[2], "Negative reviews",
             "—" if kpis["negative_pct"] is None else f"{kpis['negative_pct']:.1f}%",
             "rated 1–2★", color=KPI_COLORS[2])
    kpi_card(c[3], "Issues tracked", f"{kpis['n_issues']}", color=KPI_COLORS[3])
    kpi_card(c[4], "Emerging issues", f"{kpis['n_emerging']}", "risk ≥ Medium",
             color=KPI_COLORS[4])
    kpi_card(c[5], "High-risk issues", f"{kpis['n_high_risk']}", "High / Critical",
             color=KPI_COLORS[5])

    if not kpis["rating_by_month"].empty and len(kpis["rating_by_month"]) >= 2:
        bm = kpis["rating_by_month"]
        fig = px.line(bm, x="review_month", y="rating", markers=True,
                      title="Overall average rating over time")
        fig.update_traces(line=dict(width=4, shape="spline", color="#2563eb"),
                          marker=dict(size=10, line=dict(width=2, color="#fff")),
                          fill="tozeroy", fillcolor="rgba(37,99,235,.15)")
        flashy(fig, height=260)
        fig.update_layout(yaxis_title="avg rating", xaxis_title="month")
        fig.update_yaxes(range=[1, 5])
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("🚨 Bellwether — early warning")
    if ranked.empty:
        st.info("No predictions available.")
        return
    top = ranked.iloc[0]
    acc = RISK_COLOR.get(top["risk_level"], "#666")
    st.markdown(
        f'<div class="warn-card" style="--accent:{acc}">'
        f'<div style="font-size:1.15rem;font-weight:700">'
        f'{top["issue_label"] or top["issue_id"]} &nbsp; {badge(top["risk_level"])} '
        f'<span class="sub">confidence: {top["confidence_level"]} · {top["horizon"]}</span></div>'
        f'<div style="margin-top:6px">{top["explanation"]}</div></div>',
        unsafe_allow_html=True)

    others = ranked.iloc[1:4]
    if not others.empty:
        cols = st.columns(len(others))
        for col, (_, r) in zip(cols, others.iterrows()):
            col.markdown(
                f'{badge(r["risk_level"])}  **{r["issue_label"] or r["issue_id"]}**',
                unsafe_allow_html=True)
            col.metric("Predicted impact", fmt_stars(r["predicted_rating_impact"]),
                       help="Projected change in overall rating over the horizon")
            col.caption(f"share {fmt_pct(r['current_share'])} · "
                        f"growth {fmt_pct(r['recent_growth'],0)} · "
                        f"hist {fmt_stars(r['historical_rating_impact'])}")


def page_analytics(ctx):
    ranked, trends, impact = ctx["ranked"], ctx["trends"], ctx["impact"]
    issue_opts, labels, selected = ctx["issue_opts"], ctx["labels"], ctx["selected"]

    # ---- Emerging issues table -------------------------------------------- #
    st.subheader("Emerging issues")
    if ranked.empty:
        st.info("No issues to display.")
    else:
        levels = st.multiselect("Filter by risk", list(RISK_COLOR),
                                default=list(RISK_COLOR))
        view = ranked[ranked["risk_level"].isin(levels)]
        disp = pd.DataFrame({
            "Issue": view["issue_label"].fillna(view["issue_id"]),
            "Risk": view["risk_level"],
            "Confidence": view["confidence_level"],
            "Current share %": pd.to_numeric(view["current_share"], errors="coerce") * 100,
            "Growth %": pd.to_numeric(view["recent_growth"], errors="coerce") * 100,
            "Historical impact ★": view["historical_rating_impact"],
            "Predicted impact ★": view["predicted_rating_impact"],
            "CI low": view["lower_bound"], "CI high": view["upper_bound"],
        })
        st.dataframe(
            disp, width="stretch", hide_index=True,
            column_config={
                "Current share %": st.column_config.NumberColumn(format="%.1f%%"),
                "Growth %": st.column_config.NumberColumn(format="%.0f%%"),
                "Historical impact ★": st.column_config.NumberColumn(format="%.2f"),
                "Predicted impact ★": st.column_config.NumberColumn(format="%.2f"),
                "CI low": st.column_config.NumberColumn(format="%.2f"),
                "CI high": st.column_config.NumberColumn(format="%.2f"),
            })
        st.caption("Click a column header to sort. All values are precomputed.")

    st.divider()

    # ---- Issue trends ----------------------------------------------------- #
    st.subheader("Issue trends over time")
    if trends.empty:
        st.info("No trend data available.")
    else:
        default = [selected] if selected else issue_opts[:2]
        chosen = st.multiselect(
            "Issues to plot", issue_opts,
            default=[i for i in default if i in issue_opts] or issue_opts[:2],
            format_func=lambda i: labels.get(i, i))
        sub = trends[trends["issue_id"].isin(chosen)].copy()
        if sub.empty:
            st.info("Select at least one issue.")
        else:
            sub["date"] = pd.to_datetime(sub["date"])
            fig = go.Figure()
            for i, (iid, g) in enumerate(sub.groupby("issue_id")):
                g = g.sort_values("date")
                name = labels.get(iid, iid)
                color = COLORWAY[i % len(COLORWAY)]
                fig.add_trace(go.Scatter(
                    x=g["date"], y=g["issue_share"] * 100, mode="lines+markers",
                    name=name, line=dict(width=3.5, shape="spline", color=color),
                    marker=dict(size=8)))
                fig.add_trace(go.Scatter(x=g["date"], y=g["rolling_baseline"] * 100,
                                         mode="lines", line=dict(dash="dot", width=1.5,
                                         color=color), name=f"{name} baseline",
                                         showlegend=False, opacity=0.55))
                an = g[g["anomaly_flag"] == True]  # noqa: E712
                if not an.empty:
                    fig.add_trace(go.Scatter(
                        x=an["date"], y=an["issue_share"] * 100, mode="markers",
                        marker=dict(symbol="star", size=16, color="#dc2626",
                                    line=dict(width=1, color="#fff")),
                        name=f"{name} anomaly", showlegend=False))
            flashy(fig, height=400)
            fig.update_layout(title="Issue prevalence (% of reviews) with "
                              "rolling baseline and anomaly periods",
                              xaxis_title="week", yaxis_title="share of reviews (%)",
                              legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, width="stretch")
            st.caption("Dotted line = rolling baseline · ⭐ = anomaly period "
                       "(unusual jump vs baseline).")

    st.divider()

    # ---- Rating impact ---------------------------------------------------- #
    st.subheader("Historical rating impact")
    if impact.empty:
        st.info("No rating-impact data available.")
        return
    imp = impact.copy()
    imp["Issue"] = imp["issue_label"].fillna(imp["issue_id"])
    overall = float(imp["overall_rating"].iloc[0]) if "overall_rating" in imp else None
    left, right = st.columns(2)

    melted = imp.melt(id_vars="Issue",
                      value_vars=["average_issue_rating", "average_non_issue_rating"],
                      var_name="group", value_name="rating")
    melted["group"] = melted["group"].map({
        "average_issue_rating": "when issue present",
        "average_non_issue_rating": "when absent"})
    fig1 = px.bar(melted, x="Issue", y="rating", color="group", barmode="group",
                  title="Average rating: issue present vs absent",
                  color_discrete_map={"when issue present": "#dc2626",
                                      "when absent": "#16a34a"})
    fig1.update_traces(marker_line_width=0, opacity=0.92)
    if overall is not None:
        fig1.add_hline(y=overall, line_dash="dash", line_color="#7c3aed",
                       annotation_text=f"overall {overall:.2f}★")
    flashy(fig1, height=360)
    fig1.update_layout(yaxis_title="avg rating", yaxis_range=[1, 5],
                       legend=dict(orientation="h", y=-0.3))
    left.plotly_chart(fig1, width="stretch")

    eff = imp.dropna(subset=["regression_effect"])
    if not eff.empty:
        fig2 = go.Figure(go.Bar(
            x=eff["regression_effect"], y=eff["Issue"], orientation="h",
            error_x=dict(type="data", symmetric=False,
                         array=eff["regression_ci_high"] - eff["regression_effect"],
                         arrayminus=eff["regression_effect"] - eff["regression_ci_low"],
                         color="#7c3aed", thickness=2),
            marker=dict(color=eff["regression_effect"], colorscale="RdYlGn",
                        cmid=0, line=dict(width=0))))
        flashy(fig2, height=360)
        fig2.update_layout(title="Estimated rating penalty (★) with 95% CI",
                           xaxis_title="rating change vs other reviews (stars)")
        right.plotly_chart(fig2, width="stretch")


def page_detail(ctx):
    ranked, impact, selected = ctx["ranked"], ctx["impact"], ctx["selected"]
    mtime = ctx["mtime"]

    st.subheader("Why is Bellwether warning us?")
    if selected is None or ranked.empty:
        st.info("Select an issue in the sidebar.")
        return
    row = ranked[ranked["issue_id"] == selected].iloc[0]
    st.markdown(
        f'<div style="display:inline-block; background:#fde2e2; color:#8b0000; '
        f'font-size:1.25rem; font-weight:700; padding:8px 16px; border-radius:8px; '
        f'border:1px solid #f2b8b8;">{row["issue_label"] or selected}</div> '
        f'&nbsp; {badge(row["risk_level"])}', unsafe_allow_html=True)
    st.write(row["explanation"])
    m = st.columns(5)
    m[0].metric("Current share", fmt_pct(row["current_share"]))
    m[1].metric("Recent growth", fmt_pct(row["recent_growth"], 0))
    m[2].metric("Historical impact", fmt_stars(row["historical_rating_impact"]))
    m[3].metric("Predicted impact", fmt_stars(row["predicted_rating_impact"]),
                help=f"CI [{row['lower_bound']:.2f}, {row['upper_bound']:.2f}]"
                if pd.notna(row["lower_bound"]) else None)
    m[4].metric("Confidence", str(row["confidence_level"]).title())

    irow = impact[impact["issue_id"] == selected] if not impact.empty else pd.DataFrame()
    if not irow.empty and pd.notna(irow.iloc[0].get("interpretation")):
        st.info(irow.iloc[0]["interpretation"])

    # ---- What is this issue, really? (from users' own words) -------------- #
    try:
        ev = get_reviews(selected, mtime)
    except Exception as e:
        st.error(f"Could not load reviews: {e}")
        ev = pd.DataFrame()

    st.divider()
    st.subheader("🗣️ What are users actually facing?")
    st.markdown(_issue_gloss(ctx["kw_map"].get(selected, ""),
                             irow.iloc[0] if not irow.empty else None))
    if not ev.empty:
        st.caption("In users' own words — the most representative complaints:")
        for _, rv in ev.head(3).iterrows():
            d = pd.to_datetime(rv["review_date"]).date()
            txt = str(rv["review_text"]).strip().replace("\n", " ")
            st.markdown(f"> {txt}\n>\n> — **{int(rv['rating'])}★** · "
                        f"{rv['source_platform']} · {d}")

    st.divider()
    st.subheader("All representative reviews")
    if ev.empty:
        st.info("No reviews found for this issue.")
        return
    ev = ev.assign(review_date=pd.to_datetime(ev["review_date"]).dt.date)
    st.dataframe(
        ev.rename(columns={"review_date": "Date", "rating": "Rating",
                           "source_platform": "Source", "review_text": "Review"}),
        width="stretch", hide_index=True,
        column_config={"Review": st.column_config.TextColumn(width="large")})
    st.caption("Worst-rated examples for context. No reviewer names or "
               "personal identifiers are stored or shown.")


def _issue_gloss(keywords: str, imp_row) -> str:
    """Plain-English 'what is this issue' line built from stored stats + terms
    (no model runs here — just reads precomputed values)."""
    bits = []
    if imp_row is not None:
        n = int(imp_row.get("sample_size") or 0)
        avg = imp_row.get("average_issue_rating")
        low = imp_row.get("low_rating_share_issue")
        if n:
            bits.append(f"Grouped from **{n} complaint reviews**")
        if pd.notna(avg):
            bits.append(f"averaging **{avg:.1f}★**")
        if pd.notna(low):
            bits.append(f"**{low*100:.0f}%** of them rated 1–2★")
    lead = ", ".join(bits)
    terms = [t.strip() for t in str(keywords).split(",") if t.strip()][:6]
    if terms:
        term_str = ", ".join(f"*{t}*" for t in terms)
        tail = f"Users most often mention: {term_str}."
        return f"{lead}. {tail}" if lead else tail
    return f"{lead}." if lead else "Representative complaints are shown below."


# --------------------------------------------------------------------------- #
def render_analyze_control():
    """Sidebar panel to pick which app to analyze. Runs the pipeline on click
    (as a subprocess) — heavy work stays in run_pipeline.py, never at page load."""
    with st.sidebar.expander("🔍 Analyze an app", expanded=False):
        st.caption("Fetch live reviews for any app and rebuild the analysis. "
                   "Needs internet; the run can take a little while.")
        gp = st.text_input("Google Play package id", key="gp",
                            placeholder="com.whatsapp")
        ap_id = st.text_input("App Store app id", key="ap", placeholder="310633997")
        col1, col2 = st.columns(2)
        country = col1.text_input("App Store country", value="us", key="ctry")
        count = col2.number_input("Reviews / source", 200, 20000, 3000, step=100,
                                  key="cnt")
        name = st.text_input("App name (optional)", key="nm", placeholder="WhatsApp")
        st.caption("💡 Trends & forecasts need history — fetch a few thousand "
                   "reviews so they span several weeks. Popular apps: 3000–8000.")
        if st.button("Run analysis", type="primary", key="runbtn"):
            _run_pipeline_for(gp.strip(), ap_id.strip(), country.strip(),
                              name.strip(), int(count))


def _run_pipeline_for(gp, ap_id, country, name, count):
    if not gp and not ap_id:
        st.sidebar.warning("Enter a Google Play package id and/or an App Store id.")
        return
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--count", str(count)]
    if gp:
        cmd += ["--gplay", gp]
    if ap_id:
        cmd += ["--appstore", ap_id, "--country", country or "us"]
    if name:
        cmd += ["--app-name", name]
    with st.spinner("Ingesting reviews and running the full analysis…"):
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if res.returncode == 0:
        st.cache_data.clear()
        st.sidebar.success("Analysis complete — reloading dashboard.")
        st.rerun()
    else:
        st.sidebar.error("Pipeline failed. Check the app ids / internet and retry.")
        st.sidebar.code((res.stderr or res.stdout or "")[-1500:])


# --------------------------------------------------------------------------- #
def main():
    # Sidebar header + app picker are always available (even before any run).
    st.sidebar.title("📡 Bellwether")
    render_analyze_control()
    st.sidebar.divider()

    try:
        mtime = os.path.getmtime(db_path())
    except OSError:
        st.markdown('<div class="app-header">📡 Bellwether</div>', unsafe_allow_html=True)
        st.info("No analysis yet. Use **🔍 Analyze an app** in the sidebar to pick "
                "an app, or run the pipeline from the terminal:")
        st.code("python run_pipeline.py --sample", language="bash")
        return

    try:
        data = get_data(mtime)
    except Exception as e:  # database/read errors
        st.markdown('<div class="app-header">📡 Bellwether</div>', unsafe_allow_html=True)
        st.error(f"Could not read the analytical database: {e}")
        return

    pred, trends = data["issue_prediction"], data["issue_trends"]
    impact, cleaned = data["issue_impact"], data["cleaned_reviews"]

    if cleaned.empty:
        st.markdown('<div class="app-header">📡 Bellwether</div>', unsafe_allow_html=True)
        st.warning("The database has no reviews yet. Use **🔍 Analyze an app** in the "
                   "sidebar, or run `python run_pipeline.py --sample`.")
        return

    ranked = warning_ranked(pred)
    issue_opts = ranked["issue_id"].tolist() if not ranked.empty else []
    labels = (ranked.set_index("issue_id")["issue_label"].to_dict()
              if not ranked.empty else {})

    # ---- sidebar: navigation + focus -------------------------------------- #
    page = st.sidebar.radio("Navigate", PAGES)
    st.sidebar.divider()
    st.sidebar.header("Focus")
    selected = st.sidebar.selectbox(
        "Issue for detail sections", issue_opts,
        format_func=lambda i: f"{labels.get(i, i)} ({i})") if issue_opts else None
    as_of = pd.to_datetime(trends["date"]).max() if not trends.empty else None
    if as_of is not None:
        st.sidebar.caption(f"📅 Data as of **{as_of:%Y-%m-%d}**")

    # ---- header ----------------------------------------------------------- #
    st.markdown(f'<div class="app-header">📡 Bellwether — {page}</div>',
                unsafe_allow_html=True)

    ri = data["review_issues"]
    kw_map = (ri.groupby("issue_id")["issue_keywords"].first().to_dict()
              if not ri.empty and "issue_keywords" in ri.columns else {})

    ctx = {"kpis": compute_kpis(data), "ranked": ranked, "trends": trends,
           "impact": impact, "issue_opts": issue_opts, "labels": labels,
           "selected": selected, "mtime": mtime, "kw_map": kw_map}

    if page == PAGES[0]:
        page_overview(ctx)
    elif page == PAGES[1]:
        page_analytics(ctx)
    else:
        page_detail(ctx)


main()  # Streamlit executes the script top to bottom
