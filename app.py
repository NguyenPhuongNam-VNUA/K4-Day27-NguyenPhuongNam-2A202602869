"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path, replay_events
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupt_payload" not in st.session_state:
    st.session_state.interrupt_payload = None
if "final" not in st.session_state:
    st.session_state.final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HITL PR Review Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🤖 HITL PR Review Agent")
st.caption("Human-in-the-Loop AI Code Review Agent powered by LangGraph & SQLite Audit Trail")


# ─── Helper for fetching recent sessions ───────────────────────────────────
async def get_recent_sessions():
    async with db_conn() as conn:
        async with conn.execute(
            """
            SELECT thread_id,
                   pr_url,
                   MIN(timestamp)        AS started,
                   MAX(timestamp)        AS last_event,
                   MAX(risk_level)       AS worst_risk,
                   COUNT(*)              AS events
              FROM audit_events
             GROUP BY thread_id, pr_url
             ORDER BY MAX(timestamp) DESC
             LIMIT 15
            """
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


# ─── Sidebar — recent sessions & audit replay ──────────────────────────────
with st.sidebar:
    st.header("📋 Recent Sessions")
    try:
        sessions = asyncio.run(get_recent_sessions())
        if sessions:
            for s in sessions:
                tid = s["thread_id"]
                short_id = tid[:8]
                risk = s["worst_risk"]
                risk_emoji = "🟢" if risk == "low" else ("🟡" if risk == "med" else "🔴")
                pr_num = s["pr_url"].rstrip("/").split("/")[-1]
                label = f"{risk_emoji} PR #{pr_num} ({short_id}...)"
                if st.button(label, key=f"session_{tid}", use_container_width=True):
                    st.session_state.thread_id = tid
                    st.session_state.pr_url = s["pr_url"]
                    st.session_state.interrupt_payload = None
                    st.session_state.final = None
                    st.rerun()
        else:
            st.caption("No previous sessions recorded yet.")
    except Exception as e:
        st.caption(f"Could not load sessions: {e}")

    if st.session_state.thread_id:
        st.divider()
        st.subheader("🔍 Active Session Audit")
        st.caption(f"Thread: `{st.session_state.thread_id}`")
        try:
            events = asyncio.run(replay_events(st.session_state.thread_id))
            if events:
                for ev in events:
                    with st.expander(f"{ev['action']} ({ev['decision']})", expanded=False):
                        st.write(f"**Confidence:** {ev['confidence']:.0%}")
                        st.write(f"**Risk:** {ev['risk_level']}")
                        st.write(f"**Time:** {ev['execution_time_ms']} ms")
                        if ev.get("reason"):
                            st.write(f"**Reason:** {ev['reason']}")
            else:
                st.caption("No audit events for this thread yet.")
        except Exception as e:
            st.caption(f"Replay error: {e}")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "Pull Request URL",
        value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    col_demo1, col_demo2, col_submit = st.columns([1, 1, 2])
    with col_demo1:
        load_demo1 = st.form_submit_button("Load PR #1 (Medium Risk)")
    with col_demo2:
        load_demo2 = st.form_submit_button("Load PR #2 (High Risk)")
    with col_submit:
        submitted = st.form_submit_button("🚀 Run Review", type="primary", use_container_width=True)

if load_demo1:
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/1"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    pr_url = st.session_state.pr_url
    submitted = True

if load_demo2:
    st.session_state.pr_url = "https://github.com/VinUni-AI20k/PR-Demo/pull/2"
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None
    pr_url = st.session_state.pr_url
    submitted = True


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.info(f"⚖️ **Approval Requested** — Confidence: **{conf:.0%}** (Human Review Required)")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    st.subheader("Proposed Comments")
    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity'].upper()}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("📄 Diff Preview", expanded=False):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Reviewer Feedback (optional note to author)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("✅ Approve", type="primary", use_container_width=True):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("❌ Reject", use_container_width=True):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("✏️ Edit Review", use_container_width=True):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_card(payload: dict) -> dict | None:
    """< 58% bucket: show risk factors + question form. Return {question: answer} or None."""
    conf = payload["confidence"]
    st.warning(f"⚠️ **Strong Escalation** — Confidence: **{conf:.0%}** (Significant Risks Detected)")
    st.caption(payload["confidence_reasoning"])
    if payload.get("risk_factors"):
        st.error("🚨 **Risk Factors:** " + ", ".join(payload["risk_factors"]))
    st.markdown(payload["summary"])

    with st.form("escalation"):
        st.subheader("❓ Clarifying Questions for Reviewer")
        st.caption("Please provide context to help the agent re-synthesize and calibrate the review:")
        answers: dict[str, str] = {}
        for i, q in enumerate(payload.get("questions", [])):
            answers[q] = st.text_area(
                f"**Question {i+1}:** {q}",
                key=f"q_{i}",
                placeholder="Type your clarification or resolution here...",
            )
        submit_answers = st.form_submit_button("📤 Submit Answers & Re-synthesize", type="primary", use_container_width=True)
        if submit_answers:
            return answers
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}

        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    if not st.session_state.thread_id:
        st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupt_payload = None
    st.session_state.final = None

    with st.spinner("🔍 Fetching PR & Analyzing with AI Reviewer..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    if "__interrupt__" in result:
        st.session_state.interrupt_payload = result["__interrupt__"][0].value
    else:
        st.session_state.final = result

# Render the current interrupt card, if any
payload = st.session_state.interrupt_payload
if payload is not None:
    kind = payload["kind"]
    answer = render_approval_card(payload) if kind == "approval_request" else render_escalation_card(payload)
    if answer is not None:
        with st.spinner("⏳ Resuming review graph with your input..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        if "__interrupt__" in result:
            st.session_state.interrupt_payload = result["__interrupt__"][0].value
        else:
            st.session_state.interrupt_payload = None
            st.session_state.final = result
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"🎉 **{action}** — Review comment posted to [{st.session_state.pr_url}]({st.session_state.pr_url})")
    elif action == "rejected":
        st.warning("🛑 **Rejected** — No review comment posted to GitHub.")
    elif action == "commit_failed":
        st.warning(f"⚠️ **Review finished (action: {action})** — Check GITHUB_TOKEN if write access to GitHub PR is desired.")
    else:
        st.info(f"ℹ️ **Review status:** `{action}`")

    if "analysis" in final:
        a = final["analysis"]
        st.write(f"**Final Confidence:** {a.confidence:.0%}")
        st.write(f"**Summary:** {a.summary}")

    st.caption(
        f"Thread ID: `{st.session_state.thread_id}`  ·  Replay command: "
        f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`"
    )

