# app.py — LLM-only Provider–Patient Communication Simulator (Option A key entry)
# ------------------------------------------------------------------------------
# pip install: streamlit sqlalchemy psycopg2-binary openai pandas
# Run: streamlit run app.py

import os
import uuid
import datetime as dt
from typing import Dict, Any, List

import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

from openai import OpenAI

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="Healthcare Comm Simulator (LLM-only)", page_icon="🩺", layout="centered")

# -------------------------------
# Option A: Prompt for API key at runtime (no files, no Git)
# -------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API key", type="password", help="Your key is not saved to disk.")
    if not OPENAI_API_KEY:
        st.info("Add your OpenAI API key above to continue.", icon="🗝️")
        st.stop()

# Create OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)
MODEL_DEFAULT = "gpt-4o-mini"  # change if you like

# -------------------------------
# Database setup (SQLite by default; Postgres via DB_URL env if you want)
# -------------------------------
DB_URL = os.getenv("DB_URL", "sqlite:///comm_sim.db")
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)          # uuid
    email = Column(String, unique=True, nullable=True)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

class CaseSession(Base):
    __tablename__ = "sessions"
    id = Column(String, primary_key=True)          # uuid
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    case_id = Column(String, nullable=False)
    started_at = Column(DateTime, default=dt.datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)

    user = relationship("User", backref="sessions")
    messages = relationship("Message", backref="session", cascade="all, delete-orphan")
    score = relationship("Score", uselist=False, backref="session", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False)
    role = Column(String, nullable=False)          # "clinician" | "patient"
    content = Column(Text, nullable=False)
    ts = Column(DateTime, default=dt.datetime.utcnow)

class Score(Base):
    __tablename__ = "scores"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), nullable=False, unique=True)
    overall = Column(Integer, default=0)
    subscores = Column(JSON, default={})
    feedback = Column(Text, default="")
    created_at = Column(DateTime, default=dt.datetime.utcnow)

Base.metadata.create_all(bind=engine)

# -------------------------------
# Scenarios (5 special cases)
# -------------------------------
CASES: Dict[str, Dict[str, Any]] = {
    "bad_news_spikes": {
        "title": "Delivering Bad News (SPIKES-aligned)",
        "role": "physician",
        "prompt": "(Patient) Doctor… the tests—what did they show?",
        "must_include": ["ack_emotion", "warn_shot", "plain_lang", "next_steps", "allow_silence", "teach_back"],
        "reveals": {
            "support_person": ["family", "partner", "with you", "support", "anyone here"],
            "prior_suspicion": ["suspicious", "worried already", "thought it might"]
        },
        "info": {
            "support_person": "(Patient) My sister is in the waiting room.",
            "prior_suspicion": "(Patient) I… suspected something was wrong."
        }
    },
    "noncompliance_mi": {
        "title": "Handling Noncompliance (Motivational Interviewing)",
        "role": "physician",
        "prompt": "(Patient) I’m not taking that medication. I just don’t want to.",
        "must_include": ["elicit_reasons", "reflective_listening", "affirm", "collab_plan", "teach_back"],
        "reveals": {
            "side_effects": ["side effect", "nausea", "dizzy", "headache", "sleep"],
            "cost_barrier": ["cost", "price", "afford", "insurance", "copay"]
        },
        "info": {
            "side_effects": "(Patient) Last time it made me nauseous.",
            "cost_barrier": "(Patient) It’s expensive. I can’t afford it every month."
        }
    },
    "sensitive_topics": {
        "title": "Discussing Sensitive Topics (sexual/mental health/addiction)",
        "role": "physician",
        "prompt": "(Patient) …This is hard to talk about.",
        "must_include": ["normalize", "confidentiality", "nonjudgment", "plain_lang", "resources", "teach_back"],
        "reveals": {
            "sexual_health": ["sex", "partner", "condom", "sti", "std", "prep"],
            "mental_health": ["depress", "anx", "panic", "sleep", "hopeless"],
            "substance_use": ["drink", "alcohol", "opioid", "fentanyl", "heroin", "cocaine", "pill"]
        },
        "info": {
            "sexual_health": "(Patient) I’ve had two partners and didn’t always use condoms.",
            "mental_health": "(Patient) I haven’t been sleeping and feel on edge all day.",
            "substance_use": "(Patient) I’ve been taking pills from a friend to unwind."
        }
    },
    "cultural_barriers": {
        "title": "Cultural/Language Barriers",
        "role": "physician",
        "prompt": "(Patient) Sorry, my English not perfect…",
        "must_include": ["interpreter_offer", "plain_lang", "ask_preferences", "respect_beliefs", "teach_back"],
        "reveals": {
            "interpreter_need": ["interpreter", "language", "spanish", "mandarin", "arabic", "translator"],
            "beliefs": ["traditional", "herbal", "religion", "fasting", "prayer", "remedy"]
        },
        "info": {
            "interpreter_need": "(Patient) Interpreter would help me understand.",
            "beliefs": "(Patient) I prefer to try herbal remedies first."
        }
    },
    "time_pressured": {
        "title": "Time-Pressured Consult (Empathy still matters)",
        "role": "dentist",
        "prompt": "(Patient) I only have 5 minutes. My tooth is killing me—what can you do now?",
        "must_include": ["set_expectations", "ack_emotion", "prioritize", "safety_net", "plain_lang", "teach_back"],
        "reveals": {
            "fear_needles": ["needle", "shot", "injection", "scared", "fear"],
            "cost": ["cost", "price", "insurance", "afford", "bill"],
        },
        "info": {
            "fear_needles": "(Patient) Needles make me panic.",
            "cost": "(Patient) I’m worried about the cost today."
        }
    }
}

# -------------------------------
# Detectors for scoring
# -------------------------------
def contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n in t for n in needles)

def detect_empathy(text: str) -> bool:
    cues = ["i hear", "i understand", "that sounds", "it’s okay to feel", "i'm sorry", "that must be hard", "thank you for sharing"]
    return contains_any(text, cues)

def detect_plain_language(text: str) -> bool:
    jargon = {"etiology","prognosis","iatrogenic","adherence","endodontic","analgesic","occlusal","contraindicated"}
    words = [w.strip(".,?!;:").lower() for w in text.split()]
    return not any(w in jargon for w in words)

def detect_open_question(text: str) -> bool:
    starters = ["how", "what", "tell me", "walk me", "help me understand", "can you share"]
    return contains_any(text, starters) and text.strip().endswith("?")

def detect_teachback(text: str) -> bool:
    cues = ["in your own words", "to make sure i explained", "can you tell me back", "how will you take", "what will you do when"]
    return contains_any(text, cues)

def detect_warn_shot(text: str) -> bool:
    cues = ["i’m afraid", "the results are serious", "some difficult news", "i wish i had better news"]
    return contains_any(text, cues)

def detect_allow_silence(text: str) -> bool:
    cues = ["i’m here with you", "take your time", "we can pause", "it’s okay to take a moment"]
    return contains_any(text, cues)

def detect_affirmation(text: str) -> bool:
    cues = ["that makes sense", "you’ve done a lot", "thanks for being honest", "i appreciate you sharing"]
    return contains_any(text, cues)

def detect_interpreter_offer(text: str) -> bool:
    cues = ["interpreter", "translator", "language support"]
    return contains_any(text, cues)

def detect_ask_preferences(text: str) -> bool:
    cues = ["what matters most", "preferences", "how do you prefer", "what would you like"]
    return contains_any(text, cues)

def detect_resources(text: str) -> bool:
    cues = ["resources", "counseling", "support group", "hotline", "clinic", "referral"]
    return contains_any(text, cues)

def detect_set_expectations(text: str) -> bool:
    cues = ["we have a few minutes", "today we can", "right now we’ll", "first we’ll"]
    return contains_any(text, cues)

def detect_prioritize(text: str) -> bool:
    cues = ["let’s focus on", "top priority", "first step", "most important now"]
    return contains_any(text, cues)

def detect_safety_net(text: str) -> bool:
    cues = ["if things worsen", "return precautions", "call if", "24/7", "on-call"]
    return contains_any(text, cues)

def detect_collab_plan(text: str) -> bool:
    cues = ["let’s decide together", "shared decision", "we can choose", "options are"]
    return contains_any(text, cues)

def detect_reflective_listening(text: str) -> bool:
    cues = ["what i’m hearing is", "it sounds like you", "you’re saying"]
    return contains_any(text, cues)

def detect_normalize(text: str) -> bool:
    cues = ["many people feel this way", "it’s common", "you’re not alone"]
    return contains_any(text, cues)

def detect_confidentiality(text: str) -> bool:
    cues = ["confidential", "kept private", "only shared with your permission"]
    return contains_any(text, cues)

def reveal_keys_for_case(case_id: str, clinician_text: str) -> List[str]:
    t = clinician_text.lower()
    reveals = []
    for key, triggers in CASES[case_id]["reveals"].items():
        if any(trigger in t for trigger in triggers):
            reveals.append(key)
    return reveals

# -------------------------------
# LLM patient reply
# -------------------------------
def llm_patient_reply(case_id: str, transcript: List[Dict[str, str]], revealed: set, model: str) -> str:
    case = CASES[case_id]
    last_user = transcript[-1]["content"] if transcript and transcript[-1]["role"] == "user" else ""
    for k in reveal_keys_for_case(case_id, last_user):
        revealed.add(k)

    hidden_dump = {k: case["info"][k] for k in case["reveals"].keys()}
    system_instructions = f"""
You are a realistic PATIENT in a clinician communication training simulator. Stay in character.
CASE: {case['title']}.

Rules:
- Reply in 1–3 sentences; keep tone emotionally congruent.
- Do NOT reveal hidden facts unless the clinician probes that area.
- If clinician uses jargon, ask for simpler words.
- Accept teach-back prompts; engage in shared decision-making.

Hidden facts (reveal only if elicited): {hidden_dump}
Currently eligible to reveal: {list(revealed)}
Must-have clinician communication targets: {case['must_include']}
Respond ONLY as the patient.
""".strip()

    messages = [{"role": "system", "content": system_instructions}]
    messages.extend(transcript[-12:])  # compact context

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.7,
    )
    return resp.choices[0].message.content.strip()

# -------------------------------
# Scoring
# -------------------------------
def score_session(db, session_id: str, case_id: str) -> Dict[str, Any]:
    msgs: List[Message] = (
        db.query(Message)
        .filter(Message.session_id == session_id, Message.role == "clinician")
        .order_by(Message.ts)
        .all()
    )
    if not msgs:
        return {"overall": 0, "subscores": {}, "feedback": "No clinician messages."}

    empathy_hits = sum(detect_empathy(m.content) for m in msgs)
    open_q_hits = sum(detect_open_question(m.content) for m in msgs)
    plain_hits = sum(detect_plain_language(m.content) for m in msgs)
    teach_hits = sum(detect_teachback(m.content) for m in msgs)

    targets = CASES[case_id]["must_include"]
    target_hits = {
        "ack_emotion": any(detect_empathy(m.content) for m in msgs),
        "plain_lang": any(detect_plain_language(m.content) for m in msgs),
        "teach_back": any(detect_teachback(m.content) for m in msgs),
        "warn_shot": any(detect_warn_shot(m.content) for m in msgs),
        "allow_silence": any(detect_allow_silence(m.content) for m in msgs),
        "elicit_reasons": any(detect_open_question(m.content) for m in msgs),
        "reflective_listening": any(detect_reflective_listening(m.content) for m in msgs),
        "affirm": any(detect_affirmation(m.content) for m in msgs),
        "collab_plan": any(detect_collab_plan(m.content) for m in msgs),
        "interpreter_offer": any(detect_interpreter_offer(m.content) for m in msgs),
        "ask_preferences": any(detect_ask_preferences(m.content) for m in msgs),
        "resources": any(detect_resources(m.content) for m in msgs),
        "set_expectations": any(detect_set_expectations(m.content) for m in msgs),
        "prioritize": any(detect_prioritize(m.content) for m in msgs),
        "safety_net": any(detect_safety_net(m.content) for m in msgs),
        "next_steps": any("next step" in m.content.lower() or "plan" in m.content.lower() for m in msgs),
    }

    subs = {
        "Empathy": min(100, empathy_hits * 25),
        "Open Questions": min(100, open_q_hits * 20),
        "Plain Language": min(100, int(100 * (plain_hits / max(1, len(msgs))))),
        "Teach-Back": 100 if target_hits["teach_back"] else 0,
    }
    must = [t for t in targets if t in target_hits]
    coverage = sum(target_hits[t] for t in must) / max(1, len(must))
    subs["Case Targets"] = int(coverage * 100)

    overall = round(
        0.30 * subs["Empathy"] +
        0.20 * subs["Open Questions"] +
        0.20 * subs["Plain Language"] +
        0.15 * subs["Teach-Back"] +
        0.15 * subs["Case Targets"]
    )

    missed = [t.replace("_", " ") for t in must if not target_hits.get(t, False)]
    fb_lines = ["Strengths:"]
    if subs["Empathy"] >= 50: fb_lines.append("- You acknowledged feelings.")
    if subs["Open Questions"] >= 40: fb_lines.append("- You used open questions to explore.")
    if subs["Plain Language"] >= 70: fb_lines.append("- Your language was mostly easy to follow.")
    if subs["Teach-Back"] == 100: fb_lines.append("- You used teach-back to confirm understanding.")
    fb_lines.append("\nFocus Next Time:")
    if missed:
        for m in missed: fb_lines.append(f"- Address: {m}")
    else:
        fb_lines.append("- Solid coverage of required elements.")
    fb_lines += [
        "\nTry saying:",
        '• "It sounds like this has been really tough. I’m here with you."',
        '• "In your own words, how will you take this when you get home?"',
        '• "We can decide together—here are two options and why each might fit you."'
    ]
    return {"overall": overall, "subscores": subs, "feedback": "\n".join(fb_lines)}

# -------------------------------
# UI & Conversation Loop
# -------------------------------
st.title("🩺 Provider–Patient Communication Simulator (LLM)")
st.caption("Text-based • OpenAI patient • SQL-backed transcripts & scores")

# Bootstrap anon user
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    with SessionLocal() as db:
        if not db.query(User).filter(User.id == st.session_state.user_id).first():
            db.add(User(id=st.session_state.user_id, email=None))
            db.commit()

# Scenario picker + model picker
left, right = st.columns([2, 1])
with left:
    case_ids = list(CASES.keys())
    case_titles = [CASES[c]["title"] for c in case_ids]
    sel_idx = st.selectbox("Choose a scenario:", options=list(range(len(case_ids))),
                           format_func=lambda i: case_titles[i], index=0)
    active_case_id = case_ids[sel_idx]
    active_case = CASES[active_case_id]
with right:
    model_name = st.selectbox("OpenAI model", [MODEL_DEFAULT, "gpt-4o"], index=0)

# Start/reset session per case switch
if "active_session_id" not in st.session_state or st.session_state.get("active_case_id") != active_case_id:
    with SessionLocal() as db:
        sid = str(uuid.uuid4())
        db.add(CaseSession(id=sid, user_id=st.session_state.user_id, case_id=active_case_id))
        db.commit()
        st.session_state.active_session_id = sid
        st.session_state.active_case_id = active_case_id
        st.session_state.revealed = set()
        # Seed patient opener
        db.add(Message(session_id=sid, role="patient", content=active_case["prompt"]))
        db.commit()
    st.rerun()

st.info(f"Case: **{active_case['title']}**  •  Session: `{st.session_state.active_session_id[:8]}`")

# Display transcript
with SessionLocal() as db:
    existing = (
        db.query(Message)
        .filter(Message.session_id == st.session_state.active_session_id)
        .order_by(Message.ts)
        .all()
    )
for m in existing:
    with st.chat_message("assistant" if m.role == "patient" else "user"):
        st.markdown(m.content)

def current_transcript_for_llm(db) -> List[Dict[str, str]]:
    msgs = (
        db.query(Message)
        .filter(Message.session_id == st.session_state.active_session_id)
        .order_by(Message.ts)
        .all()
    )
    conv = []
    # Convert to OpenAI roles: user=clinician, assistant=patient
    for m in msgs:
        conv.append({"role": "assistant" if m.role == "patient" else "user", "content": m.content})
    return conv

# Chat input
prompt = st.chat_input("Type your response to the patient…")
if prompt:
    # Store clinician message
    with SessionLocal() as db:
        db.add(Message(session_id=st.session_state.active_session_id, role="clinician", content=prompt))
        db.commit()
    with st.chat_message("user"): st.markdown(prompt)

    # Build transcript and get LLM patient reply
    with SessionLocal() as db:
        transcript = current_transcript_for_llm(db)
    reply = llm_patient_reply(st.session_state.active_case_id, transcript, st.session_state.revealed, model=model_name)

    # Store patient reply
    with SessionLocal() as db:
        db.add(Message(session_id=st.session_state.active_session_id, role="patient", content=reply))
        db.commit()
    with st.chat_message("assistant"): st.markdown(reply)

# Controls
c1, c2, c3 = st.columns(3)
with c1:
    if st.button("End Session & Debrief", type="primary", use_container_width=True):
        with SessionLocal() as db:
            s = db.query(CaseSession).get(st.session_state.active_session_id)
            s.ended_at = dt.datetime.utcnow()
            db.commit()

            result = score_session(db, s.id, s.case_id)
            sc = db.query(Score).filter(Score.session_id == s.id).first()
            if not sc:
                db.add(Score(session_id=s.id, overall=result["overall"], subscores=result["subscores"], feedback=result["feedback"]))
            else:
                sc.overall = result["overall"]; sc.subscores = result["subscores"]; sc.feedback = result["feedback"]
            db.commit()
        st.success("Session saved & scored.")

with c2:
    if st.button("Start Fresh (same case)", use_container_width=True):
        with SessionLocal() as db:
            sid = str(uuid.uuid4())
            db.add(CaseSession(id=sid, user_id=st.session_state.user_id, case_id=st.session_state.active_case_id))
            db.commit()
            st.session_state.active_session_id = sid
            st.session_state.revealed = set()
            db.add(Message(session_id=sid, role="patient", content=active_case["prompt"]))
            db.commit()
        st.rerun()

with c3:
    if st.button("Switch Case (reset)", use_container_width=True):
        st.session_state.pop("active_session_id", None)
        st.rerun()

# Debrief (if exists)
with SessionLocal() as db:
    sc = db.query(Score).filter(Score.session_id == st.session_state.active_session_id).first()
    if sc:
        st.subheader("Debrief")
        colA, colB = st.columns(2)
        with colA:
            st.metric("Overall", sc.overall)
            st.write("**Subscores**")
            st.json(sc.subscores)
        with colB:
            st.write("**Feedback**")
            st.text(sc.feedback)

st.caption("Enter your API key above each run. SQLite by default. Set DB_URL env for Postgres if needed.")

