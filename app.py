# ============================================================
# Italian Pronunciation Coach — Segmental-Focused (MFA-free)
# Geminate consonants + stress clarity
# ============================================================

import os
import json
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Literal

import numpy as np
import streamlit as st
import librosa

from faster_whisper import WhisperModel
from anthropic import Anthropic
from streamlit_mic_recorder import mic_recorder


# ============================================================
# ---------------------- Data Contracts ----------------------
# ============================================================

FeatureType = Literal[
    "geminate_weight",
    "stress_clarity",
]

Severity = Literal["low", "medium"]
LearnerLevel = Literal["Beginner", "Intermediate", "Advanced"]


@dataclass
class PronunciationEvidence:
    word: str
    feature: FeatureType
    expected: str
    observed: str
    severity: Severity
    time_range: Tuple[float, float]


@dataclass
class ClaudeInputPayload:
    level: LearnerLevel
    evidence: List[PronunciationEvidence]


# ============================================================
# ---------------------- Configuration -----------------------
# ============================================================

WHISPER_MODEL = "small"
WHISPER_COMPUTE = "int8"
CLAUDE_MODEL = "claude-sonnet-4-6"

# High-frequency Italian geminate-bearing words (extendable)
GEMINATE_WORDS = {
    # Nouns
    "anno", "anni",
    "notte", "notti",
    "attimo", "attimi",
    "penna", "penne",
    "palla", "palle",
    "tetto", "tetti",
    "gonna", "gonne",
    "nonno", "nonni",
    "mamma", "mamme",

    # Adjectives
    "bello", "bella", "belli", "belle",
    "tutto", "tutta", "tutti", "tutte",
    "cattivo", "cattiva", "cattivi", "cattive",
    "piccolo", "piccola", "piccoli", "piccole",

    # Participles
    "fatto", "fatta", "fatti", "fatte",
    "scritto", "scritta", "scritti", "scritte",

    # Adverbs
    "sotto",
    "adesso",
    "appena",
}


# ============================================================
# ---------------------- Utilities ---------------------------
# ============================================================

@st.cache_resource(show_spinner=False)
def load_whisper():
    return WhisperModel(
        WHISPER_MODEL,
        device="cpu",
        compute_type=WHISPER_COMPUTE,
    )


@st.cache_resource(show_spinner=False)
def load_claude():
    api_key = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing")
    return Anthropic(api_key=api_key)


def save_audio(data: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with open(tmp.name, "wb") as f:
        f.write(data)
    return tmp.name


# ============================================================
# ------------------- Whisper Transcription ------------------
# ============================================================

def transcribe_with_words(audio_path: str):
    model = load_whisper()

    segments, _ = model.transcribe(
        audio_path,
        language="it",
        task="transcribe",
        word_timestamps=True,
        vad_filter=True,
        condition_on_previous_text=False,
    )

    words = []
    text_parts = []

    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())
        if getattr(seg, "words", None):
            for w in seg.words:
                if w.word.strip():
                    words.append({
                        "word": w.word.strip().lower(),
                        "start": float(w.start or 0.0),
                        "end": float(w.end or 0.0),
                    })

    return " ".join(text_parts).strip(), words


# ============================================================
# ------------------ Revised Heuristics ----------------------
# ============================================================

def extract_segmental_features(
    audio_path: str,
    words: List[dict],
) -> List[PronunciationEvidence]:
    y, sr = librosa.load(audio_path, sr=None)
    evidence: List[PronunciationEvidence] = []

    # ---------- GEMINATE CONSONANT HEURISTIC ----------
    for i in range(1, len(words) - 1):
        w = words[i]
        if w["word"] not in GEMINATE_WORDS:
            continue

        prev_w = words[i - 1]
        next_w = words[i + 1]

        dur = w["end"] - w["start"]
        prev_dur = prev_w["end"] - prev_w["start"]
        next_dur = next_w["end"] - next_w["start"]

        if prev_dur <= 0 or next_dur <= 0:
            continue

        local_mean = (prev_dur + next_dur) / 2
        relative_weight = dur / local_mean if local_mean > 0 else 1.0

        if relative_weight <= 1.0:
            evidence.append(
                PronunciationEvidence(
                    word=w["word"],
                    feature="geminate_weight",
                    expected="more articulation through the middle",
                    observed="too quick",
                    severity="medium",
                    time_range=(w["start"], w["end"]),
                )
            )
            return evidence  # only one item per utterance

    # ---------- STRESS CLARITY HEURISTIC ----------
    for w in words:
        dur = w["end"] - w["start"]
        if dur < 0.60 or w["word"] in GEMINATE_WORDS:
            continue

        start_idx = int(w["start"] * sr)
        end_idx = int(w["end"] * sr)
        segment = y[start_idx:end_idx]

        if len(segment) < sr * 0.4:
            continue

        slices = np.array_split(segment, 3)
        energies = [np.mean(np.abs(s)) for s in slices if len(s) > 0]

        if len(energies) < 3:
            continue

        dominance = max(energies) / (np.mean(energies) + 1e-6)

        if dominance < 1.4:
            evidence.append(
                PronunciationEvidence(
                    word=w["word"],
                    feature="stress_clarity",
                    expected="one syllable to stand out",
                    observed="flat",
                    severity="low",
                    time_range=(w["start"], w["end"]),
                )
            )
            return evidence

    return evidence


# ============================================================
# ---------- Claude coaching (bullet‑proof JSON) -------------
# ============================================================

def safe_parse_json(raw_text: str) -> dict:
    if not raw_text or not raw_text.strip():
        raise ValueError("Empty Claude response")

    raw = raw_text.strip()
    start = raw.find("{")
    end = raw.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON found in Claude output")

    return json.loads(raw[start:end + 1])


def claude_feedback(payload: ClaudeInputPayload) -> dict:
    client = load_claude()

    system_prompt = (
        "You are a pronunciation coach for learners of Italian. "
        "You give supportive, practical feedback in English. "
        "You NEVER diagnose errors or use phonetic symbols. "
        "You ONLY output valid JSON."
    )

    user_prompt = f"""
Return ONLY compact JSON:

{{
  "overall_feedback": "<1–2 sentence summary>",
  "focus_points": [
    {{
      "word": "<word>",
      "time_range": [<start>, <end>],
      "coaching_tip": "<adjustment suggestion>",
      "example_hint": "<short cue>"
    }}
  ]
}}

Rules:
- English only
- Max focus points: 1
- coaching_tip ≤ 20 words
- If evidence is empty, say pronunciation is clear and give no focus points

Learner level: {payload.level}

Pronunciation evidence:
{json.dumps([e.__dict__ for e in payload.evidence], indent=2)}
""".strip()

    for attempt in range(2):
        try:
            resp = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=500,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            if not resp.content or not hasattr(resp.content[0], "text"):
                raise ValueError("Claude returned no text")

            return safe_parse_json(resp.content[0].text)

        except Exception:
            if attempt == 1:
                raise


# ============================================================
# --------------------------- UI ------------------------------
# ============================================================

st.set_page_config(
    page_title="Italian Pronunciation Coach",
    page_icon="🗣️",
    layout="wide",
)

st.title("🗣️ Italian Pronunciation Coach")

level: LearnerLevel = st.radio(
    "Choose your pronunciation focus",
    ["Beginner", "Intermediate", "Advanced"],
)

rec = mic_recorder(
    start_prompt="🎙️ Start recording",
    stop_prompt="⏹️ Stop recording",
    format="wav",
)

audio_path = None
if rec and rec.get("bytes"):
    audio_path = save_audio(rec["bytes"])
    st.audio(rec["bytes"], format="audio/wav")

if audio_path and st.button("🧠 Analyze pronunciation"):
    with st.spinner("Transcribing…"):
        text, words = transcribe_with_words(audio_path)

    st.subheader("Transcription")
    st.write(text or "*(No speech detected)*")

    if words:
        with st.spinner("Analyzing pronunciation…"):
            evidence = extract_segmental_features(audio_path, words)

        payload = ClaudeInputPayload(level=level, evidence=evidence)

        with st.spinner("Generating coaching feedback…"):
            try:
                feedback = claude_feedback(payload)
            except Exception:
                st.warning(
                    "We couldn’t generate feedback this time. "
                    "Please click Analyze again."
                )
                st.stop()

        st.subheader("Coaching feedback")
        st.write(feedback.get("overall_feedback", ""))

        for i, fp in enumerate(feedback.get("focus_points", []), 1):
            st.markdown(
                f"**{i}. {fp['word']}** "
                f"({fp['time_range'].2f}s–{fp['time_range'].2f}s)\n\n"
                f"{fp['coaching_tip']}\n\n"
                f"*{fp['example_hint']}*"
            )

# ============================================================
# ---------------------- Footer block -------------------------
# ============================================================

st.markdown("---")

st.markdown(
    """
**How to use this tool**
This tool listens for a few key features of Italian pronunciation that make speech sound clearer and, well, more Italian! It focuses on **double consonants** (like *tt, ll, nn*), checking whether words sound heavy enough in the middle, and on **stress clarity** in longer words, making sure one syllable stands out instead of everything sounding flat. When it gives feedback, it usually highlights **one small adjustment** you can try right away. If there’s no feedback, that’s a good sign—your pronunciation was clear enough for this level. Features like open/closed vowels or the rolled r are not assessed in this version because these sounds can't be evaluated reliably without advanced tools.
"""
)
