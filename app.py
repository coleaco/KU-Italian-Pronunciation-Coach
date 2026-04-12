# ============================================================
# Italian Pronunciation Coach (MFA-free, deployable)
# Whisper + acoustic heuristics + Claude coaching
# ============================================================

import os
import json
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Literal

import numpy as np
import streamlit as st
import librosa
import soundfile as sf

from faster_whisper import WhisperModel
from anthropic import Anthropic
from streamlit_mic_recorder import mic_recorder


# ============================================================
# ---------------------- Data Contracts ----------------------
# ============================================================

FeatureType = Literal[
    "word_duration",
    "speech_rate",
    "final_vowel",
]

Severity = Literal["low", "medium", "high"]
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

MAX_SECONDS = 60
WHISPER_MODEL = "small"
WHISPER_COMPUTE = "int8"
CLAUDE_MODEL = "claude-sonnet-4-6"


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
    """
    Returns:
        full_text: str
        words: list of dicts with word, start, end
    """
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
                        "word": w.word.strip(),
                        "start": float(w.start or 0.0),
                        "end": float(w.end or 0.0),
                    })

    return " ".join(text_parts).strip(), words


# ============================================================
# ------------- acoustic heuristics (MFA-free) ---------------
# ============================================================

def extract_features(
    audio_path: str,
    words: List[dict],
    level: LearnerLevel,
) -> List[PronunciationEvidence]:

    y, sr = librosa.load(audio_path, sr=None)
    total_duration = librosa.get_duration(y=y, sr=sr)

    evidence: List[PronunciationEvidence] = []

    # ---- Speech rate (global) ----
    if len(words) >= 5:
        rate = len(words) / max(total_duration, 0.1)
        if rate > 3.5:
            evidence.append(
                PronunciationEvidence(
                    word="(overall)",
                    feature="speech_rate",
                    expected="steady pace",
                    observed="rushed",
                    severity="medium",
                    time_range=(0.0, total_duration),
                )
            )

    # ---- Word-level checks ----
    durations = [
        (w["word"], w["end"] - w["start"], w["start"], w["end"])
        for w in words
        if w["end"] > w["start"]
    ]

    if durations:
        avg_duration = np.mean([d[1] for d in durations])

        for word, dur, start, end in durations:
            # Possible geminate proxy: unusually short word
            if dur < 0.6 * avg_duration and len(word) >= 4:
                evidence.append(
                    PronunciationEvidence(
                        word=word,
                        feature="word_duration",
                        expected="more held",
                        observed="compressed",
                        severity="medium",
                        time_range=(start, end),
                    )
                )

            # Final vowel truncation proxy
            tail_start = int(end * sr)
            tail = y[tail_start:tail_start + int(0.08 * sr)]
            if tail.size > 0 and np.mean(np.abs(tail)) < 0.01:
                evidence.append(
                    PronunciationEvidence(
                        word=word,
                        feature="final_vowel",
                        expected="clearly pronounced",
                        observed="cut short",
                        severity="low",
                        time_range=(start, end),
                    )
                )

    return evidence


# ============================================================
# ---------------- Claude coaching output --------------------
# ============================================================

def claude_feedback(payload: ClaudeInputPayload) -> dict:
    client = load_claude()

    system_prompt = """
You are a pronunciation coach for learners of Italian.
You give supportive, practical feedback in English.
You do not diagnose errors or use phonetic terms.
You only explain adjustments learners can try.
You output ONLY valid JSON.
""".strip()

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
- Max focus points:
  Beginner: 2
  Intermediate: 3
  Advanced: 2
- coaching_tip ≤ 20 words
- If evidence is empty, say pronunciation is clear and give no focus points

Learner level: {payload.level}

Pronunciation evidence:
{json.dumps([e.__dict__ for e in payload.evidence], indent=2)}
""".strip()

    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        temperature=0,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return json.loads(resp.content[0].text)


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
            evidence = extract_features(audio_path, words, level)

        payload = ClaudeInputPayload(level=level, evidence=evidence)

        with st.spinner("Generating coaching feedback…"):
            feedback = claude_feedback(payload)

        st.subheader("Coaching feedback")
        st.write(feedback.get("overall_feedback", ""))

        for i, fp in enumerate(feedback.get("focus_points", []), 1):
            st.markdown(
                f"**{i}. {fp['word']}** "
                f"({fp['time_range'][0]:.2f}s–{fp['time_range'][1]:.2f}s)\n\n"
                f"{fp['coaching_tip']}\n\n"
                f"*{fp['example_hint']}*"
            )
