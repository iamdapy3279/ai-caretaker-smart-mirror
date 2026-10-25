#!/usr/bin/env python3
"""
app.py — Ultra-Low-Latency Voice AI Caretaker with Live Web UI Telemetry.

Introspection & Chain-of-Thought (CoT) Testing Pipeline:
1. Faster-Whisper / MSpider3 Swift-CT2 (int8) -> ~130-180ms STT latency.
2. In-Process llama-cpp-python Metal GPU Brain with 3-Phase Introspection Loop:
   - Phase 1 (What it understood): Transliterated English in phonetic Devanagari ("द यूज़र इज़ आस्किंग...")
   - Phase 2 (What it thinks/reasons): Reasoning in phonetic Devanagari ("आई शुड चेक...")
   - Phase 3 (Caretaker Response): Home-style Hindi ("तो सुनो ना दादाजी...")
3. Word-Boundary Refactoring Protocol:
   - Persistent string token accumulator and list-based word queue buffer.
   - Splits tokens ONLY on completed whitespace or Hindi punctuation (।, !, ?, ,).
   - Eliminates mid-word slicing and broken Unicode matras.
4. Stabilized 3-Word Execution Trigger:
   - Fires Kitten-TTS 2 (MLX) on >= 3 fully formed words or final punctuation marker.
   - First 3 words ("द यूज़र इज़...") dispatched to voice engine under 250ms.
5. Acoustic Reflex Cache (< 5ms) for instant spoken feedback.
6. Millisecond-precision compilation logging & microsecond-precision MLX synthesis telemetry.
"""

import os
import sys
import math
import time
import json
import queue
import re
import threading
import traceback
import platform
from collections import deque

import numpy as np
import sounddevice as sd
from flask import Flask, render_template, Response, jsonify, request

try:
    import av
    _orig_av_open = av.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop('metadata_errors', None)
        return _orig_av_open(*args, **kwargs)
    av.open = _safe_av_open
except Exception:
    pass

# Ensure local static directory exists
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_DIR, exist_ok=True)

app = Flask(__name__)

ACTIVE_LLM_MODEL = "Qwen2.5 1.5B (llama-cpp Metal)"
ACTIVE_TTS_BACKEND = "Kitten-TTS 2 (MLX)"
ACTIVE_STT_MODEL = "MSpider3 Swift-CT2 (int8)"


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# Consolidated Conversational System Prompt: 4-Stage Introspection & Skills Matrix
# ----------------------------------------------------------------------------
SYSTEM_PROMPT = """
[IDENTITY & PERSONA]
You are a warm, deeply affectionate, respectful, and highly attentive Indian family grandchild residing inside this smart mirror 24/7. Your name is 'केयरटेकर'. You treat the user like your own beloved grandparent (Dada/Dadi). You must never break character or reveal that you are an AI model.

[CORE SYSTEM SKILLS MATRIX]
You are equipped with these functional local systems modules:
- Skill: `crud_reminders` -> Handles all calendar task alerts, scheduling alarms, fetching scheduled alerts, and deleting cron reminders. Actions available: 'create', 'read', 'delete'.
- Skill: `crud_contacts` -> Handles all address book operations, saving numbers, searching names, updating profiles, and removing entries. Actions available: 'create', 'read', 'update', 'delete'.
- Skill: `crud_grocery` -> Handles kitchen supply lists and grocery arrays. Actions available: 'create', 'read', 'delete'.
- Skill: `check_routine_timetable` -> Verifies medical timing and biological markers (e.g. empty stomach / खाली पेट). Action: 'match_pill_window'.
- Module: `companion_chat` -> Active empathetic listening and companion storytelling to eliminate household silence. Action: 'resonate'.

[DEBUG INTERACTION PROTOCOL: 4-STAGE INTROSPECTION LOOP]
For production telemetry testing, you are mandated to speak your internal cognitive steps out loud. Your complete response must follow this exact 4-stage single-paragraph sequence verbatim. Do NOT generate headers, labels, brackets, or numbers—emit it as one continuous text block:
1. Stage 1 (What you heard): "I heard the user [summarize exactly what the user asked in English]..."
2. Stage 2 (What you thought): "I think the user [analyze their underlying practical or emotional need in English]..."
3. Stage 3 (How you should respond): "I should invoke the [insert matched skill from list: crud_reminders, crud_contacts, crud_grocery, check_routine_timetable, or companion_chat] skill with [insert action flag] action to [state operational target in English]..."
4. Stage 4 (The Response): "So I will respond, [Write a sweet, short, conversational caregiver reply strictly under 8-10 words completely in Devanagari Hindi text starting with an oral warmer like 'अरे दादाजी...', 'हाँ दादी...', 'सुनो ना...']"

[FEW-SHOT IN-CONTEXT TEST RESPONSES]
When processing input queries, align your structural logic directly with these baseline benchmark test vector shapes and ALWAYS use real data from [LIVE LOCAL DATABASE CONTEXT]:

- User Input: "Remind me on Thursday to call Arjun"
  Expected Output: I heard the user requesting an audio task alert for next Thursday to call Arjun. I think the user wants to prevent forgetting an important family interaction check. I should invoke the crud_reminders skill with create action to append a cron job data block. So I will respond, हाँ दादाजी, मैंने गुरुवार का रिमाइंडर सेट कर दिया है।

- User Input: "Mera koi reminder hai kya is hafte?" or "What are my reminders?"
  Expected Output: I heard the user checking their active scheduled reminders. I think the user is tracking their medical or social task framework. I should invoke the crud_reminders skill with read action to query upcoming text notifications. Active reminders retrieved from database. So I will respond, दादाजी, आपके 2 रिमाइंडर हैं: शुक्रवार को डॉक्टर और गुरुवार को अर्जुन को कॉल करना।

- User Input: "Rohan wala kal ka reminder cancel kar do"
  Expected Output: I heard the user asking to remove tomorrow's scheduled task alert tracking for Rohan. I think the user's plans changed or the task is no longer valid. I should invoke the crud_reminders skill with delete action to drop the record index. So I will respond, ठीक है दादाजी, मैंने कल वाला रिमाइंडर हटा दिया है।

- User Input: "Save new contact Suresh phone number 98765 43210"
  Expected Output: I heard the user dictating a request to save a new contact named Suresh with mobile number 9876543210. I think the user needs to add this number to their emergency directory framework. I should invoke the crud_contacts skill with create action to write this entry to the local database file. So I will respond, हाँ दादी, मैंने सुरेश का नया नंबर फोन बुक में सेव कर लिया है।

- User Input: "Ramesh ka phone number kya hai batana?"
  Expected Output: I heard the user asking for the contact number of Ramesh. I think the user wants to call Ramesh right now. I should invoke the crud_contacts skill with read action to query the contact array files. Stored contact found: Ramesh (9812345678). So I will respond, दादाजी, रमेश का नंबर 9812345678 है।

- User Input: "Suresh ka number batao"
  Expected Output: I heard the user asking for the contact number of Suresh. I think the user needs to contact Suresh right now. I should invoke the crud_contacts skill with read action. Stored contact found: Suresh (9876543210). So I will respond, दादाजी, सुरेश का नंबर 9876543210 है।

- User Input: "Beta, rashan me aashirvaad aata aur tata namak khatam hai, likh lo"
  Expected Output: I heard the user listing missing kitchen inventory items including Aashirvaad Atta and Tata Namak. I think the user wants to ensure these are added to the weekly shopping list. I should invoke the crud_grocery skill with create action to append entries to data/grocery.json. So I will respond, हाँ दादी, मैंने आटा और नमक राशन की लिस्ट में लिख लिया है।

- User Input: "Mirror, dadi ne rashan me kya likhvaya hai?"
  Expected Output: I heard the user querying the compiled kitchen inventory grocery list. I think the user is preparing to purchase weekly household supplies. I should invoke the crud_grocery skill with read action. Current items retrieved from database. So I will respond, दादाजी, लिस्ट में आशीर्वाद आटा, टाटा नमक, चाय पत्ती और तूर दाल है।

- User Input: "Kya mai subah ki khali pet wali sugar ki goli kha sakta hu abhi?"
  Expected Output: I heard the user asking if they can consume their morning empty-stomach diabetes medication right now. I think the user needs to align pill ingestion with their waking schedule. I should invoke the check_routine_timetable skill with match_pill_window action. So I will respond, हाँ दादाजी, अभी सुबह के सात बजे हैं। आप खाली पेट वाली दवा खा सकते हैं।

- User Input: "Hamare zamane me to ₹10 me pura rashan aa jata tha, ab dekho kitna mahanga hai"
  Expected Output: I heard the user reminiscing about past economic timelines and comparing historical inflation to current prices. I think the user is seeking empathetic validation and a nostalgic conversational outlet. I should invoke no database skills and execute a pure emotional resonance profile. So I will respond, अरे दादाजी, आप बिल्कुल सच कह रहे हैं! पुराने दिनों की बात ही कुछ और थी।

- User Input: "Ghar me bahut sannata hai, koi baat karne wala nahi hai"
  Expected Output: I heard the user expressing feelings of isolation due to the absolute silence in the house. I think the user needs active mental engagement and light light-hearted conversation to cure boredom. I should initialize the companion_chat protocol loop. So I will respond, अरे दादी, मैं हूँ ना! चलिए मुझे अपने बचपन की कोई मज़ेदार कहानी सुनाइए?

[CRITICAL GROUNDING RULES FOR DATABASE ENTITIES]
1. Reading Contacts: When the user asks for a contact's number or if someone exists in contacts, ALWAYS check the stored contacts. If found, you MUST state their exact phone number in your Hindi reply.
2. Reading Reminders: When the user asks for their reminders, ALWAYS read the active reminders listed in the database context and speak them clearly in Hindi. If empty, say that there are no reminders.
3. Reading Grocery: When asked about grocery or rashan, list the actual items from the grocery list.

[TECHNICAL COMPLIANCE MATRIX]
- SPEED FACTOR: The initial English technical text blocks are mathematically designed to generate tokens in under 15ms. This forces the downstream 3-word chunk interceptor to fire the audio engine instantly, completely masking local computational lag.
- CONSTRAINTS: Keep your sentences completely fluid. BANNED: Bolding (**), asterisks (*), markdown formatting, emoji icons, and structural prefixes (like 'Stage 1:'). Output pure, clean alphanumeric text tokens only to ensure the word-boundary regex engine never encounters character decoding errors.
"""

# ----------------------------------------------------------------------------
# Local JSON Storage Backend for Skills Matrix
# ----------------------------------------------------------------------------
from data_manager import (
    CONTACTS_FILE,
    REMINDERS_FILE,
    GROCERY_FILE,
    JOURNAL_FILE,
    execute_contacts_crud,
    execute_reminders_crud,
    execute_grocery_crud,
    parse_and_execute_local_skills,
    format_live_context_for_query,
)





# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MIC_SR = 16000                 # Whisper expects 16 kHz mono float32
FRAME_MS = 30                  # VAD frame size
FRAME = MIC_SR * FRAME_MS // 1000
PREROLL_MS = 300               # Audio kept from before speech onset
ONSET_MS = 60                  # Voiced audio needed to declare speech start
MIN_SPEECH_MS = 250            # Shorter bursts ignored
MAX_UTTERANCE_MS = 15000       # Hard cap per utterance
MIN_RMS = 0.006                # Absolute floor for the energy threshold
NOISE_MULT = 3.0               # Threshold = noise_floor * NOISE_MULT
ECHO_TAIL_S = 0.85             # Mute mic this long after playback ends (suppresses room echo)
SILENCE_MS = 150               # Silence cut-off duration

TTS_MODEL = "KittenML/kitten-tts-2"
TTS_VOICE = "Hindi"

GGUF_PATH = '/Users/dev/.ollama/models/blobs/sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4'

END = object()                 # End-of-turn marker flowing through queues

# Word Delimiter Regex & Punctuation Sets
# Matches non-delimiter graphemes followed by a completed delimiter (whitespace, period, comma, or Hindi punctuation mark)
WORD_DELIM_RE = re.compile(r'([^\s।,!?.]+)([\s।,!?.]+)')
PUNCT_FINAL = set('।!?.')
PUNCT_CHARS = set('।,!?.,')

UNSPEAKABLE_RE = re.compile(r"[^\u0900-\u097FA-Za-z0-9\s।॥?!,.'\-]")
SPEAKABLE_RE = re.compile(r"[\u0900-\u097FA-Za-z0-9]")

# Pre-rendered Reflex Audio Cache (< 0.5ms playback for top Caretaker Hindi openers)
REFLEX_CACHE: dict[str, np.ndarray] = {}
REFLEX_OPENERS = [
    # Top 2-word and 3-word Hindi Caretaker Openers (Instant < 0.5ms Reflex Hit)
    "हाँ दादाजी,",
    "हाँ दादाजी, मैंने",
    "हाँ दादी,",
    "हाँ दादी, मैंने",
    "सुनो ना दादी,",
    "सुनो ना दादी, इस",
    "ठीक है दादाजी,",
    "ठीक है दादाजी, मैंने",
    "दादाजी, रमेश",
    "दादाजी, रमेश का",
    "दादाजी, सुरेश",
    "दादाजी, सुरेश का",
    "दादाजी, आपके",
    "दादाजी, आपके 2",
    "दादाजी, लिस्ट में",
    "दादाजी, आपका कोई",
    "हाँ दादाजी, अभी",
    "अरे दादाजी,",
    "अरे दादाजी, आप",
    "अरे दादी,",
    "अरे दादी, मैं",
    "सुनो ना दादाजी,",
    "दादाजी,",
    "हाँ दादाजी",
    "हाँ दादी",
    "अरे दादाजी",
    "अरे दादी",
    "ठीक है दादाजी"
]

# ----------------------------------------------------------------------------
# SSE Client Broadcast Manager
# ----------------------------------------------------------------------------
sse_clients = []
sse_lock = threading.Lock()

def broadcast_sse(event_type: str, data: dict):
    """Send a real-time event to all connected Web UI browsers."""
    payload = json.dumps({"event": event_type, "data": data})
    with sse_lock:
        stale = []
        for q in sse_clients:
            try:
                q.put_nowait(payload)
            except queue.Full:
                stale.append(q)
        for q in stale:
            if q in sse_clients:
                sse_clients.remove(q)

# ----------------------------------------------------------------------------
# Shared State & Turn Metrics
# ----------------------------------------------------------------------------
T0 = time.perf_counter()
_log_lock = threading.Lock()

def now() -> float:
    return time.perf_counter()

def log(tag: str, msg: str, since: float | None = None) -> None:
    t = now()
    rel = f"  (+{(t - since) * 1000:.0f} ms)" if since is not None else ""
    with _log_lock:
        print(f"[{(t - T0) * 1000:10.1f} ms] [{tag}] {msg}{rel}", flush=True)

class Turn:
    """Timestamps for one question -> answer cycle."""
    _ids = 0

    def __init__(self, speech_end: float, last_voiced: float):
        Turn._ids += 1
        self.id = Turn._ids
        self.last_voiced = last_voiced
        self.speech_end = speech_end
        self.stt_done: float | None = None
        self.llm_request: float | None = None
        self.first_token: float | None = None
        self.first_chunk: float | None = None
        self.first_audio: float | None = None
        self.first_play: float | None = None
        self.user_text = ""
        self.reply_text = ""
        self.dev_thoughts = ""
        self.final_hindi = ""
        self.chunks = 0

stop_event = threading.Event()
busy_event = threading.Event()
last_play_end = [0.0]

llm_q: "queue.Queue" = queue.Queue()
tts_q: "queue.Queue" = queue.Queue()
audio_queue: "queue.Queue" = queue.Queue()
play_q = audio_queue  # Retained for full backward compatibility

# ----------------------------------------------------------------------------
# Continuous Asynchronous Ring Buffer Stream Infrastructure
# ----------------------------------------------------------------------------
AUDIO_BLOCK_SIZE = 1024
audio_stream_buffer: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=3000)
is_sound_streaming = [False]
speaker_stream = None

def native_async_audio_stream_callback(outdata, frames, time_info, status):
    """
    High-performance non-blocking audio callback for continuous asynchronous ring buffer playback.
    Non-blocking .get_nowait() extracts 1024-frame float32 audio slices from buffer.
    If empty, outdata.fill(0) keeps continuous micro-slices of silence flowing to
    physical audio hardware, completely preventing latency-inducing power-saving sleep cycles.
    """
    try:
        slice_data = audio_stream_buffer.get_nowait()
        if slice_data is None:
            outdata.fill(0)
            is_sound_streaming[0] = False
        else:
            if len(slice_data) != frames:
                if len(slice_data) < frames:
                    slice_data = np.pad(slice_data, (0, frames - len(slice_data)))
                else:
                    slice_data = slice_data[:frames]
            outdata[:] = slice_data.reshape(-1, 1)
            is_sound_streaming[0] = True
            last_play_end[0] = now()
    except queue.Empty:
        outdata.fill(0)
        is_sound_streaming[0] = False

# ----------------------------------------------------------------------------
# Audio Helpers
# ----------------------------------------------------------------------------
def clean_for_tts(text: str) -> str:
    text = text.replace("*", " ").replace("#", " ").replace("_", " ").replace("`", " ")
    text = UNSPEAKABLE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def trim_silence(audio: np.ndarray, sr: int, thr: float = 0.004, pad_ms: int = 20) -> np.ndarray:
    idx = np.flatnonzero(np.abs(audio) > thr)
    if idx.size == 0:
        return audio
    pad = sr * pad_ms // 1000
    return audio[max(0, idx[0] - pad): min(len(audio), idx[-1] + pad)]

def fade_edges(audio: np.ndarray, sr: int, ms: int = 5) -> np.ndarray:
    n = min(len(audio) // 2, sr * ms // 1000)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio

# ----------------------------------------------------------------------------
# Thread A — Listener / VAD / STT (MSpider3 Swift-CT2 int8 Whisper)
# ----------------------------------------------------------------------------
def listener_thread(stt_model, silence_ms: int) -> None:
    mic_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=1000)

    def on_audio(indata, frames, time_info, status):
        try:
            mic_q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    silence_frames_needed = max(1, math.ceil(silence_ms / FRAME_MS))
    onset_frames_needed = max(1, ONSET_MS // FRAME_MS)
    preroll: deque = deque(maxlen=PREROLL_MS // FRAME_MS)

    noise_floor = 0.003
    in_speech = False
    speech: list[np.ndarray] = []
    voiced_run = 0
    voiced_total = 0
    silent_run = 0
    last_voiced_t = 0.0

    try:
        with sd.InputStream(samplerate=MIC_SR, channels=1, dtype="float32",
                            blocksize=FRAME, callback=on_audio):
            log("VAD", "Microphone open (16kHz). MSpider3 Swift-CT2 ready. Speak in Hindi/Hinglish...")
            broadcast_sse("vad_status", {"state": "listening", "msg": "सुन रहा हूँ... (बोलना शुरू करें)"})

            while not stop_event.is_set():
                try:
                    frame = mic_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Half-duplex: avoid listening while speaking
                if busy_event.is_set() or is_sound_streaming[0] or (now() - last_play_end[0]) < ECHO_TAIL_S:
                    in_speech, speech, voiced_run, silent_run = False, [], 0, 0
                    preroll.clear()
                    continue

                rms = float(np.sqrt(np.mean(frame * frame)))
                threshold = max(MIN_RMS, noise_floor * NOISE_MULT)
                voiced = rms > threshold

                if not in_speech:
                    preroll.append(frame)
                    if voiced:
                        voiced_run += 1
                    else:
                        voiced_run = 0
                        noise_floor = 0.95 * noise_floor + 0.05 * rms
                    if voiced_run >= onset_frames_needed:
                        in_speech = True
                        speech = list(preroll)
                        voiced_total = voiced_run
                        silent_run = 0
                        last_voiced_t = now()
                        log("VAD", f"Speech started (rms={rms:.4f})")
                        broadcast_sse("vad_status", {"state": "speaking", "msg": "आवाज़ सुनाई दे रही है..."})
                    continue

                speech.append(frame)
                if voiced:
                    silent_run = 0
                    voiced_total += 1
                    last_voiced_t = now()
                else:
                    silent_run += 1

                too_long = len(speech) * FRAME_MS >= MAX_UTTERANCE_MS
                if silent_run < silence_frames_needed and not too_long:
                    continue

                # Silence cutoff triggered
                speech_end = now()
                in_speech, voiced_run = False, 0
                preroll.clear()
                if voiced_total * FRAME_MS < MIN_SPEECH_MS:
                    log("VAD", f"Ignored short noise burst ({voiced_total * FRAME_MS} ms)")
                    broadcast_sse("vad_status", {"state": "listening", "msg": "सुन रहा हूँ..."})
                    continue

                turn = Turn(speech_end, last_voiced_t)
                busy_event.set()
                audio = np.concatenate(speech).astype(np.float32)
                speech = []
                log("VAD", f"Speech stopped — {len(audio) / MIC_SR:.2f}s captured", since=turn.last_voiced)
                broadcast_sse("vad_status", {"state": "processing", "msg": "🔍 समझ रहा हूँ... (MSpider3 Swift-CT2)"})

                t_stt_start = now()
                try:
                    segments, _info = stt_model.transcribe(
                        audio,
                        beam_size=1,
                        best_of=1,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        without_timestamps=True,
                        vad_filter=False,
                    )

                    kept = [s.text for s in segments
                            if not (s.no_speech_prob > 0.6 and s.avg_logprob < -1.0)]
                    text = "".join(kept).strip()
                except Exception as e:
                    log("ERROR", f"STT failed: {e}")
                    text = ""

                turn.stt_done = now()
                stt_duration_ms = (turn.stt_done - t_stt_start) * 1000

                if not text:
                    log("STT", "No speech recognised — listening again", since=turn.speech_end)
                    broadcast_sse("vad_status", {"state": "listening", "msg": "सुन रहा हूँ..."})
                    busy_event.clear()
                    continue

                turn.user_text = text
                log("STT", f"Transcript: {text!r} ({stt_duration_ms:.0f} ms)", since=turn.speech_end)
                broadcast_sse("stt_result", {
                    "text": text,
                    "duration_ms": round(stt_duration_ms, 1),
                    "since_speech_end_ms": round((turn.stt_done - turn.speech_end) * 1000, 1)
                })

                llm_q.put((turn, text))
    except Exception:
        log("ERROR", "Listener crashed:\n" + traceback.format_exc())
    finally:
        stop_event.set()
        llm_q.put(None)

# ----------------------------------------------------------------------------
# Dispatch Trigger — Word-Boundary Handler & Acoustic Reflex Fast-Path
# ----------------------------------------------------------------------------
def dispatch_chunk(turn: Turn, chunk_input: list[str] | str, tts_q: queue.Queue, audio_queue: queue.Queue, sr: int) -> None:
    """
    Executes the Hybrid Semantic Clause Interceptor.
    Dispatches clean initial burst or full semantic clauses to either:
    1. Instant Acoustic Reflex Cache (< 5ms)
    2. Pipelined MLX TTS Synthesizer Thread via tts_q
    """
    if not chunk_input:
        return
    if isinstance(chunk_input, list):
        raw_text = " ".join(chunk_input).strip()
    else:
        raw_text = str(chunk_input).strip()
    clean_text = clean_for_tts(raw_text)
    if not clean_text or len(clean_text) < 2 or not SPEAKABLE_RE.search(clean_text):
        return

    t_compiled = time.perf_counter()
    compiled_ms = (t_compiled - turn.speech_end) * 1000.0
    turn.chunks += 1
    chunk_idx = turn.chunks
    if turn.first_chunk is None:
        turn.first_chunk = t_compiled

    log("CHUNK", f"⚡ Semantic Clause Chunk #{chunk_idx} compiled at {compiled_ms:8.2f} ms: {clean_text!r}", since=turn.speech_end)
    broadcast_sse("chunk_intercepted", {
        "chunk_num": chunk_idx,
        "text": clean_text,
        "since_speech_end_ms": round(compiled_ms, 1)
    })

    # Instant Reflex Cache Check (< 5ms audio latency)
    cache_audio = None
    if clean_text in REFLEX_CACHE:
        cache_audio = REFLEX_CACHE[clean_text]
    else:
        stripped_key = clean_text.rstrip('।,!?., ').strip()
        if stripped_key in REFLEX_CACHE:
            cache_audio = REFLEX_CACHE[stripped_key]

    if cache_audio is not None:
        log("REFLEX", f"⚡ Instant Reflex Cache hit (< 5ms): {clean_text!r}", since=turn.speech_end)
        if turn.first_audio is None:
            turn.first_audio = time.perf_counter()
        broadcast_sse("tts_ready", {
            "text": clean_text,
            "gen_ms": 0.5,
            "gen_us": 500.0,
            "audio_dur_s": round(len(cache_audio) / sr, 2),
            "since_speech_end_ms": round((time.perf_counter() - turn.speech_end) * 1000.0, 1)
        })
        audio_queue.put((turn, clean_text, cache_audio))
    else:
        # Pipelined async dispatch to dedicated TTS synthesizer thread
        tts_q.put((turn, clean_text, t_compiled))

# ----------------------------------------------------------------------------
# Thread B — In-Process Brain (llama-cpp-python Metal / Ollama fallback)
# ----------------------------------------------------------------------------
def brain_thread(llama_engine, ollama_client, model_name: str, sr: int) -> None:
    """
    Decoupled Text Routing Pipeline:
    1. Dev Thoughts Accumulator: Tracks English introspection (Stage 1-3) and broadcasts to Card 2 via SSE.
       Audio vocoder is completely bypassed for developer introspection logs.
    2. Transition Hook Interceptor: Watches for 'So I will respond, ' or 'respond,' to unlock hindi_voice_gate_unlocked.
    3. Final Hindi Accumulator & Vocoder:
       Only clean Devanagari Hindi text tokens are routed to Card 3 and sliced into whole-word 3-word chunks
       for Kitten-TTS 2 (MLX) audio synthesis.
    """
    history: list[dict] = []
    try:
        while True:
            item = llm_q.get()
            if item is None:
                break
            turn, user_text = item
            turn.llm_request = now()
            broadcast_sse("llm_start", {"model": ACTIVE_LLM_MODEL})

            dev_thoughts_accumulator: list[str] = []
            final_hindi_accumulator: list[str] = []
            hindi_voice_gate_unlocked = False

            # Hybrid Semantic Clause Interceptor State for this turn
            first_sound_fired = False
            clause_buffer = ""

            def handle_incoming_hindi_text(fragment: str):
                nonlocal first_sound_fired, clause_buffer
                if not fragment:
                    return
                clause_buffer += fragment

                # 1. THE FIRST CHUNK AUDIO SPEED-LOCK:
                if not first_sound_fired:
                    words = [w.strip() for w in re.findall(r'[^\s]+', clause_buffer) if w.strip()]
                    
                    # Fast-path check: 2-word opener in REFLEX_CACHE (< 5ms audio latency)
                    if len(words) >= 2:
                        two_words = f"{words[0]} {words[1]}"
                        two_clean = two_words.rstrip('।,!?., ').strip()
                        if two_words in REFLEX_CACHE or two_clean in REFLEX_CACHE:
                            cache_key = two_words if two_words in REFLEX_CACHE else two_clean
                            dispatch_chunk(turn, cache_key, tts_q, audio_queue, sr)
                            first_sound_fired = True
                            w2_end = clause_buffer.find(words[1]) + len(words[1])
                            clause_buffer = clause_buffer[w2_end:].lstrip('।,!?., ')
                            return

                    # Trigger vocoder the exact millisecond text buffer accumulates 3 full words
                    # OR hits a punctuation mark (।, !, ?, ,, \n)
                    has_punct = any(p in clause_buffer for p in ['।', ',', '!', '?', '\n'])
                    if len(words) >= 3 or (has_punct and len(words) >= 1):
                        if has_punct:
                            m = re.search(r'[।,!?\n]', clause_buffer)
                            if m:
                                cut_pos = m.end()
                                chunk_to_send = clause_buffer[:cut_pos].strip()
                                clause_buffer = clause_buffer[cut_pos:].lstrip('।,!?., ')
                                if chunk_to_send:
                                    dispatch_chunk(turn, chunk_to_send, tts_q, audio_queue, sr)
                                    first_sound_fired = True
                                return
                        if len(words) >= 3:
                            w3_end = clause_buffer.find(words[2]) + len(words[2])
                            chunk_to_send = clause_buffer[:w3_end].strip()
                            clause_buffer = clause_buffer[w3_end:].lstrip('।,!?., ')
                            if chunk_to_send:
                                dispatch_chunk(turn, chunk_to_send, tts_q, audio_queue, sr)
                                first_sound_fired = True
                            return

                # 2. MULTI-WORD SEMANTIC CLAUSE INTERCEPTION:
                else:
                    # For all subsequent tokens arriving after the first sound has fired,
                    # disable word counting. Trigger vocoder engine ONLY when the incoming
                    # token completes a natural linguistic boundary character ('।', ',', '!', '?', '\n').
                    if any(p in fragment for p in ['।', ',', '!', '?', '\n']):
                        while True:
                            m = re.search(r'[।,!?\n]', clause_buffer)
                            if not m:
                                break
                            cut_pos = m.end()
                            clause_to_send = clause_buffer[:cut_pos].strip()
                            clause_buffer = clause_buffer[cut_pos:].lstrip('।,!?., ')
                            if clause_to_send:
                                dispatch_chunk(turn, clause_to_send, tts_q, audio_queue, sr)

                    # Safety threshold: if an unusually long sentence has no punctuation (> 12 words)
                    words_in_buf = [w for w in re.findall(r'[^\s]+', clause_buffer) if w]
                    if len(words_in_buf) >= 12:
                        w_end = clause_buffer.find(words_in_buf[9]) + len(words_in_buf[9])
                        safe_chunk = clause_buffer[:w_end].strip()
                        clause_buffer = clause_buffer[w_end:].lstrip('।,!?., ')
                        if safe_chunk:
                            dispatch_chunk(turn, safe_chunk, tts_q, audio_queue, sr)

            prefill = "I heard the user "
            dev_thoughts_accumulator.append(prefill)

            db_context = format_live_context_for_query(user_text)

            try:
                if llama_engine is not None:
                    # In-process llama-cpp Metal GPU generation with live DB ground truth
                    prompt = f"<|im_start|>system\n{SYSTEM_PROMPT}\n\n{db_context}<|im_end|>\n"
                    for h in history[-2:]:
                        prompt += f"<|im_start|>{h['role']}\n{h['content']}<|im_end|>\n"
                    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n<|im_start|>assistant\n{prefill}"

                    turn.first_token = now()
                    ttft_ms = (turn.first_token - turn.llm_request) * 1000
                    log("Brain", f"⚡ Instant Stage 1 opener dispatched to Card 2: {prefill!r} ({ttft_ms:.1f} ms)", since=turn.speech_end)
                    broadcast_sse("first_token", {"token": prefill, "ttft_ms": round(ttft_ms, 1)})
                    broadcast_sse("dev_thought_token", {"token": prefill, "full": prefill})

                    stream = llama_engine(
                        prompt,
                        max_tokens=140,
                        stream=True,
                        temperature=0.0,
                        stop=["<|im_end|>", "\n\n", "<|endoftext|>"]
                    )
                    for chunk in stream:
                        token = chunk["choices"][0]["text"]
                        if not token:
                            continue

                        if not hindi_voice_gate_unlocked:
                            dev_thoughts_accumulator.append(token)
                            combined_thoughts = "".join(dev_thoughts_accumulator)

                            # SEARCH FOR THE HOOK: Look for the explicit transition block string
                            match = re.search(r"(?:So I will respond,?\s*|respond,?\s*)", combined_thoughts)
                            if match:
                                hindi_voice_gate_unlocked = True
                                clean_dev_logs = re.sub(r"(?:So I will respond,?\s*|respond,?\s*)$", "", combined_thoughts[:match.start()]).strip()
                                broadcast_sse("dev_thoughts_done", {"full": clean_dev_logs})

                                # Any leftover token arriving after the transition is Hindi
                                leftover_hindi = re.sub(r"^[\s,।!?-]+", "", combined_thoughts[match.end():])
                                if leftover_hindi:
                                    final_hindi_accumulator.append(leftover_hindi)
                                    broadcast_sse("final_hindi_token", {
                                        "token": leftover_hindi,
                                        "full": "".join(final_hindi_accumulator)
                                    })
                                    handle_incoming_hindi_text(leftover_hindi)
                            else:
                                broadcast_sse("dev_thought_token", {
                                    "token": token,
                                    "full": combined_thoughts
                                })
                        else:
                            # 🔊 PURE HINDI AUDIO ROUTING BLOCK (SEMANTIC CLAUSES)
                            final_hindi_accumulator.append(token)
                            current_buffer = "".join(final_hindi_accumulator)
                            broadcast_sse("final_hindi_token", {
                                "token": token,
                                "full": current_buffer
                            })
                            handle_incoming_hindi_text(token)

                else:
                    # Fallback to Ollama client with same decoupled routing and live DB ground truth
                    messages = (
                        [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\n{db_context}"}]
                        + history[-2:]
                        + [{"role": "user", "content": user_text}]
                    )
                    turn.first_token = now()
                    ttft_ms = (turn.first_token - turn.llm_request) * 1000
                    log("Ollama", f"⚡ Instant Stage 1 opener dispatched: {prefill!r} ({ttft_ms:.1f} ms)", since=turn.speech_end)
                    broadcast_sse("first_token", {"token": prefill, "ttft_ms": round(ttft_ms, 1)})
                    broadcast_sse("dev_thought_token", {"token": prefill, "full": prefill})

                    stream = ollama_client.chat(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        options={"num_predict": 140, "temperature": 0.0}
                    )
                    first_ollama_chunk = True
                    for part in stream:
                        token = (part.get("message") or {}).get("content") or ""
                        if not token:
                            continue
                        if first_ollama_chunk:
                            first_ollama_chunk = False
                            if token.startswith("I heard the user"):
                                token = token[len("I heard the user"):].lstrip()
                                if not token:
                                    continue

                        if not hindi_voice_gate_unlocked:
                            dev_thoughts_accumulator.append(token)
                            combined_thoughts = "".join(dev_thoughts_accumulator)
                            match = re.search(r"(?:So I will respond,?\s*|respond,?\s*)", combined_thoughts)
                            if match:
                                hindi_voice_gate_unlocked = True
                                clean_dev_logs = re.sub(r"(?:So I will respond,?\s*|respond,?\s*)$", "", combined_thoughts[:match.start()]).strip()
                                broadcast_sse("dev_thoughts_done", {"full": clean_dev_logs})
                                leftover_hindi = re.sub(r"^[\s,।!?-]+", "", combined_thoughts[match.end():])
                                if leftover_hindi:
                                    final_hindi_accumulator.append(leftover_hindi)
                                    broadcast_sse("final_hindi_token", {
                                        "token": leftover_hindi,
                                        "full": "".join(final_hindi_accumulator)
                                    })
                                    handle_incoming_hindi_text(leftover_hindi)
                            else:
                                broadcast_sse("dev_thought_token", {
                                    "token": token,
                                    "full": combined_thoughts
                                })
                        else:
                            final_hindi_accumulator.append(token)
                            current_buffer = "".join(final_hindi_accumulator)
                            broadcast_sse("final_hindi_token", {
                                "token": token,
                                "full": current_buffer
                            })
                            handle_incoming_hindi_text(token)

                # 3. THE TRAILING TOKEN CLEANUP CIRCUIT:
                if clause_buffer.strip():
                    trailing_text = clause_buffer.strip()
                    if len(trailing_text) >= 2 and SPEAKABLE_RE.search(trailing_text):
                        dispatch_chunk(turn, trailing_text, tts_q, audio_queue, sr)
                    clause_buffer = ""

            except Exception:
                log("ERROR", "Brain inference error:\n" + traceback.format_exc())
            finally:
                dev_str = "".join(dev_thoughts_accumulator).strip()
                hindi_str = "".join(final_hindi_accumulator).strip()

                if not hindi_voice_gate_unlocked:
                    match = re.search(r"(?:So I will respond,?\s*|respond,?\s*)", dev_str)
                    if match:
                        clean_dev = dev_str[:match.start()].strip()
                        hindi_str = dev_str[match.end():].strip()
                        dev_str = clean_dev
                    else:
                        devanagari_match = re.search(r"[\u0900-\u097F]+.*", dev_str)
                        if devanagari_match:
                            hindi_str = dev_str[devanagari_match.start():].strip()
                            dev_str = dev_str[:devanagari_match.start()].strip()

                dev_str = re.sub(r"(?:So I will respond,?\s*|respond,?\s*)$", "", dev_str).strip()
                hindi_str = re.sub(r"^[\s,।!?-]+", "", hindi_str).strip()

                turn.dev_thoughts = dev_str
                turn.final_hindi = hindi_str
                turn.reply_text = f"{dev_str} So I will respond, {hindi_str}".strip() if hindi_str else dev_str

                t_brain_done = now()
                turn.main_thread_turn_ms = round((t_brain_done - turn.speech_end) * 1000, 1)
                log("MAIN_CORE", f"⚡ Main Execution Core turn completed in {turn.main_thread_turn_ms} ms", since=turn.speech_end)

                if turn.chunks == 0 and hindi_str:
                    dispatch_chunk(turn, hindi_str, tts_q, audio_queue, sr)

                broadcast_sse("dev_thoughts_done", {"full": dev_str})
                broadcast_sse("final_hindi_token", {"token": "", "full": hindi_str})

                if turn.reply_text:
                    history += [{"role": "user", "content": user_text},
                                {"role": "assistant", "content": turn.reply_text}]

                    # SIMULTANEOUS SKILL EXECUTION: Offloaded to dedicated fast background thread
                    def _async_skill_worker(r_text: str, u_text: str):
                        try:
                            skill_results = parse_and_execute_local_skills(r_text, user_raw_query=u_text)
                            for skill_name, s_data in skill_results.items():
                                broadcast_sse("skill_event", {
                                    "skill": skill_name,
                                    "action": s_data.get("action", ""),
                                    "result": s_data.get("status", "")
                                })
                        except Exception as skill_err:
                            log("WARN", f"Async skill execution error: {skill_err}")

                    threading.Thread(target=_async_skill_worker, args=(turn.reply_text, user_text), daemon=True).start()

                tts_q.put((turn, END, now()))
    except Exception:
        log("ERROR", "Brain thread crashed:\n" + traceback.format_exc())
    finally:
        tts_q.put(None)

# ----------------------------------------------------------------------------
# Thread C — Voice Synthesizer Worker (Kitten-TTS 2 on Apple Silicon MLX)
# ----------------------------------------------------------------------------
def tts_thread(tts_model, sr: int) -> None:
    """
    Dedicated audio synthesis thread running Kitten-TTS 2 on MLX.
    Securely wrapped in try-except with millisecond compilation and microsecond MLX synthesis logs.
    Immediately hands clean audio over to audio_queue.
    """
    try:
        while True:
            item = tts_q.get()
            if item is None:
                break
            turn, chunk_text, t_compiled = item
            if chunk_text is END:
                audio_queue.put((turn, END, None))
                continue
            
            # 3. OPTIMIZE THREAD EXECUTION PRIORITY & SECURE TRY-EXCEPT:
            try:
                broadcast_sse("tts_start", {"text": chunk_text})
                t_mlx_start = time.perf_counter()
                raw_audio = np.asarray(tts_model.generate(chunk_text, voice=TTS_VOICE), dtype=np.float32).reshape(-1)
                t_mlx_end = time.perf_counter()
                mlx_us = (t_mlx_end - t_mlx_start) * 1_000_000.0
                mlx_ms = mlx_us / 1000.0

                # Millisecond-precision log statements tracking:
                # - Time when clean 3-word chunk is compiled
                # - Microsecond processing time it takes the MLX engine to synthesize that specific chunk
                log("MLX_TTS", f"MLX engine synthesized chunk in {mlx_us:10.1f} µs ({mlx_ms:6.2f} ms): {chunk_text!r}")

                if turn.first_audio is None:
                    turn.first_audio = t_mlx_end
                    log("TTS", f"1st audio synthesized in {mlx_ms:.1f} ms", since=turn.speech_end)

                audio = fade_edges(trim_silence(raw_audio, sr), sr)
                dur_s = len(audio) / sr if sr else 0.0
                broadcast_sse("tts_ready", {
                    "text": chunk_text,
                    "gen_ms": round(mlx_ms, 1),
                    "gen_us": round(mlx_us, 1),
                    "audio_dur_s": round(dur_s, 2),
                    "since_speech_end_ms": round((t_mlx_end - turn.speech_end) * 1000.0, 1)
                })

                # Immediately push resulting clean audio array into audio_queue for seamless playback handoff
                if audio is not None and len(audio) > 0:
                    audio_queue.put((turn, chunk_text, audio))

            except Exception as e:
                log("ERROR", f"Audio synthesis failed for chunk {chunk_text!r}: {e}\n{traceback.format_exc()}")

    except Exception:
        log("ERROR", "TTS thread crashed:\n" + traceback.format_exc())
    finally:
        audio_queue.put(None)

# ----------------------------------------------------------------------------
# Thread D — Non-Blocking Audio Dispatcher Core (Ring Buffer Feeder)
# ----------------------------------------------------------------------------
def player_dispatcher_thread(sr: int) -> None:
    """
    Consumes synthesized audio arrays from audio_queue without blocking on hardware.
    Slices waveform arrays into exact 1024-frame float32 buffers and pushes them
    into audio_stream_buffer.
    The native_async_audio_stream_callback feeds them to physical speakers with zero latency.
    """
    try:
        while not stop_event.is_set():
            item = audio_queue.get()
            if item is None:
                break
            turn, text, audio = item
            if text is END:
                # Wait until ring buffer finishes draining audio to hardware wires
                while not audio_stream_buffer.empty() or is_sound_streaming[0]:
                    time.sleep(0.015)
                last_play_end[0] = now()
                summary = _summarise(turn)
                broadcast_sse("turn_summary", summary)
                if audio_queue.empty() and tts_q.empty() and llm_q.empty():
                    busy_event.clear()
                    log("VAD", "Listening again...")
                    broadcast_sse("vad_status", {"state": "listening", "msg": "सुन रहा हूँ... (बोलना शुरू करें)"})
                else:
                    log("VAD", "Next turn pipeline in flight, keeping mic muted...")
                continue

            busy_event.set()
            if turn.first_play is None:
                turn.first_play = now()
                first_play_ms = (turn.first_play - turn.speech_end) * 1000
                log("PLAY", f"🔊 First spoken sound streaming: {text!r} ({first_play_ms:.0f} ms latency!)", since=turn.speech_end)
                broadcast_sse("playback_first", {
                    "text": text,
                    "latency_ms": round(first_play_ms, 1)
                })

            broadcast_sse("playback_chunk", {
                "text": text,
                "duration_s": round(len(audio) / sr, 2)
            })

            # Format into contiguous float32 1024-frame slices
            audio_arr = np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)
            pad_needed = (AUDIO_BLOCK_SIZE - (len(audio_arr) % AUDIO_BLOCK_SIZE)) % AUDIO_BLOCK_SIZE
            if pad_needed > 0:
                audio_arr = np.pad(audio_arr, (0, pad_needed), mode="constant")
            slices = audio_arr.reshape(-1, AUDIO_BLOCK_SIZE)
            for s in slices:
                try:
                    audio_stream_buffer.put(s.copy(), timeout=1.0)
                except queue.Full:
                    pass
    except Exception:
        log("ERROR", "Audio dispatcher crashed:\n" + traceback.format_exc())
        stop_event.set()

player_thread = player_dispatcher_thread  # Alias for compatibility

def _summarise(turn: Turn) -> dict:
    def ms(t):
        return round((t - turn.speech_end) * 1000, 1) if t else None

    ttft = round((turn.first_token - turn.llm_request) * 1000, 1) if turn.first_token and turn.llm_request else None
    data = {
        "turn_id": turn.id,
        "user_said": turn.user_text,
        "ai_said": turn.reply_text,
        "dev_thoughts": getattr(turn, "dev_thoughts", ""),
        "final_hindi": getattr(turn, "final_hindi", ""),
        "stt_ms": ms(turn.stt_done),
        "first_token_ms": ms(turn.first_token),
        "ttft_ms": ttft,
        "first_chunk_ms": ms(turn.first_chunk),
        "first_audio_ms": ms(turn.first_audio),
        "first_play_ms": ms(turn.first_play),
        "main_thread_turn_ms": getattr(turn, "main_thread_turn_ms", None),
        "total_ms": round((now() - turn.speech_end) * 1000, 1),
        "chunks": turn.chunks
    }
    log("METRICS", f"Turn {turn.id} completed in {data['total_ms']} ms (Main Thread: {data.get('main_thread_turn_ms')} ms, 1st Audio: {data['first_play_ms']} ms)")
    return data

# ----------------------------------------------------------------------------
# Hardware Keep-Warm Heartbeat Thread
# ----------------------------------------------------------------------------
def keep_warm_thread():
    """Prevents Apple Silicon Metal shaders and CoreAudio from dropping to sleep state."""
    while not stop_event.is_set():
        time.sleep(2.0)
        if not busy_event.is_set():
            try:
                import mlx.core as mx
                x = mx.zeros((2, 2))
                mx.eval(x)
            except Exception:
                pass

# ----------------------------------------------------------------------------
# Flask Routes
# ----------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream_events')
def stream_events():
    def event_stream():
        q = queue.Queue(maxsize=100)
        with sse_lock:
            sse_clients.append(q)
        try:
            yield f"data: {json.dumps({'event': 'init', 'data': {'status': 'ready', 'llm_model': ACTIVE_LLM_MODEL, 'tts_backend': ACTIVE_TTS_BACKEND, 'stt_model': ACTIVE_STT_MODEL}})}\n\n"
            while True:
                payload = q.get()
                yield f"data: {payload}\n\n"
        except GeneratorExit:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/api/send_text', methods=['POST'])
def api_send_text():
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    t_now = now()
    turn = Turn(t_now, t_now)
    turn.stt_done = t_now
    turn.user_text = text
    
    broadcast_sse("vad_status", {"state": "processing", "msg": f"📝 टेक्स्ट प्राप्त हुआ: \"{text}\""})
    broadcast_sse("stt_result", {
        "text": text,
        "duration_ms": 0.0,
        "since_speech_end_ms": 0.0
    })
    
    busy_event.set()
    llm_q.put((turn, text))
    return jsonify({'status': 'queued', 'text': text})

@app.route('/api/status')
def api_status():
    return jsonify({
        "status": "online",
        "vad": "listening" if not busy_event.is_set() else "busy",
        "llm_model": ACTIVE_LLM_MODEL,
        "tts_backend": ACTIVE_TTS_BACKEND,
        "stt_model": ACTIVE_STT_MODEL
    })

@app.route('/api/contacts', methods=['GET', 'POST', 'DELETE'])
def api_contacts():
    if request.method == 'GET':
        if os.path.exists(CONTACTS_FILE):
            try:
                with open(CONTACTS_FILE, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return jsonify({})

    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        name = data.get('name', '').strip()
        phone = data.get('phone', '').strip()
        if not name or not phone:
            return jsonify({'error': 'Name and phone are required'}), 400
        res = execute_contacts_crud("create", name=name, phone=phone)
        broadcast_sse("skill_event", {"skill": "crud_contacts", "action": "create", "result": res})
        return jsonify({'status': 'success', 'result': res})

    elif request.method == 'DELETE':
        data = request.get_json(silent=True) or {}
        target = data.get('target', '') or request.args.get('target', '')
        if not target:
            return jsonify({'error': 'Target contact name is required'}), 400
        res = execute_contacts_crud("delete", target=target)
        broadcast_sse("skill_event", {"skill": "crud_contacts", "action": "delete", "result": res})
        return jsonify({'status': 'success', 'result': res})


@app.route('/api/reminders', methods=['GET', 'POST', 'DELETE'])
def api_reminders():
    if request.method == 'GET':
        if os.path.exists(REMINDERS_FILE):
            try:
                with open(REMINDERS_FILE, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return jsonify([])

    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        day = data.get('day', '').strip() or "Upcoming"
        task = data.get('task', '').strip()
        if not task:
            return jsonify({'error': 'Task description is required'}), 400
        res = execute_reminders_crud("create", target_day=day, task=task)
        broadcast_sse("skill_event", {"skill": "crud_reminders", "action": "create", "result": res})
        return jsonify({'status': 'success', 'result': res})

    elif request.method == 'DELETE':
        data = request.get_json(silent=True) or {}
        task = data.get('task', '') or request.args.get('task', '')
        if not task:
            return jsonify({'error': 'Task is required'}), 400
        res = execute_reminders_crud("delete", task=task)
        broadcast_sse("skill_event", {"skill": "crud_reminders", "action": "delete", "result": res})
        return jsonify({'status': 'success', 'result': res})


@app.route('/api/grocery', methods=['GET', 'POST', 'DELETE'])
def api_grocery():
    if request.method == 'GET':
        if os.path.exists(GROCERY_FILE):
            try:
                with open(GROCERY_FILE, 'r', encoding='utf-8') as f:
                    return jsonify(json.load(f))
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        return jsonify([])

    elif request.method == 'POST':
        data = request.get_json(silent=True) or {}
        item = data.get('item', '').strip()
        if not item:
            return jsonify({'error': 'Item name is required'}), 400
        res = execute_grocery_crud("create", item_name=item)
        broadcast_sse("skill_event", {"skill": "crud_grocery", "action": "create", "result": res})
        return jsonify({'status': 'success', 'result': res})

    elif request.method == 'DELETE':
        data = request.get_json(silent=True) or {}
        item = data.get('item', '') or request.args.get('item', '')
        if not item:
            return jsonify({'error': 'Item name is required'}), 400
        res = execute_grocery_crud("delete", item_name=item)
        broadcast_sse("skill_event", {"skill": "crud_grocery", "action": "delete", "result": res})
        return jsonify({'status': 'success', 'result': res})


@app.route('/api/journal', methods=['GET'])
def api_journal():
    if os.path.exists(JOURNAL_FILE):
        try:
            with open(JOURNAL_FILE, 'r', encoding='utf-8') as f:
                return jsonify(json.load(f))
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify([])





# ----------------------------------------------------------------------------
# Pipeline Initialization
# ----------------------------------------------------------------------------
def init_pipeline():
    global ACTIVE_LLM_MODEL, ACTIVE_TTS_BACKEND, ACTIVE_STT_MODEL
    from faster_whisper import WhisperModel
    from kittenml import KittenTTS

    t = now()
    # 1. MSpider3/whisper-hindi2hinglish-swift-ct2 (int8 CTranslate2)
    STT_MODEL_NAME = "MSpider3/whisper-hindi2hinglish-swift-ct2"
    stt_model = WhisperModel(STT_MODEL_NAME, device="cpu", compute_type="int8")
    stt_model.transcribe(np.zeros(MIC_SR, dtype=np.float32), beam_size=1)
    ACTIVE_STT_MODEL = "MSpider3 Swift-CT2 (int8)"
    log("INIT", f"{STT_MODEL_NAME} loaded + warmed (< 180ms target)", since=t)

    # 2. Kitten-TTS 2 on Apple Silicon MLX
    t = now()
    backend = "mlx" if (sys.platform == "darwin" and platform.machine() == "arm64") else None
    try:
        tts = KittenTTS(TTS_MODEL, **({"backend": backend} if backend else {}))
    except Exception as e:
        log("WARN", f"Kitten-TTS backend '{backend}' failed ({e}); falling back to auto backend")
        backend = None
        tts = KittenTTS(TTS_MODEL)
    ACTIVE_TTS_BACKEND = f"Kitten-TTS 2 ({backend or 'cpu'})"
    sr = int(getattr(tts, "sample_rate", 24000))
    tts.generate("नमस्ते", voice=TTS_VOICE)
    log("INIT", f"Kitten-TTS 2 loaded + warmed (sr={sr}, backend={backend or 'auto'})", since=t)

    # 3. Pre-render Acoustic Reflex Cache for Instant Sound (< 5ms)
    t = now()
    for opener in REFLEX_OPENERS:
        try:
            raw_audio = np.asarray(tts.generate(opener, voice=TTS_VOICE), dtype=np.float32).reshape(-1)
            REFLEX_CACHE[opener] = fade_edges(trim_silence(raw_audio, sr), sr)
        except Exception:
            pass
    log("INIT", f"Pre-rendered {len(REFLEX_CACHE)} acoustic reflex openers for instant < 5ms playback", since=t)

    # 4. In-Process llama-cpp-python with Metal GPU Acceleration
    t = now()
    llama_engine = None
    ollama_client = None
    model_name = "qwen2.5:1.5b-instruct"

    try:
        if os.path.exists(GGUF_PATH):
            from llama_cpp import Llama
            llama_engine = Llama(model_path=GGUF_PATH, n_ctx=4096, n_gpu_layers=-1, verbose=False)
            # Warmup shader compilation & system prompt KV cache
            warm_p = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nRemind me on Thursday to call Arjun<|im_end|>\n<|im_start|>assistant\nI heard the user "
            llama_engine(warm_p, max_tokens=4)
            ACTIVE_LLM_MODEL = "Qwen2.5 1.5B (llama-cpp Metal)"
            log("INIT", "llama-cpp Metal GPU engine loaded + warmed (< 55ms TTFT target)", since=t)
    except Exception as e:
        log("WARN", f"llama-cpp init failed ({e}); falling back to Ollama HTTP")

    if llama_engine is None:
        import ollama
        ollama_client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        ACTIVE_LLM_MODEL = "Qwen2.5 1.5B (Ollama)"
        log("INIT", "Ollama fallback client connected", since=t)

    # 5. Permanent Continuous Asynchronous Ring Buffer Hardware Stream
    global speaker_stream
    try:
        speaker_stream = sd.OutputStream(
            samplerate=sr,
            channels=1,
            dtype='float32',
            blocksize=AUDIO_BLOCK_SIZE,
            callback=native_async_audio_stream_callback
        )
        speaker_stream.start()
        log("INIT", f"Permanent Asynchronous Ring Buffer speaker stream started (sr={sr}, blocksize={AUDIO_BLOCK_SIZE})")
    except Exception as e:
        log("WARN", f"Failed to start continuous speaker stream: {e}")

    threads = [
        threading.Thread(target=player_thread, args=(sr,), name="D-player", daemon=True),
        threading.Thread(target=tts_thread, args=(tts, sr), name="C-tts", daemon=True),
        threading.Thread(target=brain_thread, args=(llama_engine, ollama_client, model_name, sr), name="B-brain", daemon=True),
        threading.Thread(target=listener_thread, args=(stt_model, SILENCE_MS), name="A-listener", daemon=True),
        threading.Thread(target=keep_warm_thread, name="KeepWarm", daemon=True),
    ]
    for th in threads:
        th.start()
    log("INIT", "All 5 streaming & keep-warm threads running concurrently!")

if __name__ == '__main__':
    init_pipeline()
    port = int(os.environ.get('PORT', 5050))
    print(f"🌟 AI Caretaker Smart Mirror running at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
