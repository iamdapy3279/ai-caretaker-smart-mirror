#!/usr/bin/env python3
"""
caretaker_stream.py — Parallel streaming voice pipeline for a Hindi caretaker.

    Thread A  Listener : mic (16 kHz) -> energy VAD -> faster-whisper -> llm_q
    Thread B  Brain    : ollama stream=True -> chunk interceptor -> tts_q
    Thread C  TTS      : Kitten-TTS 2 ("Hindi" voice) -> play_q
    Thread D  Player   : gapless sounddevice.OutputStream playback

All hand-offs use queue.Queue. A turn ends when an END marker travels the
whole chain (A/B -> C -> D). The player then re-opens the microphone.

Run:
    source venv/bin/activate
    python caretaker_stream.py                     # defaults from the spec
    python caretaker_stream.py --tts-backend mlx   # Apple-silicon backend (needs: pip install "kittenml[mlx]")
"""

import argparse
import math
import os
import platform
import queue
import re
import sys
import threading
import time
import traceback
from collections import deque

import numpy as np
import sounddevice as sd

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
MIC_SR = 16000                 # Whisper expects 16 kHz mono float32
FRAME_MS = 30                  # VAD frame size
FRAME = MIC_SR * FRAME_MS // 1000
PREROLL_MS = 300               # audio kept from before speech onset
ONSET_MS = 60                  # voiced audio needed to declare speech start
MIN_SPEECH_MS = 250            # shorter bursts are ignored (coughs, clicks)
MAX_UTTERANCE_MS = 15000       # hard cap per utterance
MIN_RMS = 0.006                # absolute floor for the energy threshold
NOISE_MULT = 3.0               # threshold = noise_floor * NOISE_MULT
ECHO_TAIL_S = 0.35             # keep mic muted this long after playback ends

SYSTEM_PROMPT = (
    "You are a warm Hindi caretaker. Speak exclusively in simple, "
    "conversational Hindi. Keep answers sweet and under 10 words per response."
)
PREFERRED_MODELS = ["gemma2:2b", "qwen2.5:3b", "qwen2.5:1.5b-instruct", "gemma3:1b"]
HISTORY_TURNS = 2              # past exchanges sent back to the LLM

TTS_MODEL = "KittenML/kitten-tts-2"
TTS_VOICE = "Hindi"

END = object()                 # end-of-turn marker that flows through the queues

# Chunk interceptor patterns
BOUNDARY_RE = re.compile(r"[।॥?!,.]")
WORD_RE = re.compile(r"\S+\s+")                               # completed words only
UNSPEAKABLE_RE = re.compile(r"[^\u0900-\u097FA-Za-z0-9\s।॥?!,.'\-]")
SPEAKABLE_RE = re.compile(r"[\u0900-\u097FA-Za-z0-9]")

# ----------------------------------------------------------------------------
# Logging with millisecond precision
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


# ----------------------------------------------------------------------------
# Shared state
# ----------------------------------------------------------------------------
class Turn:
    """Timestamps for one question -> answer cycle (all perf_counter seconds)."""

    _ids = 0

    def __init__(self, speech_end: float, last_voiced: float):
        Turn._ids += 1
        self.id = Turn._ids
        self.last_voiced = last_voiced      # last frame above the VAD threshold
        self.speech_end = speech_end        # moment VAD declared end of speech
        self.stt_done: float | None = None
        self.llm_request: float | None = None
        self.first_token: float | None = None
        self.first_chunk: float | None = None
        self.first_audio: float | None = None
        self.first_play: float | None = None
        self.user_text = ""
        self.reply_text = ""
        self.chunks = 0


stop_event = threading.Event()
busy_event = threading.Event()             # set from end of speech until playback finishes
last_play_end = [0.0]                      # boxed float, written by player, read by listener

llm_q: "queue.Queue" = queue.Queue()
tts_q: "queue.Queue" = queue.Queue()
play_q: "queue.Queue" = queue.Queue()


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def clean_for_tts(text: str) -> str:
    """Remove emoji / markdown that the LLM sometimes emits."""
    text = text.replace("*", " ").replace("#", " ").replace("_", " ").replace("`", " ")
    text = UNSPEAKABLE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def pop_chunks(buf: str, max_words: int) -> tuple[list[str], str]:
    """
    Chunk interceptor. Cut the buffer at the first punctuation boundary, or
    after `max_words` completed words, whichever comes first. A word only
    counts as complete once whitespace follows it, so a word still arriving
    token-by-token is never split.
    """
    chunks: list[str] = []
    while buf:
        m = BOUNDARY_RE.search(buf)
        word_ends = [w.end() for w in WORD_RE.finditer(buf)]
        if m and (len(word_ends) < max_words or m.end() <= word_ends[max_words - 1]):
            cut = m.end()
        elif len(word_ends) >= max_words:
            cut = word_ends[max_words - 1]
        else:
            break
        piece = clean_for_tts(buf[:cut])
        buf = buf[cut:].lstrip()
        if SPEAKABLE_RE.search(piece):
            chunks.append(piece)
    return chunks, buf


def trim_silence(audio: np.ndarray, sr: int, thr: float = 0.004, pad_ms: int = 20) -> np.ndarray:
    """Trim leading/trailing silence so consecutive chunks join without gaps."""
    idx = np.flatnonzero(np.abs(audio) > thr)
    if idx.size == 0:
        return audio
    pad = sr * pad_ms // 1000
    return audio[max(0, idx[0] - pad): min(len(audio), idx[-1] + pad)]


def fade_edges(audio: np.ndarray, sr: int, ms: int = 5) -> np.ndarray:
    """Short linear fades prevent clicks at chunk boundaries."""
    n = min(len(audio) // 2, sr * ms // 1000)
    if n > 0:
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        audio[:n] *= ramp
        audio[-n:] *= ramp[::-1]
    return audio


def msg_content(part) -> str:
    """Works with both ollama ChatResponse objects and plain dicts."""
    try:
        return part.message.content or ""
    except AttributeError:
        return (part.get("message") or {}).get("content") or ""


# ----------------------------------------------------------------------------
# Thread A — Listener / VAD / STT
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
            log("VAD", f"Microphone open. Silence cut-off = {silence_ms} ms. Speak in Hindi...")
            while not stop_event.is_set():
                try:
                    frame = mic_q.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Half-duplex: drop mic audio while we are answering (prevents
                # the speaker output from being transcribed as a new question).
                if busy_event.is_set() or (now() - last_play_end[0]) < ECHO_TAIL_S:
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
                        noise_floor = 0.95 * noise_floor + 0.05 * rms   # adapt only on silence
                    if voiced_run >= onset_frames_needed:
                        in_speech = True
                        speech = list(preroll)
                        voiced_total = voiced_run
                        silent_run = 0
                        last_voiced_t = now()
                        log("VAD", f"Speech started (rms={rms:.4f}, thr={threshold:.4f})")
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

                # ---- end of utterance -------------------------------------
                speech_end = now()
                in_speech, voiced_run = False, 0
                preroll.clear()
                if voiced_total * FRAME_MS < MIN_SPEECH_MS:
                    log("VAD", f"Ignored short noise burst ({voiced_total * FRAME_MS} ms)")
                    continue

                turn = Turn(speech_end, last_voiced_t)
                busy_event.set()
                audio = np.concatenate(speech).astype(np.float32)
                speech = []
                log("VAD", f"Speech stopped — {len(audio) / MIC_SR:.2f}s captured"
                           f"{' (max length hit)' if too_long else ''}", since=turn.last_voiced)

                segments, _info = stt_model.transcribe(
                    audio,
                    language="hi",
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
                turn.stt_done = now()

                if not text:
                    log("STT", "No speech recognised — listening again", since=turn.speech_end)
                    busy_event.clear()
                    continue

                turn.user_text = text
                log("STT", f"Transcript: {text!r}", since=turn.speech_end)
                llm_q.put((turn, text))
    except Exception:
        log("ERROR", "Listener crashed:\n" + traceback.format_exc())
    finally:
        stop_event.set()
        llm_q.put(None)        # cascade shutdown: B -> C -> D


# ----------------------------------------------------------------------------
# Thread B — Brain (Ollama streaming) + Chunk Interceptor
# ----------------------------------------------------------------------------
def brain_thread(client, model: str, max_words: int) -> None:
    history: list[dict] = []
    try:
        while True:
            item = llm_q.get()
            if item is None:
                break
            turn, user_text = item
            messages = ([{"role": "system", "content": SYSTEM_PROMPT}]
                        + history[-2 * HISTORY_TURNS:]
                        + [{"role": "user", "content": user_text}])
            buf = ""
            reply_parts: list[str] = []
            try:
                turn.llm_request = now()
                stream = client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    keep_alive="30m",
                    options={"num_predict": 48, "temperature": 0.6},
                )
                for part in stream:
                    token = msg_content(part)
                    if not token:
                        continue
                    if turn.first_token is None:
                        turn.first_token = now()
                        log("Ollama", f"First token arrived: {token!r}  "
                                      f"(request->token {(turn.first_token - turn.llm_request) * 1000:.0f} ms, "
                                      f"target < 50 ms)", since=turn.speech_end)
                    reply_parts.append(token)
                    buf += token
                    chunks, buf = pop_chunks(buf, max_words)
                    for chunk in chunks:
                        _emit_chunk(turn, chunk)

                tail = clean_for_tts(buf)
                if SPEAKABLE_RE.search(tail):
                    _emit_chunk(turn, tail)
            except Exception:
                log("ERROR", "Ollama request failed:\n" + traceback.format_exc())
            finally:
                turn.reply_text = "".join(reply_parts).strip()
                if turn.reply_text:
                    history += [{"role": "user", "content": user_text},
                                {"role": "assistant", "content": turn.reply_text}]
                tts_q.put((turn, END))
    except Exception:
        log("ERROR", "Brain thread crashed:\n" + traceback.format_exc())
    finally:
        tts_q.put(None)


def _emit_chunk(turn: Turn, chunk: str) -> None:
    turn.chunks += 1
    if turn.first_chunk is None:
        turn.first_chunk = now()
        log("Chunk", f"First phrase -> TTS: {chunk!r}", since=turn.speech_end)
    else:
        log("Chunk", f"Phrase {turn.chunks} -> TTS: {chunk!r}")
    tts_q.put((turn, chunk))


# ----------------------------------------------------------------------------
# Thread C — Voice synthesiser (Kitten-TTS 2)
# ----------------------------------------------------------------------------
def tts_thread(tts, sr: int) -> None:
    try:
        while True:
            item = tts_q.get()
            if item is None:
                break
            turn, text = item
            if text is END:
                play_q.put((turn, END, None))
                continue
            try:
                t = now()
                audio = np.asarray(tts.generate(text, voice=TTS_VOICE), dtype=np.float32).reshape(-1)
                gen = now() - t
                audio = fade_edges(trim_silence(audio, sr), sr)
                dur = len(audio) / sr
                if turn.first_audio is None:
                    turn.first_audio = now()
                    log("TTS", f"First audio chunk generated ({gen * 1000:.0f} ms for {dur:.2f}s audio)",
                        since=turn.speech_end)
                else:
                    log("TTS", f"Chunk generated: {text!r} ({gen * 1000:.0f} ms for {dur:.2f}s, "
                               f"RTF {gen / dur if dur else 0:.2f})")
                play_q.put((turn, text, audio))
            except Exception:
                log("ERROR", f"TTS failed for {text!r}:\n" + traceback.format_exc())
    except Exception:
        log("ERROR", "TTS thread crashed:\n" + traceback.format_exc())
    finally:
        play_q.put(None)


# ----------------------------------------------------------------------------
# Thread D — Gapless audio player
# ----------------------------------------------------------------------------
def player_thread(sr: int) -> None:
    # A single long-lived OutputStream with blocking write() queues each chunk
    # directly behind the previous one. (sd.play() would stop the current
    # sound every time it is called, so chunks would cut each other off.)
    try:
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32", latency="low") as out:
            while True:
                item = play_q.get()
                if item is None:
                    break
                turn, text, audio = item
                if text is END:
                    time.sleep(out.latency)          # let the device buffer drain
                    last_play_end[0] = now()
                    _summarise(turn)
                    busy_event.clear()
                    log("VAD", "Listening again...")
                    continue
                if turn.first_play is None:
                    turn.first_play = now()
                    log("PLAY", f"First audio chunk playing: {text!r}  (target < 250 ms)",
                        since=turn.speech_end)
                out.write(audio.reshape(-1, 1))      # blocks only while the device buffer is full
    except Exception:
        log("ERROR", "Player crashed:\n" + traceback.format_exc())
        stop_event.set()


def _summarise(turn: Turn) -> None:
    def ms(t):
        return f"{(t - turn.speech_end) * 1000:7.0f} ms" if t else "      n/a"

    first_tok = (f"{(turn.first_token - turn.llm_request) * 1000:.0f} ms"
                 if turn.first_token and turn.llm_request else "n/a")
    lines = [
        f"Turn {turn.id} summary (times measured from VAD end-of-speech)",
        f"  user      : {turn.user_text}",
        f"  reply     : {turn.reply_text}",
        f"  STT done  : {ms(turn.stt_done)}",
        f"  1st token : {ms(turn.first_token)}   (LLM request->token: {first_tok})",
        f"  1st chunk : {ms(turn.first_chunk)}",
        f"  1st audio : {ms(turn.first_audio)}",
        f"  1st play  : {ms(turn.first_play)}",
        f"  chunks    : {turn.chunks}",
    ]
    log("METRICS", "\n".join(lines))


# ----------------------------------------------------------------------------
# Startup
# ----------------------------------------------------------------------------
def pick_model(client, requested: str | None) -> str:
    if requested:
        return requested
    installed: list[str] = []
    try:
        resp = client.list()
        for m in getattr(resp, "models", None) or resp.get("models", []):
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None) \
                   or (m.get("name") if isinstance(m, dict) else None)
            if name:
                installed.append(name)
    except Exception as e:
        log("WARN", f"Could not list Ollama models ({e})")
    for pref in PREFERRED_MODELS:
        for name in installed:
            if name == pref or name.startswith(pref + ":") or name.split(":latest")[0] == pref:
                if pref not in PREFERRED_MODELS[:2]:
                    log("WARN", f"gemma2:2b / qwen2.5:3b not installed — falling back to {name}. "
                                f"Run `ollama pull gemma2:2b` for better Hindi.")
                return name
    raise SystemExit(f"None of {PREFERRED_MODELS} is installed in Ollama. Run: ollama pull gemma2:2b")


def main() -> None:
    ap = argparse.ArgumentParser(description="Low-latency Hindi caretaker voice pipeline")
    ap.add_argument("--model", help="Ollama model (default: first installed of %s)" % PREFERRED_MODELS)
    ap.add_argument("--silence-ms", type=int, default=150, help="end-of-speech silence (default 150)")
    ap.add_argument("--max-words", type=int, default=3, help="max words per TTS chunk (default 3)")
    default_backend = "mlx" if (sys.platform == "darwin" and platform.machine() == "arm64") else None
    ap.add_argument("--tts-backend", default=default_backend,
                    help="Kitten-TTS backend (default: 'mlx' on Apple silicon, else auto). Use 'auto' to force auto.")
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    import ollama
    from kittenml import KittenTTS

    t = now()
    stt_model = WhisperModel("small", device="cpu", compute_type="int8")
    stt_model.transcribe(np.zeros(MIC_SR, dtype=np.float32), language="hi", beam_size=1)
    log("INIT", "faster-whisper small (int8) loaded + warmed", since=t)

    t = now()
    backend = None if args.tts_backend in (None, "auto") else args.tts_backend
    try:
        tts = KittenTTS(TTS_MODEL, **({"backend": backend} if backend else {}))
    except Exception as e:
        if backend is None:
            raise
        log("WARN", f"Kitten-TTS backend '{backend}' failed ({e}); falling back to auto backend")
        backend = None
        tts = KittenTTS(TTS_MODEL)
    args.tts_backend = backend
    sr = int(getattr(tts, "sample_rate", 24000))
    tts.generate("नमस्ते", voice=TTS_VOICE)
    log("INIT", f"Kitten-TTS 2 loaded + warmed (sr={sr}, backend={args.tts_backend or 'auto'})", since=t)

    client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    model = pick_model(client, args.model)
    t = now()
    client.chat(model=model, messages=[{"role": "user", "content": "नमस्ते"}],
                keep_alive="30m", options={"num_predict": 1})
    log("INIT", f"Ollama model '{model}' loaded into memory", since=t)

    threads = [
        threading.Thread(target=player_thread, args=(sr,), name="D-player", daemon=True),
        threading.Thread(target=tts_thread, args=(tts, sr), name="C-tts", daemon=True),
        threading.Thread(target=brain_thread, args=(client, model, args.max_words), name="B-brain", daemon=True),
        threading.Thread(target=listener_thread, args=(stt_model, args.silence_ms), name="A-listener", daemon=True),
    ]
    for th in threads:
        th.start()

    try:
        while not stop_event.is_set() and all(th.is_alive() for th in threads):
            time.sleep(0.2)
    except KeyboardInterrupt:
        log("MAIN", "Ctrl+C — shutting down")
    finally:
        stop_event.set()
        for th in threads:
            th.join(timeout=3)


if __name__ == "__main__":
    main()
