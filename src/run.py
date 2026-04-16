"""
VLM Orchestrator — Starter Template

This is where you implement your pipeline. The harness feeds you frames
and audio in real-time. You call VLMs, detect events, and emit them back.

Usage:
    python src/run.py \\
        --procedure data/clip_procedures/CLIP.json \\
        --video path/to/Video_pitchshift.mp4 \\
        --output output/events.json \\
        --speed 1.0
"""

import json
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from dataclasses import asdict
import cv2
import requests
import numpy as np
import base64
import asyncio
import threading
import concurrent.futures
import io
import wave

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.harness import StreamingHarness
from src.data_loader import load_procedure_json, validate_procedure_format
from src.prompt import prompts
from stats.step_stats import step_stats




# ==========================================================================
# VLM API HELPER (provided — feel free to modify)
# ==========================================================================

def call_vlm(
    api_key: str,
    frame_base64: str,
    prompt: str,
    model: str = "google/gemma-4-26b-a4b-it", #"google/gemini-2.5-flash",
    stream: bool = False,
) -> str:
    """
    Call a VLM via OpenRouter.

    Args:
        api_key: OpenRouter API key
        frame_base64: Base64-encoded JPEG frame
        prompt: Text prompt
        model: OpenRouter model string
        stream: If True, use streaming (SSE) responses for lower time-to-first-token

    Returns:
        Model response text
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcor-labs/vlm-orchestrator-eval",
        "X-Title": "VLM Orchestrator Evaluation",
    }
    payload = {
        "model": model,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    },
                ],
            }
        ],
    }

    if stream:
        # Streaming: read SSE chunks as they arrive
        resp = requests.post(url, json=payload, headers=headers, stream=True, timeout=30)
        resp.raise_for_status()
        full_text = ""
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        full_text += delta["content"]
                except (json.JSONDecodeError, KeyError):
                    pass
        return full_text
    else:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


# ==========================================================================
# STT API HELPER (provided — feel free to modify)
# ==========================================================================
def pcm_to_wav_bytes(pcm_bytes):
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(16000)
        wf.writeframes(pcm_bytes)
    return buffer.getvalue()

def call_stt(
    api_key: str,
    audio_data,
    model: str = "google/gemini-2.0-flash-001", #"google/gemini-2.5-flash-lite", #"google/gemini-2.5-flash",#"mistralai/voxtral-small-24b-2507", #"google/gemini-2.5-flash",
    stream: bool = False,
) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/alcorlabs/realtime-vlm-playground",
        "X-Title": "Realtime VLM Playground - STT",
    }
    audio_b64 = base64.b64encode(audio_data).decode("utf-8")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an accurate transcriber. Transcribe the spoken audio verbatim. Do not guess or hallucinate. If unclear, noisy, or silent ONLY return 'silence'"}, #an empty string
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,  # Use the encoded string, not raw bytes
                            "format": "wav"   # Gemini supports pcm16 directly
                        }
                    }
                ]
            }
        ]
    }

    try:
        # Use json=payload which handles the dictionary-to-JSON conversion
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
        
        if resp.status_code != 200:
            print(f"STT API Error Details: {resp.text}")
            
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        if content:
            return content.strip()
    except Exception as e:
        print(f"[STT Error] {e}")
        return ""
    


# ==========================================================================
# Step State Manager
# ==========================================================================

class StepStateManager:
    def __init__(self, procedure: dict):
        self.procedure = procedure
        self.steps = sorted(procedure["steps"], key=lambda x: x["step_id"])

        self.current_step_id = 1
        self.completed_steps: list[int] = []
        self.events_history = []

        self.step_start_time = 0.0  

    # --- Step Access ---
    def get_current_step(self):
        return self.steps[self.current_step_id - 1]
        

    def get_current_step_id(self):
        return self.current_step_id

    def get_next_expected_step(self):
        if self.current_step_id < len(self.steps):
            step = self.steps[self.current_step_id]
            return step["description"]
        else:
            return "No more steps"

    def update_step(self,):
        """
        probably a more general update step doing current one step or jump update
        """
        pass

    # --- Step Progression (STRICT CONTROL) ---
    def complete_current_step(self, timestamp: float):
        """
        Only valid way to advance step.
        """
        if self.current_step_id in self.completed_steps:
            return False  # already done
        if self.current_step_id > len(self.steps):
            return False
        self.completed_steps.append(self.current_step_id)

        self.add_event(
            event="step_completion",
            timestep=timestamp,
            step_id=self.current_step_id,
        )

        self.current_step_id += 1
        self.step_start_time = timestamp  #reset timer

        return True

    # --- Recovery Jump (CONTROLLED) ---
    def jump_to_step(self, step_id: int, timestamp: float):
        """
        Used ONLY in recovery mode.
        """
        if step_id <= self.current_step_id:
            return False

        # limit jump size (important)
        if step_id > self.current_step_id + 3:
            return False

        self.current_step_id = step_id
        self.step_start_time = timestamp

        return True

    # --- Events ---
    def add_event(self, event, timestep):
        if event == "error_detected":
            desc = f"Error at step {self.current_step_id}"
        elif event == "step_completion":
            desc = f"Step {self.current_step_id} completed"
        else:
            desc = "Unknown event"

        self.events_history.append((timestep, event, self.current_step_id, desc))

    def get_event_history(self, max_items=5):
        """
        Keep history SHORT (important for prompt quality)
        """
        recent = self.events_history[-max_items:]
        message = ""
        for t, e, step_id, desc in recent:
            if e != "error_detected":
                message += f"Time: {t:.1f}, {desc}\n"
        return message

    # --- Timing ---
    def get_time_in_step(self, current_time: float):
        return current_time - self.step_start_time

    # --- Prompt Context ---
    def get_prompt_context(self, mode="strict") -> str:
        step = self.get_current_step()
        if mode == "strict":
            return json.dumps({
                "procedure_title": self.procedure.get("task_name", "Unknown"),
                "current_step_id": self.current_step_id,
                "current_step_description": step["description"] if step else "None",
                "next_step_description": self.get_next_expected_action(),
            }, indent=2)
        else:
            return json.dumps({
                "procedure_title": self.procedure.get("task_name", "Unknown"),
                "current_step_id": self.current_step_id,
                "steps": self.steps,
            }, indent=2)
            

def decide_mode(
    current_step_id: int,
    time_in_step: float,
    step_time_stats: dict= step_stats,
):
    """
    Realtime-safe mode decision using only per-step timing.

    Returns:
        mode: "strict" | "watchful" | "recovery"
        debug: dict
    """

    stats = step_time_stats.get(str(current_step_id), None)

    if stats is None:
        return "strict", {}

    mean = stats["mean"]
    std = max(stats["std"], 1e-3)

    # Optional: clamp extreme std (important for your data)
    std = min(std, mean * 2)

    z_score = (time_in_step - mean) / std

    # --- thresholds ---
    Z_STRICT = 1.0
    Z_WATCHFUL = 2.0

    if z_score < Z_STRICT:
        mode = "strict"

    elif z_score < Z_WATCHFUL:
        mode = "watchful"

    else:
        mode = "recovery"

    debug = {
        "z_score": z_score,
        "time_in_step": time_in_step,
        "expected_mean": mean,
        "std": std,
    }

    return mode, debug

# ==========================================================================
# Helper function - getting procedure description
# ==========================================================================

def get_desc(procedure : dict) -> str:
    name = procedure.get("task_name", "Unknown task")
    desc = procedure.get("description", "No description provided.")
    safety = procedure.get("safety_notes", "None")

    steps_list = procedure.get("steps", [])
    formatted_steps = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps_list)])

    return f"""
        Procedure name: {name}
        Description: {desc}
        Safety notes: {safety}
        STEPS:
            {formatted_steps}
        """


# ==========================================================================
# YOUR PIPELINE — IMPLEMENT THESE CALLBACKS
# ==========================================================================

class Pipeline:
    """
    Your VLM orchestration pipeline.

    The harness calls on_frame() and on_audio() in real-time as the video plays.
    When you detect an event, call self.harness.emit_event({...}).

    Key design decisions you need to make:
    - Which frames to send to the VLM (not every frame — budget is limited)
    - Whether/how to use audio (speech-to-text for instructor corrections?)
    - Which model to use and when (cheap for easy frames, expensive for hard ones?)
    - How to track procedure state (current step, completed steps)
    - How to generate spoken responses for errors
    """

    def __init__(self, harness: StreamingHarness, api_key: str, procedure: Dict[str, Any]):
        self.harness = harness
        self.api_key = api_key
        self.procedure = procedure
        self.task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
        self.steps = procedure["steps"]
        
        self.state = StepStateManager(procedure)
        self.last_frame = None
        self.last_vlm_call_time = 0.0
        self.audio_buffer = [] #list of (timestamp, transcript)
        self.system_prompt = prompts["v1"]
        self.desc_history = []
        self.proc_context = get_desc(procedure=procedure)
        
        self.vlm_calls = 0
        self.strict_calls = 0 #strict cheap vlm calls
        self.exp_calls = 0      # expensive vlm model calls.

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._pending_futures = []
        self._last_submitted_time = 0.0
        self._state_lock = threading.Lock()

        #audio
        self.audio_accumulator = b""
        self._audio_lock = threading.Lock()
        self.last_stt_call_time = 0.0

        #error
        self.last_error_time = 0.0
        self.last_error_step = -1
        self.ERROR_COOLDOWN = 3.0
        # TODO: Initialize your pipeline state here
        # Examples:
        #   self.current_step = 0
        #   self.completed_steps = set()
        #   self.frame_buffer = []
        #   self.last_activity_time = 0
        #   self.api_calls = 0
        #   self.total_cost = 0

    def on_frame(self, frame: np.ndarray, timestamp_sec: float, frame_base64: str):
        """
        Called by the harness for each video frame.

        Args:
            frame: BGR numpy array (raw frame)
            timestamp_sec: Current video timestamp
            frame_base64: Pre-encoded JPEG base64 string (ready for VLM API)

        TODO: Implement your frame processing logic.
        When you detect an event, call:
            self.harness.emit_event({
                "timestamp_sec": timestamp_sec,
                "type": "step_completion",  # or "error_detected" or "idle_detected"
                "step_id": 1,
                "confidence": 0.9,
                "description": "...",
                "source": "video",
                "vlm_observation": "...",
                # For errors, also include:
                "spoken_response": "Stop — you need to turn off the power first.",
            })
        """
        if self.last_frame is not None:
            diff = cv2.absdiff(self.last_frame, frame)
            motion_score = cv2.mean(diff)[0]
            significant_change = motion_score > 15.0
        else:
            significant_change = True
        self.last_frame = frame.copy()
        # Throttle VLM calls to avoid rate limits: require a minimum
        # interval between calls even if significant motion is detected.
        min_interval = 2.0
        should_call = (
            significant_change and
            (timestamp_sec - self.last_vlm_call_time > min_interval)
            ) or (timestamp_sec - self.last_vlm_call_time >3.0)
        
        if should_call:
            self.last_vlm_call_time = timestamp_sec
            self.vlm_calls +=1
            mode, _ = decide_mode(self.state.current_step_id, self.state.get_time_in_step(timestamp_sec))
            
            if mode == "strict":
                self.strict_calls +=1
            else:
                self.exp_calls +=1
            #snapshot shared state before handing off to worker
            with self._state_lock:
                snapshot_step_id    = self.state.current_step_id
                snapshot_step_time  = self.state.step_start_time
                snapshot_obs        = list(self.desc_history[-3:])
                snapshot_events     = self.state.get_event_history(max_items=3)
                snapshot_speech    = [
                    f"{t:.1f}: {text}"
                    for t, text in self.audio_buffer
                    if t > timestamp_sec - 15.0
                ]
            self._executor.submit(
                self._call_and_emit,
                frame_base64,
                timestamp_sec,
                snapshot_step_id,
                snapshot_step_time,
                snapshot_obs,
                snapshot_events,
                snapshot_speech,
                mode
            )
    
    def _call_and_emit(self, 
                       frame_base64: str, 
                       timestamp_sec: float,
                       step_id: int,
                       step_start_time: float,
                       obs_history: list,
                       events_history: str,
                       snapshot_speech: list,
                       mode: str,
                       ):
        """
            Runs in background thread. Does VLM call + emits event.
        """
        try:
            response = self._call_vlm(
                frame_base64,
                timestamp_sec,
                step_id,
                step_start_time,
                obs_history,
                events_history,
                snapshot_speech,
                mode
            )
        except Exception as e:
            print(f" [pipeline] VLM call failed: {e}")
            return
        
        if response is None:
            return
        
        with self._state_lock:
            raw_obs = response['observation'].split('.')[0] + '.'
            self.desc_history.append(
                f"time= {timestamp_sec} seconds: {raw_obs}"
            )
            #potentially cap the size of desc_history to 10 or 20 to not grow forever
            if response["event_type"] == "step_completion":
                if self.state.current_step_id == step_id: #gurad against stale worker
                    if self.state.complete_current_step(timestamp_sec):
                        self.harness.emit_event({
                            "timestamp_sec": timestamp_sec,
                            "type": "step_completion",  # or "error_detected" or "idle_detected"
                            "step_id": step_id,
                            "confidence": response["confidence"],
                            "description": response["description"],
                            "source": "video",
                            "vlm_observation": response["observation"],
                        })
                    
            elif response["event_type"] == "error_detected":
                    time_since_last_error = timestamp_sec - self.last_error_time
                    if time_since_last_error < self.ERROR_COOLDOWN:
                        # still in cooldown, skip emission
                        pass
                    else:
                        self.last_error_time = timestamp_sec
                        self.state.add_event(response["event_type"], timestamp_sec)
                        self.harness.emit_event({
                        "timestamp_sec": timestamp_sec,
                        "type": "error_detected",  # or "error_detected" or "idle_detected"
                        "step_id": step_id,
                        "confidence": response["confidence"],
                        "description": response["description"],
                        "source": "video",
                        "vlm_observation": response["observation"],
                        "spoken_response": response["speech"],
                        })
                    
        

    def on_audio(self, audio_bytes: bytes, start_sec: float, end_sec: float):
        """
        Called by the harness for each audio chunk.

        Args:
            audio_bytes: Raw PCM audio (16kHz, mono, 16-bit)
            start_sec: Chunk start time in video
            end_sec: Chunk end time in video

        TODO: Implement your audio processing logic.
        Consider: speech-to-text, keyword detection, silence detection.
        The instructor's verbal corrections are a strong signal for errors.
        """
        CHUNK_SECONDS = 5.0
        BYTES_PER_SECOND = 16000 * 2  # 16kHz * 16-bit
        CHUNK_SIZE = int(CHUNK_SECONDS * BYTES_PER_SECOND)
        with self._audio_lock:
            self.audio_accumulator += audio_bytes

            if len(self.audio_accumulator) < CHUNK_SIZE:
                return
            
            chunk = self.audio_accumulator[:CHUNK_SIZE]

            self.audio_accumulator = self.audio_accumulator[CHUNK_SIZE:]


        self._stt_in_flight = True

        with self._state_lock:
            step_id_snapshot = self.state.current_step_id

        self._executor.submit(
            self._stt_worker_thread,
            chunk,
            end_sec,
            step_id_snapshot
        )

        # 3. Trigger every ~1 second (assuming the harness sends chunks often)
        # Check if 1 second has passed since the last STT call
        # if end_sec - self.last_stt_call_time >= 1.0:
        #     self.last_stt_call_time = end_sec
            
        #     with self._audio_lock:
        #         snapshot_audio = self.audio_accumulator
            
        #     with self._state_lock:
        #         snapshot_step_id = self.state.current_step_id

        #     # Send to worker
        #     self._executor.submit(
        #         self._stt_worker_thread, 
        #         snapshot_audio, 
        #         end_sec, 
        #         snapshot_step_id
        #     )
        #print(self.audio_buffer)
       

    def _call_vlm(
            self, 
            frame_base64: str, 
            timestamp_sec: float,
            step_id: int,
            step_start_time: float,
            obs_history: list,
            events_history: str,
            snapshot_speech: list,
            mode: str,
        ):
        if mode == "strict":
            task_description = json.dumps({
                "procedure_title": self.procedure.get("task_name", "Unknown"),
                "current_step_id": step_id,
                "steps": self.state.steps,
                # "current_step_description": self.state.steps[step_id - 1]["description"],
                # "next_step_description": (
                #     self.state.steps[step_id]["description"]
                #     if step_id < len(self.state.steps) else "No more steps"
                # ),
            }, indent=2)
            model = "google/gemini-2.5-flash"
        else:# mode == "watchfull":
            task_description = self.state.get_prompt_context("watchfull")
            model = "openai/gpt-5.2"

        recent_audio = self.audio_buffer[-4:]
        recent_transcript = "\n".join(snapshot_speech) if snapshot_speech else "No recent speech."#" ".join([f"{time}: {t}\n" for time, t in recent_audio])
        #recent_transcript = recent_transcript.split(". ")[-3:]
        recent_obs = self.desc_history[-5:]
        obs_str = "\n".join(obs_history)
        #print("here2")
        prompt = self.system_prompt.format(
                task_description = task_description,
                seconds = timestamp_sec,
                #cur_step = self.state.get_current_step(),
                events_history = events_history,
                obs_history = obs_str,
                speech = recent_transcript,
        )
        print("*****", prompt)
        raw = call_vlm(
            self.api_key,
            frame_base64,
            prompt,
            stream=True,
            model=model
        )
        print(f"in {mode} call_vlm: ", raw)
        # The model is instructed to return strict JSON, but the API
        # returns a string. Parse JSON safely and provide a sensible
        # fallback if parsing fails so downstream code can index keys.
        if isinstance(raw, str):
            try:
                parsed = json.loads(raw)
                return parsed
            except json.JSONDecodeError:
                # Try to extract a JSON object substring if the model
                # included extra text or formatting around the JSON.
                s = raw
                start = s.find("{")
                end = s.rfind("}")
                if start != -1 and end != -1 and end > start:
                    try:
                        parsed = json.loads(s[start:end+1])
                        return parsed
                    except json.JSONDecodeError:
                        pass
                # Fallback: return a minimal structured dict so callers
                # can safely read keys.
                return {
                    "observation": raw.strip(),
                    "event_type": "none",
                    "confidence": 0.0,
                    "description": "",
                    "speech": "",
                }

        if isinstance(raw, dict):
            return raw

        # Any other unexpected type -> stringify
        return {
            "observation": str(raw),
            "event_type": "none",
            "confidence": 0.0,
            "description": "",
            "speech": "",
        }

    def _stt_worker_thread(self, audio_data: bytes, timestamp: float, step_id: int):
        try:
            wav_bytes = pcm_to_wav_bytes(audio_data)
            transcript = call_stt(self.api_key, wav_bytes)
            # try:
            #     data = json.loads(raw)
            #     transcript = data.get("text", "").strip()
            # except Exception:
            #     return
            if not transcript:
                return
            transcript = transcript.strip()

            # Ignore garbage / tiny outputs
            if len(transcript) < 3:
                return

            with self._state_lock:
                if not self.audio_buffer:
                    self.audio_buffer.append((timestamp, transcript))
                    return

                last_ts, last_text = self.audio_buffer[-1]

                # ✅ Dedup logic (cleaner)
                if transcript == last_text:
                    return

                if transcript.startswith(last_text):
                    # model extended previous phrase
                    self.audio_buffer[-1] = (timestamp, transcript)

                elif last_text.endswith(transcript):
                    # overlapping partial repeat → ignore
                    return

                else:
                    # new phrase
                    self.audio_buffer.append((timestamp, transcript))

        except Exception as e:
            print(f"!!! [STT Error]: {e}")

        finally:
            self._stt_in_flight = False



# ==========================================================================
# MAIN ENTRY POINT
# ==========================================================================

def main():
    parser = argparse.ArgumentParser(description="VLM Orchestrator Pipeline")
    parser.add_argument("--procedure", required=True, help="Path to procedure JSON")
    parser.add_argument("--video", required=True, help="Path to video MP4 (with audio)")
    parser.add_argument("--output", default="output/events.json", help="Output JSON path")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed (1.0 = real-time, 2.0 = 2x, etc.)")
    parser.add_argument("--frame-fps", type=float, default=2.0,
                        help="Frames per second delivered to pipeline (default: 2)")
    parser.add_argument("--audio-chunk-sec", type=float, default=5.0,
                        help="Audio chunk duration in seconds (default: 5)")
    parser.add_argument("--api-key", help="OpenRouter API key (or set OPENROUTER_API_KEY)")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs only")
    args = parser.parse_args()

    # Load procedure
    print("=" * 60)
    print("  VLM ORCHESTRATOR")
    print("=" * 60)
    print()

    procedure = load_procedure_json(args.procedure)
    validate_procedure_format(procedure)
    task_name = procedure.get("task") or procedure.get("task_name", "Unknown")
    print(f"  Procedure: {task_name} ({len(procedure['steps'])} steps)")
    print(f"  Video:     {args.video}")
    print(f"  Speed:     {args.speed}x")
    print()

    if args.dry_run:
        if not Path(args.video).exists():
            print(f"  WARNING: Video not found: {args.video}")
            print("  [DRY RUN] Procedure validated. Video not checked (file missing).")
        else:
            print("  [DRY RUN] Inputs validated. Skipping pipeline.")
        return

    if not Path(args.video).exists():
        print(f"  ERROR: Video not found: {args.video}")
        sys.exit(1)

    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("  ERROR: Set OPENROUTER_API_KEY or pass --api-key")
        sys.exit(1)

    # Create harness and pipeline
    harness = StreamingHarness(
        video_path=args.video,
        procedure_path=args.procedure,
        speed=args.speed,
        frame_fps=args.frame_fps,
        audio_chunk_sec=args.audio_chunk_sec,
    )

    pipeline = Pipeline(harness, api_key, procedure)

    # Register callbacks
    harness.on_frame(pipeline.on_frame)
    harness.on_audio(pipeline.on_audio)

    # Run
    results = harness.run()

    # Save
    harness.save_results(results, args.output)

    print()
    print(f"  Output: {args.output}")
    print(f"  Events: {len(results.events)}")
    print()

    if not results.events:
        print("  WARNING: No events detected. Implement Pipeline.on_frame() and Pipeline.on_audio().")


if __name__ == "__main__":
    main()
