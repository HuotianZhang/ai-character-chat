"""
AI Character Chat System - Main Entry Point
Pure Python standard library HTTP server, no Flask required.
Integrates: TimeController, ProactiveEventSystem, ConversationEngine
"""
import os
import sys
import json
import threading
import traceback
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

# Ensure project directory is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, CHARACTER_FILE, FLASK_HOST, FLASK_PORT
from character_generator import generate_character, save_character, load_character
from character_state import CharacterState
from conversation_engine import ConversationEngine
from time_controller import TimeController
from proactive_events import ProactiveEventSystem

# Global instances
engine = None
time_ctrl = None
proactive_sys = None

# Thread safety: lock for all state-mutating operations (chat, inject, etc.)
# This ensures that concurrent requests don't corrupt character state.
_engine_lock = threading.Lock()

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")


def _init_subsystems(character_state, is_new=False):
    """Initialize TimeController and ProactiveEventSystem alongside ConversationEngine."""
    global engine, time_ctrl, proactive_sys

    engine = ConversationEngine(character_state)

    # Time controller
    time_ctrl = TimeController(speed_multiplier=60.0, start_hour=9)
    if not is_new:
        time_ctrl.load()

    # Proactive events
    proactive_sys = ProactiveEventSystem(character_state, time_ctrl)

    # Sync storyline with time controller
    character_state.storyline.current_day = time_ctrl.get_virtual_day()
    character_state.storyline.current_time_slot = time_ctrl.get_time_slot()


class ChatHandler(BaseHTTPRequestHandler):
    """HTTP Request Handler"""

    def log_message(self, format, *args):
        """Simplified logging output"""
        msg = format % args
        if "/api/" in msg or "GET / " in msg:
            print(f"  [{self.log_date_time_string()}] {msg}")

    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def _send_html(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(content.encode("utf-8"))
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404 Not Found")

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 0:
            body = self.rfile.read(length)
            return json.loads(body.decode("utf-8"))
        return {}

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "":
            self._send_html(os.path.join(TEMPLATE_DIR, "chat.html"))
        elif path == "/api/status":
            self._handle_status()
        elif path == "/api/character":
            self._handle_character_info()
        elif path == "/api/time":
            self._handle_time_status()
        elif path == "/api/events":
            self._handle_poll_events()
        elif path == "/api/debug/state":
            self._handle_debug_state()
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404")

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/init":
            self._handle_init()
        elif path == "/api/chat":
            self._handle_chat()
        elif path == "/api/advance_time":
            self._handle_advance_time()
        elif path == "/api/time/speed":
            self._handle_set_speed()
        elif path == "/api/time/pause":
            self._handle_time_pause()
        elif path == "/api/time/resume":
            self._handle_time_resume()
        elif path == "/api/time/jump":
            self._handle_time_jump()
        elif path == "/api/debug/inject":
            self._handle_debug_inject()
        elif path == "/api/debug/simulate_silence":
            self._handle_debug_simulate_silence()
        elif path == "/api/debug/evaluate_relationship":
            self._handle_debug_evaluate_relationship()
        else:
            self._send_json({"error": "Not found"}, 404)

    # ---- Core API Handlers ----

    def _handle_status(self):
        global engine, time_ctrl
        character_exists = os.path.exists(CHARACTER_FILE)
        character_loaded = engine is not None

        result = {
            "character_exists": character_exists,
            "character_loaded": character_loaded,
        }

        if character_loaded:
            char = engine.state.character
            name = char.get("基础信息", {}).get("名字", "Unknown")
            status = engine.state.get_status_summary()
            result["name"] = name
            result["status_line"] = status.get("mood", "Online")
            if time_ctrl:
                result["time"] = time_ctrl.get_status()

        self._send_json(result)

    def _handle_init(self):
        global engine, time_ctrl, proactive_sys
        data = self._read_body()
        action = data.get("action", "new")

        try:
            if action == "new":
                print("\n[Main] Starting to generate new character...")
                character = generate_character()
                save_character(character)

                state = CharacterState(character)
                _init_subsystems(state, is_new=True)

                print("[Main] Calling LLM to generate backstory and storyline...")
                character = engine.initialize_character()

                # Save time state
                time_ctrl.save()

                name = character.get("基础信息", {}).get("名字", "Unknown")
                backstory = character.get("人生经历", {})
                snapshot = backstory.get("current_situation", f"Hello, I am {name}")

                self._send_json({
                    "success": True,
                    "name": name,
                    "status_line": "Online",
                    "snapshot": snapshot,
                    "time": time_ctrl.get_status(),
                    "character_summary": {
                        "Basics": character.get("基础信息", {}),
                        "Aura": character.get("性格维度", {}).get("外在气场", ""),
                        "MBTI": character.get("衍生属性", {}).get("MBTI", ""),
                    }
                })

            elif action == "load":
                character = load_character()
                if not character:
                    self._send_json({"success": False, "error": "No saved character found"})
                    return

                state = CharacterState(character)
                state.load()
                _init_subsystems(state, is_new=False)

                name = character.get("基础信息", {}).get("名字", "Unknown")
                status = engine.state.get_status_summary()

                self._send_json({
                    "success": True,
                    "name": name,
                    "status_line": status.get("mood", "Online"),
                    "time": time_ctrl.get_status(),
                })

        except Exception as e:
            traceback.print_exc()
            self._send_json({"success": False, "error": str(e)})

    def _handle_chat(self):
        global engine, time_ctrl, _engine_lock

        if engine is None:
            self._send_json({"error": "Character not initialized"}, 400)
            return

        data = self._read_body()
        message = data.get("message", "").strip()

        if not message:
            self._send_json({"error": "Message cannot be empty"}, 400)
            return

        # Acquire lock — if another chat request is in-flight (LLM call),
        # this will block until that one finishes. The frontend queues messages
        # client-side and sends them sequentially, so this mostly protects
        # against edge cases (double-click, race conditions).
        with _engine_lock:
            try:
                # Sync time before processing
                if time_ctrl:
                    engine.state.storyline.current_day = time_ctrl.get_virtual_day()
                    engine.state.storyline.current_time_slot = time_ctrl.get_time_slot()

                # Notify silence monitor that user sent a message
                if proactive_sys:
                    proactive_sys.notify_user_message()

                # The message may contain multiple lines if the frontend batched them
                # (user typed several messages while LLM was thinking).
                # We pass the combined text directly — the LLM sees it as one turn,
                # which is natural (like sending multiple chat bubbles quickly).
                result = engine.chat(message)

                # Add time info to response
                if time_ctrl:
                    result["time"] = time_ctrl.get_status()
                    time_ctrl.save()

                self._send_json(result)
            except Exception as e:
                traceback.print_exc()
                self._send_json({
                    "reply": "[System Error]",
                    "messages": [{"type": "text", "content": f"[System: {str(e)[:100]}]"}],
                    "status": {},
                    "inner_thought": "",
                })

    def _handle_character_info(self):
        global engine, time_ctrl
        if engine is None:
            self._send_json({"error": "Character not loaded"})
            return

        result = {
            "character": engine.state.character,
            "status": engine.state.get_status_summary(),
            "emotion": engine.state.emotion.to_dict(),
            "pressure": engine.state.pressure.to_dict(),
            "affinity": engine.state.affinity.to_dict(),
            "memory_count": {
                "short_term": len(engine.state.memory.short_term),
                "long_term": len(engine.state.memory.long_term),
                "triggers": len(engine.state.memory.emotional_memory),
            },
            "storyline": engine.state.storyline.to_dict(),
        }
        if time_ctrl:
            result["time"] = time_ctrl.get_status()
        if proactive_sys:
            result["silence"] = proactive_sys.get_silence_status()

        self._send_json(result)

    # ---- Time Control Handlers ----

    def _handle_time_status(self):
        global time_ctrl
        if time_ctrl is None:
            self._send_json({"error": "Time controller not initialized"})
            return
        self._send_json(time_ctrl.get_status())

    def _handle_set_speed(self):
        global time_ctrl
        if time_ctrl is None:
            self._send_json({"error": "Time controller not initialized"})
            return

        data = self._read_body()
        speed = data.get("speed", 60.0)

        try:
            speed = float(speed)
            time_ctrl.set_speed(speed)
            time_ctrl.save()
            self._send_json(time_ctrl.get_status())
        except (ValueError, TypeError):
            self._send_json({"error": "Invalid speed value"}, 400)

    def _handle_time_pause(self):
        global time_ctrl
        if time_ctrl is None:
            self._send_json({"error": "Time controller not initialized"})
            return
        time_ctrl.pause()
        time_ctrl.save()
        self._send_json(time_ctrl.get_status())

    def _handle_time_resume(self):
        global time_ctrl
        if time_ctrl is None:
            self._send_json({"error": "Time controller not initialized"})
            return
        time_ctrl.resume()
        time_ctrl.save()
        self._send_json(time_ctrl.get_status())

    def _handle_time_jump(self):
        global time_ctrl, engine
        if time_ctrl is None:
            self._send_json({"error": "Time controller not initialized"})
            return

        data = self._read_body()

        # Jump modes: "slot" (next time slot), "hours" (forward N hours)
        if "slot" in data:
            time_ctrl.jump_to_slot(data["slot"])
        elif "hours" in data:
            try:
                hours = float(data["hours"])
                time_ctrl.jump_forward_hours(hours)
            except (ValueError, TypeError):
                self._send_json({"error": "Invalid hours value"}, 400)
                return
        else:
            self._send_json({"error": "Provide 'slot' or 'hours'"}, 400)
            return

        # Sync storyline
        if engine:
            engine.state.storyline.current_day = time_ctrl.get_virtual_day()
            engine.state.storyline.current_time_slot = time_ctrl.get_time_slot()
            engine.state.save()

        time_ctrl.save()
        self._send_json(time_ctrl.get_status())

    def _handle_advance_time(self):
        """Legacy advance_time: jump to next time slot."""
        global time_ctrl, engine
        if engine is None:
            self._send_json({"error": "Character not loaded"})
            return

        if time_ctrl:
            # Use time controller to advance
            slots = ["morning", "afternoon", "evening", "night"]
            current = time_ctrl.get_time_slot()
            idx = slots.index(current) if current in slots else 0
            next_slot = slots[(idx + 1) % len(slots)]
            time_ctrl.jump_to_slot(next_slot)
            engine.state.storyline.current_day = time_ctrl.get_virtual_day()
            engine.state.storyline.current_time_slot = time_ctrl.get_time_slot()
        else:
            engine.state.storyline.advance_time()

        # Apply mood effects from events (uses legacy compat method)
        events = engine.state.storyline.get_current_events()
        for event in events:
            impact = event.get("mood_impact", 0)
            label = event.get("mood_label", "")
            if impact != 0:
                engine.state.emotion.apply_legacy_stimulus(label, impact, reason=event.get("event", ""))

        engine.state.save()
        if time_ctrl:
            time_ctrl.save()

        result = {
            "day": engine.state.storyline.current_day + 1,
            "time_slot": engine.state.storyline.current_time_slot,
            "events": events,
            "status": engine.state.get_status_summary(),
        }
        if time_ctrl:
            result["time"] = time_ctrl.get_status()

        self._send_json(result)

    # ---- Proactive Events ----

    def _handle_poll_events(self):
        """Poll for proactive messages from the character."""
        global engine, proactive_sys, _engine_lock

        if engine is None or proactive_sys is None:
            self._send_json({"events": []})
            return

        # Use non-blocking lock attempt: if chat is in-flight, skip this poll cycle
        # rather than blocking (polls happen every 3s, we can try again next time)
        if not _engine_lock.acquire(blocking=False):
            self._send_json({"events": []})
            return

        try:
            pending = proactive_sys.get_pending(engine)
            self._send_json({"events": pending})
        except Exception as e:
            print(f"[Events] Error polling events: {e}")
            traceback.print_exc()
            self._send_json({"events": []})
        finally:
            _engine_lock.release()


    # ---- Debug API Handlers ----

    def _handle_debug_state(self):
        """GET /api/debug/state — Full state dump for debugging emotional realism."""
        global engine, time_ctrl, proactive_sys
        if engine is None:
            self._send_json({"error": "Character not loaded"})
            return

        from character_state import TensionDetector

        state = engine.state
        tensions = TensionDetector.detect(state.emotion)

        result = {
            "emotion_axes": {k: round(v, 4) for k, v in state.emotion.axes.items()},
            "mood_scalar": round(state.emotion.get_mood_scalar(), 2),
            "mood_desc": state.emotion.get_mood_description(),
            "dominant_emotions": [(ax, round(v, 3)) for ax, v in state.emotion.get_dominant(5)],
            "total_energy": round(state.emotion.total_energy(), 3),
            "tension_states": [{"name": t["name"], "label": t["label_cn"], "behavior": t["behavior"]} for t in tensions],
            "pressure": {
                "channels": state.pressure.pressure,
                "thresholds": state.pressure.threshold,
                "burst_count": state.pressure.burst_count,
                "profile": state.pressure.profile_key,
            },
            "affinity": state.affinity.affinity,
            "special_affinity": state.affinity.special_affinity,
            "interaction_count": state.interaction_count,
            "recent_stimuli": state.emotion.stimulus_history[-10:],
            "relationship_judge": state.judge.to_dict(),
        }
        if proactive_sys:
            result["silence"] = proactive_sys.get_silence_status()
        if time_ctrl:
            result["time"] = time_ctrl.get_status()

        self._send_json(result)

    def _handle_debug_inject(self):
        """POST /api/debug/inject — Inject emotion/pressure changes for testing.

        Body: {
            "emotion": {"joy": 0.3, "anger": 0.2},   // optional
            "pressure": {"no_reply": 0.5},              // optional
            "affinity_delta": 5,                        // optional
            "reason": "debug test"                      // optional
        }
        """
        global engine
        if engine is None:
            self._send_json({"error": "Character not loaded"}, 400)
            return

        data = self._read_body()
        reason = data.get("reason", "debug_inject")

        # Inject emotions
        emotion_changes = data.get("emotion", {})
        if emotion_changes:
            engine.state.emotion.apply_stimulus(emotion_changes, reason=reason)

        # Inject pressure
        pressure_changes = data.get("pressure", {})
        burst_results = []
        for channel, amount in pressure_changes.items():
            result = engine.state.pressure.record(channel, float(amount))
            if result and result.get("burst"):
                burst_results.append(result)
                # Apply burst emotions
                burst_emo = result.get("emotion_changes", {})
                if burst_emo:
                    engine.state.emotion.apply_stimulus(burst_emo, reason=f"burst_{channel}")

        # Inject affinity
        aff_delta = data.get("affinity_delta", 0)
        if aff_delta:
            engine.state.affinity.modify_affinity(int(aff_delta), reason=reason)

        engine.state.save()

        self._send_json({
            "success": True,
            "status": engine.state.get_status_summary(),
            "bursts": burst_results,
            "emotion_axes": {k: round(v, 3) for k, v in engine.state.emotion.axes.items()},
        })

    def _handle_debug_simulate_silence(self):
        """POST /api/debug/simulate_silence — Fast-forward silence effects.

        Body: {"minutes": 120}  // simulate 120 virtual minutes of silence
        """
        global engine, proactive_sys
        if engine is None or proactive_sys is None:
            self._send_json({"error": "Not initialized"}, 400)
            return

        data = self._read_body()
        minutes = float(data.get("minutes", 60))

        sm = proactive_sys.silence_monitor

        # Save original state for comparison
        before = {k: round(v, 4) for k, v in engine.state.emotion.axes.items()}
        before_pressure = round(engine.state.pressure.pressure.get("no_reply", 0), 4)

        # Simulate by manipulating the last_user_message time backward
        import datetime
        if sm.time_ctrl and sm.last_user_message_virtual:
            sm.last_user_message_virtual -= datetime.timedelta(minutes=minutes)
        sm.last_user_message_real -= (minutes * 60) / (sm.time_ctrl.speed_multiplier if sm.time_ctrl else 1)
        sm._last_phase_applied = 0
        sm._proactive_silence_sent = False

        # Run multiple ticks to process the silence effects
        events_generated = []
        for i in range(20):
            sm._silence_tick_count = i * 10  # simulate many ticks
            sm._last_tick_time = 0  # force tick to run
            event = sm.tick()
            if event:
                events_generated.append(event)

        after = {k: round(v, 4) for k, v in engine.state.emotion.axes.items()}
        after_pressure = round(engine.state.pressure.pressure.get("no_reply", 0), 4)

        # Calculate deltas
        deltas = {}
        for k in before:
            d = after[k] - before[k]
            if abs(d) > 0.001:
                deltas[k] = round(d, 4)

        engine.state.save()

        self._send_json({
            "success": True,
            "simulated_minutes": minutes,
            "before": before,
            "after": after,
            "deltas": deltas,
            "pressure_before": before_pressure,
            "pressure_after": after_pressure,
            "events_generated": len(events_generated),
            "status": engine.state.get_status_summary(),
            "silence": proactive_sys.get_silence_status(),
        })


    def _handle_debug_evaluate_relationship(self):
        """POST /api/debug/evaluate_relationship — Force a relationship evaluation.

        Body: {} (no params needed, uses current state + conversation history)
        """
        global engine
        if engine is None:
            self._send_json({"error": "Character not loaded"}, 400)
            return

        judge = engine.state.judge
        result = judge.evaluate(engine.conversation_history)
        engine.state.save()

        self._send_json({
            "success": True,
            "relationship_judge": result,
            "status": engine.state.get_status_summary(),
        })


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle each request in a new thread so the server doesn't block
    while waiting for LLM responses. Thread safety is handled by _engine_lock."""
    daemon_threads = True


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    port = FLASK_PORT
    host = FLASK_HOST

    print("=" * 60)
    print("  AI Character Chat System v2.5")
    print("  + EmotionVector + SilenceMonitor + RelationshipJudge")
    print("=" * 60)
    print(f"  Open browser:  http://localhost:{port}")
    print(f"  Debug panel:   Click the gear button in top right")
    print(f"  Character API: http://localhost:{port}/api/character")
    print(f"  Time API:      http://localhost:{port}/api/time")
    print(f"  Debug APIs:")
    print(f"    GET  /api/debug/state              Full emotion state dump")
    print(f"    POST /api/debug/inject              Inject emotion/pressure")
    print(f"    POST /api/debug/simulate_silence    Simulate no-reply time")
    print(f"    POST /api/debug/evaluate_relationship  Force relationship eval")
    print("=" * 60)
    print()

    server = ThreadedHTTPServer((host, port), ChatHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped")
        server.server_close()


if __name__ == "__main__":
    main()
