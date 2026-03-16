#!/usr/bin/env python3
"""
Debug Tools — Standalone module for testing emotional realism.

Run directly:  python debug_tools.py
Or import:     from debug_tools import EmotionDebugger

This module lets you:
  1. Simulate silence and see how emotions/pressure change per personality
  2. Test specific scenarios (repeated criticism, love-bombing, ghosting)
  3. Compare different attachment styles side-by-side
  4. Verify tension detection triggers correctly
  5. Stress-test pressure burst mechanics

No LLM or server needed — pure state-engine testing.
"""
import sys
import os
import json
import time

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from character_state import EmotionVector, TensionDetector, PressureAccumulator, CharacterState
from character_generator import generate_character, calculate_per_axis_decay_rates
from silence_monitor import SilenceMonitor


class EmotionDebugger:
    """Interactive debugger for emotional realism testing."""

    def __init__(self, character=None):
        if character is None:
            character = generate_character()
        self.character = character
        self.state = CharacterState(character)

        name = character.get("基础信息", {}).get("名字", "Unknown")
        attachment = character.get("性格维度", {}).get("依恋模式", "")
        cognitive = character.get("性格维度", {}).get("认知风格", "")
        print(f"\n{'='*60}")
        print(f"  Debug Character: {name}")
        print(f"  Attachment: {attachment}")
        print(f"  Cognitive:  {cognitive}")
        print(f"  Pressure profile: {self.state.pressure.profile_key}")
        print(f"{'='*60}\n")

    def show_state(self, label="Current State"):
        """Print full emotional state."""
        s = self.state
        axes = s.emotion.axes
        tensions = TensionDetector.detect(s.emotion)

        print(f"\n--- {label} ---")
        print(f"  Mood scalar: {s.emotion.get_mood_scalar():.1f}/10 ({s.emotion.get_mood_description()})")
        print(f"  Total energy: {s.emotion.total_energy():.2f}")
        print(f"  Axes:")
        for ax in EmotionVector.AXES:
            val = axes[ax]
            bar = "█" * int(val * 30) + "░" * (30 - int(val * 30))
            print(f"    {ax:12s} {bar} {val:.3f}")

        if tensions:
            print(f"  Tensions: {', '.join(t['label_cn'] for t in tensions)}")
        else:
            print(f"  Tensions: none")

        pressures = s.pressure.get_active_pressures()
        if pressures:
            print(f"  Pressures: {pressures}")
        print(f"  Affinity: {s.affinity.affinity}/100")
        print()

    def inject(self, emotion_changes=None, pressure_channel=None, pressure_amount=0.15,
               affinity_delta=0, reason="debug"):
        """Inject emotional stimulus manually."""
        if emotion_changes:
            self.state.emotion.apply_stimulus(emotion_changes, reason=reason)
            print(f"  [Inject] Emotion: {emotion_changes}")

        if pressure_channel:
            result = self.state.pressure.record(pressure_channel, pressure_amount)
            print(f"  [Inject] Pressure: {pressure_channel}+{pressure_amount:.2f}")
            if result and result.get("burst"):
                burst_emo = result.get("emotion_changes", {})
                self.state.emotion.apply_stimulus(burst_emo, reason=f"burst_{pressure_channel}")
                print(f"  *** BURST on {pressure_channel}! Burst #{result['burst_count']} ***")
                print(f"  *** Burst emotions: {burst_emo}")

        if affinity_delta:
            self.state.affinity.modify_affinity(affinity_delta, reason=reason)
            print(f"  [Inject] Affinity: {affinity_delta:+d}")

    def simulate_silence(self, minutes, tick_interval=10):
        """
        Simulate N virtual minutes of silence.
        Shows emotional state at each phase transition.
        """
        print(f"\n{'='*60}")
        print(f"  SIMULATING {minutes} MINUTES OF SILENCE")
        print(f"  Profile: {self.state.pressure.profile_key}")
        print(f"{'='*60}")

        self.show_state("Before Silence")

        # Create a mock time controller
        from time_controller import TimeController
        tc = TimeController(speed_multiplier=1.0)
        sm = SilenceMonitor(self.state, tc)
        sm.on_user_message()

        # Simulate by stepping through time
        import datetime
        elapsed = 0
        last_phase = 0

        while elapsed < minutes:
            step = min(tick_interval, minutes - elapsed)
            elapsed += step

            # Push silence backward
            sm.last_user_message_virtual -= datetime.timedelta(minutes=step)
            sm.last_user_message_real -= step * 60
            sm._last_tick_time = 0
            sm._silence_tick_count += 1

            event = sm.tick()

            # Detect phase changes
            current_phase = 0
            for i, threshold in enumerate(SilenceMonitor.PHASE_THRESHOLDS):
                if elapsed >= threshold:
                    current_phase = i

            if current_phase > last_phase:
                self.show_state(f"Phase {current_phase} (at {elapsed:.0f} min)")
                last_phase = current_phase

            if event:
                print(f"  >>> PROACTIVE TRIGGER at {elapsed:.0f}min: {event.get('inner_thought', '')}")

        self.show_state("After Silence")

    def simulate_scenario(self, scenario_name):
        """
        Run a predefined scenario to test emotional realism.
        """
        scenarios = {
            "repeated_no_reply": self._scenario_repeated_no_reply,
            "love_bomb": self._scenario_love_bomb,
            "slow_ghost": self._scenario_slow_ghost,
            "criticism_spiral": self._scenario_criticism_spiral,
            "push_pull": self._scenario_push_pull,
            "boundary_violation": self._scenario_boundary_violation,
        }

        if scenario_name not in scenarios:
            print(f"Unknown scenario. Available: {list(scenarios.keys())}")
            return

        print(f"\n{'='*60}")
        print(f"  SCENARIO: {scenario_name}")
        print(f"{'='*60}")

        scenarios[scenario_name]()

    def _scenario_repeated_no_reply(self):
        """User repeatedly ignores messages over several days."""
        print("  Day 1: Character sends message, user doesn't reply for 2 hours")
        self.simulate_silence(120)

        print("\n  User finally replies briefly, then goes silent again for 4 hours")
        self.state.emotion.apply_stimulus({"joy": 0.1, "anxiety": -0.05}, reason="user replied")
        self.state.pressure.pressure["no_reply"] *= 0.5  # partial relief
        self.show_state("After brief reply")
        self.simulate_silence(240)

        print("\n  Day 2: Same pattern — another 3 hours of silence")
        self.simulate_silence(180)
        self.show_state("FINAL — After repeated no-reply cycle")

    def _scenario_love_bomb(self):
        """User sends many warm/flirty messages in rapid succession."""
        print("  Rapid compliments and warm messages...")
        warmth_steps = [
            {"joy": 0.15, "trust": 0.05},
            {"joy": 0.12, "attachment": 0.08},
            {"joy": 0.10, "trust": 0.10, "attachment": 0.05},
            {"joy": 0.08, "attachment": 0.12},
            {"joy": 0.15, "trust": 0.08, "attachment": 0.10},
        ]
        for i, changes in enumerate(warmth_steps):
            self.inject(emotion_changes=changes, affinity_delta=3, reason=f"warm_msg_{i+1}")
        self.show_state("After love-bombing")

        print("  Then sudden silence...")
        self.simulate_silence(180)
        self.show_state("FINAL — Love bomb then silence")

    def _scenario_slow_ghost(self):
        """User gradually increases reply times until silence."""
        delays = [10, 30, 60, 120, 240]
        for i, delay in enumerate(delays):
            print(f"\n  Round {i+1}: User takes {delay} min to reply")
            self.simulate_silence(delay)
            # Brief relief from reply
            self.state.emotion.apply_stimulus(
                {"joy": max(0.02, 0.1 - i*0.02), "anxiety": -0.02},
                reason="eventual_reply"
            )
        self.show_state("FINAL — After slow ghosting pattern")

    def _scenario_criticism_spiral(self):
        """User makes several critical/dismissive comments."""
        criticisms = [
            ({"anger": 0.1, "sadness": 0.05}, "criticized", "that's not very smart"),
            ({"anger": 0.12, "sadness": 0.08, "trust": -0.05}, "criticized", "you always do this"),
            ({"anger": 0.15, "sadness": 0.10, "disgust": 0.05}, "criticized", "whatever, you wouldn't understand"),
            ({"anger": 0.10, "sadness": 0.12, "trust": -0.08}, "criticized", "I don't even know why I bother"),
        ]
        for emo, channel, reason in criticisms:
            self.inject(emotion_changes=emo, pressure_channel=channel, reason=reason)
        self.show_state("FINAL — After criticism spiral")

    def _scenario_push_pull(self):
        """Alternating warmth and coldness to test emotional whiplash."""
        steps = [
            ("warm", {"joy": 0.15, "trust": 0.1, "attachment": 0.08}),
            ("cold", {"sadness": 0.1, "anxiety": 0.08}),
            ("warm", {"joy": 0.12, "attachment": 0.1}),
            ("cold", {"anger": 0.1, "sadness": 0.08, "trust": -0.05}),
            ("warm", {"joy": 0.08, "trust": 0.05, "attachment": 0.12}),
            ("cold", {"anger": 0.12, "disgust": 0.05, "trust": -0.08}),
        ]
        for label, changes in steps:
            print(f"  {label.upper()}: {changes}")
            self.inject(emotion_changes=changes, reason=f"push_pull_{label}")
            self.show_state(f"After {label}")

    def _scenario_boundary_violation(self):
        """User pushes boundaries repeatedly."""
        steps = [
            ({"anger": 0.05, "anxiety": 0.05}, "boundary_pushed", 0.2, "asks something personal too early"),
            ({"anger": 0.08, "disgust": 0.05}, "boundary_pushed", 0.25, "insists despite deflection"),
            ({"anger": 0.12, "disgust": 0.10, "trust": -0.08}, "boundary_pushed", 0.3, "gets pushy about it"),
            ({"anger": 0.15, "disgust": 0.12, "trust": -0.10}, "boundary_pushed", 0.35, "crosses a clear line"),
        ]
        for emo, channel, amount, reason in steps:
            self.inject(emotion_changes=emo, pressure_channel=channel,
                       pressure_amount=amount, reason=reason)
        self.show_state("FINAL — After boundary violations")


def compare_attachment_styles(scenario_name="repeated_no_reply"):
    """
    Run the same scenario across all 4 attachment profiles,
    showing how each reacts differently.
    """
    from character_generator import generate_character

    profiles = {
        "anxious": "焦虑·依恋",
        "avoidant": "回避·安全",
        "secure": "安全·松弛自洽",
        "fearful": "恐惧回避·推拉矛盾",
    }

    results = {}
    for label, attachment in profiles.items():
        print(f"\n{'#'*60}")
        print(f"  ATTACHMENT PROFILE: {label} ({attachment})")
        print(f"{'#'*60}")

        # Generate character with specific attachment
        char = generate_character()
        char["性格维度"]["依恋模式"] = attachment

        dbg = EmotionDebugger(char)
        dbg.simulate_scenario(scenario_name)

        results[label] = {
            "axes": {k: round(v, 3) for k, v in dbg.state.emotion.axes.items()},
            "mood": round(dbg.state.emotion.get_mood_scalar(), 1),
            "tensions": [t["label_cn"] for t in TensionDetector.detect(dbg.state.emotion)],
            "pressures": dbg.state.pressure.get_active_pressures(),
        }

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY: {scenario_name}")
    print(f"{'='*60}")
    for label, data in results.items():
        print(f"\n  {label}:")
        print(f"    Mood: {data['mood']}/10")
        axes_str = ", ".join(f"{k}={v}" for k, v in data["axes"].items() if v >= 0.05)
        print(f"    Active: {axes_str or 'none'}")
        print(f"    Tensions: {data['tensions'] or 'none'}")
        print(f"    Pressures: {data['pressures'] or 'none'}")


def interactive_mode():
    """Run an interactive debug session."""
    print("\n=== EMOTION DEBUG INTERACTIVE MODE ===")
    print("Commands:")
    print("  new                     — Generate new character")
    print("  show                    — Show current state")
    print("  inject joy=0.3 anger=0.2  — Inject emotions")
    print("  pressure no_reply 0.3   — Add pressure")
    print("  silence 120             — Simulate 120 min silence")
    print("  scenario <name>         — Run scenario")
    print("  compare <scenario>      — Compare all attachment styles")
    print("  scenarios               — List available scenarios")
    print("  quit                    — Exit")
    print()

    dbg = EmotionDebugger()
    dbg.show_state()

    while True:
        try:
            cmd = input("debug> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not cmd:
            continue

        parts = cmd.split()
        action = parts[0].lower()

        if action == "quit" or action == "exit":
            break
        elif action == "new":
            dbg = EmotionDebugger()
            dbg.show_state()
        elif action == "show":
            dbg.show_state()
        elif action == "inject":
            changes = {}
            for part in parts[1:]:
                if "=" in part:
                    k, v = part.split("=")
                    changes[k] = float(v)
            if changes:
                dbg.inject(emotion_changes=changes)
                dbg.show_state()
            else:
                print("  Usage: inject joy=0.3 anger=0.2")
        elif action == "pressure":
            if len(parts) >= 3:
                channel = parts[1]
                amount = float(parts[2])
                dbg.inject(pressure_channel=channel, pressure_amount=amount)
                dbg.show_state()
            else:
                print("  Usage: pressure no_reply 0.3")
        elif action == "silence":
            minutes = float(parts[1]) if len(parts) > 1 else 60
            dbg.simulate_silence(minutes)
        elif action == "scenario":
            name = parts[1] if len(parts) > 1 else "repeated_no_reply"
            dbg = EmotionDebugger()  # fresh character
            dbg.simulate_scenario(name)
        elif action == "compare":
            name = parts[1] if len(parts) > 1 else "repeated_no_reply"
            compare_attachment_styles(name)
        elif action == "scenarios":
            print("  Available: repeated_no_reply, love_bomb, slow_ghost, "
                  "criticism_spiral, push_pull, boundary_violation")
        else:
            print(f"  Unknown command: {action}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        action = sys.argv[1]
        if action == "compare":
            scenario = sys.argv[2] if len(sys.argv) > 2 else "repeated_no_reply"
            compare_attachment_styles(scenario)
        elif action == "scenario":
            scenario = sys.argv[2] if len(sys.argv) > 2 else "repeated_no_reply"
            dbg = EmotionDebugger()
            dbg.simulate_scenario(scenario)
        elif action == "silence":
            minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 120
            dbg = EmotionDebugger()
            dbg.simulate_silence(minutes)
        else:
            print(f"Usage: python debug_tools.py [compare|scenario|silence] [name|minutes]")
    else:
        interactive_mode()
