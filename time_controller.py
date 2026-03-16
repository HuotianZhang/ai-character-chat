"""
Time Controller - Manages virtual time flow for the character system.
Supports: speed multiplier, pause/resume, manual time jumps, auto time-slot advancement.
"""
import time
import json
import os
from datetime import datetime, timedelta


class TimeController:
    """
    Virtual time system that decouples game time from real time.

    Key concepts:
    - virtual_time: the "game clock" the character lives in
    - speed_multiplier: how fast virtual time flows vs real time
      e.g. 60.0 = 1 real second = 1 virtual minute
           1.0 = real-time
           3600.0 = 1 real second = 1 virtual hour (fast debug mode)
    - time_slot: morning/afternoon/evening/night derived from virtual hour
    - virtual_day: how many days have passed since game start
    """

    # Time slot boundaries (hour ranges)
    TIME_SLOTS = [
        ("morning",   6, 12),   # 06:00 - 11:59
        ("afternoon", 12, 18),  # 12:00 - 17:59
        ("evening",   18, 22),  # 18:00 - 21:59
        ("night",     22, 6),   # 22:00 - 05:59 (wraps around)
    ]

    def __init__(self, speed_multiplier=60.0, start_hour=9):
        """
        Args:
            speed_multiplier: virtual seconds per real second (default 60 = 1min/s)
            start_hour: what hour the virtual day starts at (default 9am)
        """
        # Virtual clock anchor points
        now = datetime.now()
        self.virtual_start = now.replace(hour=start_hour, minute=0, second=0, microsecond=0)
        self.real_anchor = time.time()  # real timestamp when we set the anchor

        self.speed_multiplier = speed_multiplier
        self.paused = False
        self._pause_real_time = None  # real time when paused
        self._accumulated_pause = 0.0  # total real seconds spent paused

        # Track time slot changes for event triggering
        self._last_time_slot = self.get_time_slot()
        self._last_virtual_day = 0

        # Event callbacks
        self._on_slot_change_callbacks = []
        self._on_day_change_callbacks = []

    def get_virtual_now(self):
        """Get current virtual datetime."""
        if self.paused:
            real_elapsed = self._pause_real_time - self.real_anchor - self._accumulated_pause
        else:
            real_elapsed = time.time() - self.real_anchor - self._accumulated_pause

        virtual_elapsed_seconds = real_elapsed * self.speed_multiplier
        return self.virtual_start + timedelta(seconds=virtual_elapsed_seconds)

    def get_time_slot(self):
        """Get current time slot name based on virtual hour."""
        hour = self.get_virtual_now().hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def get_virtual_day(self):
        """Get how many days have passed since game start (0-indexed)."""
        delta = self.get_virtual_now() - self.virtual_start
        return max(0, delta.days)

    def get_display_time(self):
        """Get formatted virtual time string for display."""
        vt = self.get_virtual_now()
        return vt.strftime("%H:%M")

    def get_display_date(self):
        """Get formatted virtual date for display."""
        vt = self.get_virtual_now()
        weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        return f"Day {self.get_virtual_day() + 1} ({weekdays[vt.weekday()]}) {vt.strftime('%H:%M')}"

    def set_speed(self, multiplier):
        """
        Change time flow speed.
        Preserves current virtual time, then adjusts speed going forward.
        """
        # Snapshot current virtual time
        current_virtual = self.get_virtual_now()

        # Reset anchors
        self.real_anchor = time.time()
        self._accumulated_pause = 0.0
        self.virtual_start = current_virtual
        self.speed_multiplier = max(0.0, multiplier)

        if self.paused:
            self._pause_real_time = time.time()

        print(f"[TimeCtrl] Speed set to {multiplier}x (1 real sec = {multiplier} virtual sec)")

    def pause(self):
        """Pause virtual time."""
        if not self.paused:
            self.paused = True
            self._pause_real_time = time.time()
            print("[TimeCtrl] Time paused")

    def resume(self):
        """Resume virtual time."""
        if self.paused:
            pause_duration = time.time() - self._pause_real_time
            self._accumulated_pause += pause_duration
            self.paused = False
            self._pause_real_time = None
            print(f"[TimeCtrl] Time resumed (was paused for {pause_duration:.1f}s real)")

    def jump_to_slot(self, target_slot):
        """
        Jump virtual time forward to the next occurrence of target_slot.
        target_slot: "morning" / "afternoon" / "evening" / "night"
        """
        slot_hours = {"morning": 9, "afternoon": 14, "evening": 19, "night": 23}
        if target_slot not in slot_hours:
            print(f"[TimeCtrl] Invalid slot: {target_slot}")
            return

        current_virtual = self.get_virtual_now()
        target_hour = slot_hours[target_slot]

        # Build target datetime
        target = current_virtual.replace(hour=target_hour, minute=0, second=0, microsecond=0)
        if target <= current_virtual:
            target += timedelta(days=1)  # Next day

        # Reset anchors to jump
        self.real_anchor = time.time()
        self._accumulated_pause = 0.0
        self.virtual_start = target

        if self.paused:
            self._pause_real_time = time.time()

        print(f"[TimeCtrl] Jumped to {target.strftime('%Y-%m-%d %H:%M')} ({target_slot})")

    def jump_forward_hours(self, hours):
        """Jump virtual time forward by N hours."""
        current_virtual = self.get_virtual_now()
        new_time = current_virtual + timedelta(hours=hours)

        self.real_anchor = time.time()
        self._accumulated_pause = 0.0
        self.virtual_start = new_time

        if self.paused:
            self._pause_real_time = time.time()

        print(f"[TimeCtrl] Jumped forward {hours}h to {new_time.strftime('%Y-%m-%d %H:%M')}")

    def check_transitions(self):
        """
        Check if time slot or day changed since last check.
        Returns dict of transitions that occurred.
        """
        current_slot = self.get_time_slot()
        current_day = self.get_virtual_day()

        transitions = {
            "slot_changed": False,
            "day_changed": False,
            "old_slot": self._last_time_slot,
            "new_slot": current_slot,
            "old_day": self._last_virtual_day,
            "new_day": current_day,
        }

        if current_slot != self._last_time_slot:
            transitions["slot_changed"] = True
            self._last_time_slot = current_slot

        if current_day != self._last_virtual_day:
            transitions["day_changed"] = True
            self._last_virtual_day = current_day

        return transitions

    def get_status(self):
        """Get full time status for API/debug."""
        vt = self.get_virtual_now()
        return {
            "virtual_time": vt.strftime("%Y-%m-%d %H:%M:%S"),
            "display": self.get_display_date(),
            "time_slot": self.get_time_slot(),
            "day": self.get_virtual_day(),
            "speed": self.speed_multiplier,
            "paused": self.paused,
            "real_elapsed_sec": round(time.time() - self.real_anchor, 1),
        }

    def to_dict(self):
        """Serialize for persistence."""
        return {
            "virtual_now": self.get_virtual_now().isoformat(),
            "speed_multiplier": self.speed_multiplier,
            "paused": self.paused,
        }

    def restore_from_dict(self, d):
        """Restore from saved state."""
        saved_time_str = d.get("virtual_now")
        if saved_time_str:
            try:
                saved_time = datetime.fromisoformat(saved_time_str)
                self.virtual_start = saved_time
                self.real_anchor = time.time()
                self._accumulated_pause = 0.0
            except ValueError:
                pass

        self.speed_multiplier = d.get("speed_multiplier", self.speed_multiplier)
        if d.get("paused", False):
            self.paused = True
            self._pause_real_time = time.time()

        self._last_time_slot = self.get_time_slot()
        self._last_virtual_day = self.get_virtual_day()

    def save(self, filepath=None):
        """Save time state to file."""
        if filepath is None:
            from config import DATA_DIR
            filepath = os.path.join(DATA_DIR, "time_state.json")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, filepath=None):
        """Load time state from file."""
        if filepath is None:
            from config import DATA_DIR
            filepath = os.path.join(DATA_DIR, "time_state.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.restore_from_dict(data)
            print(f"[TimeCtrl] Restored: {self.get_display_date()}, speed={self.speed_multiplier}x")
