"""
AI Character State Engine - manages emotion, affinity, memory, storyline
This is the character's "living core"

v2.1: EmotionVector (7-axis) + TensionDetector + PressureAccumulator
"""
import json
import os
import time
import math
from datetime import datetime, timedelta
from collections import deque


# ============================================================
# EmotionVector — 7 independent emotion axes (replaces single scalar)
# ============================================================

class EmotionVector:
    """
    Multi-axis emotion model. Each axis is 0.0-1.0, independent.
    Opposing emotions can coexist (e.g., high attachment + high anger = 又爱又恨).

    Axes: joy, sadness, anger, anxiety, trust, disgust, attachment
    """

    AXES = ["joy", "sadness", "anger", "anxiety", "trust", "disgust", "attachment"]

    # Default decay half-lives per axis (in virtual minutes)
    # Lower = faster decay. Personality overrides these.
    DEFAULT_HALF_LIVES = {
        "joy": 45,
        "sadness": 90,
        "anger": 60,
        "anxiety": 120,
        "trust": 240,      # trust decays very slowly
        "disgust": 75,
        "attachment": 180,  # attachment is persistent
    }

    def __init__(self, decay_rates=None, baseline_mood=6.0):
        """
        decay_rates: dict of axis -> half_life_minutes (personality-derived)
        baseline_mood: legacy compatibility (1-10 scale, converted internally)
        """
        # Current axis values (0.0-1.0)
        self.axes = {axis: 0.0 for axis in self.AXES}

        # Set initial state from baseline_mood (legacy compat)
        # baseline 6.0/10 → slight positive: joy=0.15, trust=0.2
        if baseline_mood >= 6:
            self.axes["joy"] = min(0.3, (baseline_mood - 5) * 0.06)
            self.axes["trust"] = 0.2
        else:
            self.axes["sadness"] = min(0.3, (5 - baseline_mood) * 0.06)

        # Per-axis decay half-lives (minutes)
        self.half_lives = dict(self.DEFAULT_HALF_LIVES)
        if decay_rates:
            self.half_lives.update(decay_rates)

        self.last_decay_time = time.time()
        self.baseline_mood = baseline_mood

        # History of stimuli applied
        self.stimulus_history = []  # [{axis, delta, reason, time}, ...]

    def apply_stimulus(self, changes, reason=""):
        """
        Apply emotion stimulus.
        changes: dict of axis -> delta (e.g., {"joy": 0.3, "anxiety": -0.1})
        reason: what caused this change
        """
        now = time.time()
        for axis, delta in changes.items():
            if axis not in self.axes:
                continue
            old = self.axes[axis]
            self.axes[axis] = max(0.0, min(1.0, old + delta))

            self.stimulus_history.append({
                "axis": axis,
                "delta": delta,
                "old": round(old, 3),
                "new": round(self.axes[axis], 3),
                "reason": reason,
                "time": now,
            })

        # Keep history bounded
        if len(self.stimulus_history) > 60:
            self.stimulus_history = self.stimulus_history[-60:]

    def apply_legacy_stimulus(self, emotion_label, intensity, reason=""):
        """
        Backward-compatible: convert old single-scalar stimulus to multi-axis.
        emotion_label: Chinese or English emotion name
        intensity: -5.0 to +5.0 (old scale)
        """
        # Map emotion labels to axis changes
        label_map = {
            # Positive
            "happy": {"joy": 0.15}, "开心": {"joy": 0.15},
            "excited": {"joy": 0.25}, "兴奋": {"joy": 0.25},
            "moved": {"joy": 0.1, "trust": 0.15, "attachment": 0.1}, "感动": {"joy": 0.1, "trust": 0.15, "attachment": 0.1},
            "relaxed": {"joy": 0.05, "anxiety": -0.1}, "放松": {"joy": 0.05, "anxiety": -0.1},
            "grateful": {"joy": 0.1, "trust": 0.1}, "感激": {"joy": 0.1, "trust": 0.1},
            # Negative
            "sad": {"sadness": 0.2}, "难过": {"sadness": 0.2},
            "angry": {"anger": 0.2}, "愤怒": {"anger": 0.2}, "生气": {"anger": 0.15},
            "anxious": {"anxiety": 0.2}, "焦虑": {"anxiety": 0.2},
            "embarrassed": {"anxiety": 0.1, "disgust": 0.05}, "尴尬": {"anxiety": 0.1, "disgust": 0.05},
            "offended": {"anger": 0.15, "disgust": 0.1}, "被冒犯": {"anger": 0.15, "disgust": 0.1},
            "bored": {"joy": -0.1, "disgust": 0.05}, "无聊": {"joy": -0.1, "disgust": 0.05},
            "lost": {"sadness": 0.1, "anxiety": 0.1}, "迷茫": {"sadness": 0.1, "anxiety": 0.1},
            "disappointed": {"sadness": 0.15, "trust": -0.1}, "失望": {"sadness": 0.15, "trust": -0.1},
            "lonely": {"sadness": 0.1, "attachment": 0.15}, "孤独": {"sadness": 0.1, "attachment": 0.15},
            "jealous": {"anger": 0.1, "anxiety": 0.1, "attachment": 0.1}, "吃醋": {"anger": 0.1, "anxiety": 0.1, "attachment": 0.1},
            # Neutral
            "小开心": {"joy": 0.08},
            "有点烦": {"anger": 0.08},
            "平静": {},
        }

        # Scale factor from old intensity (-5 to +5) to multiplier
        scale = max(0.2, min(3.0, abs(intensity) / 1.5))

        base_changes = label_map.get(emotion_label, {})
        if not base_changes and intensity != 0:
            # Generic fallback
            if intensity > 0:
                base_changes = {"joy": 0.1}
            else:
                base_changes = {"sadness": 0.1}

        # Apply scaled changes
        scaled = {axis: delta * scale for axis, delta in base_changes.items()}
        self.apply_stimulus(scaled, reason=reason)

    def decay(self, elapsed_minutes=None):
        """
        Natural decay of all axes toward 0.
        Each axis decays independently based on its half-life.
        """
        now = time.time()
        if elapsed_minutes is None:
            elapsed_minutes = (now - self.last_decay_time) / 60.0
        self.last_decay_time = now

        if elapsed_minutes < 0.5:
            return

        for axis in self.AXES:
            if self.axes[axis] < 0.005:
                self.axes[axis] = 0.0
                continue
            half_life = self.half_lives.get(axis, 60)
            decay_factor = math.pow(0.5, elapsed_minutes / half_life)
            self.axes[axis] *= decay_factor

    def get_dominant(self, top_n=3):
        """Get the N strongest emotion axes."""
        sorted_axes = sorted(self.axes.items(), key=lambda x: x[1], reverse=True)
        return [(axis, val) for axis, val in sorted_axes[:top_n] if val > 0.05]

    def total_energy(self):
        """Total emotional energy (sum of all axes)."""
        return sum(self.axes.values())

    def get_mood_scalar(self):
        """
        Legacy compatibility: convert vector to single mood scalar (1-10).
        Positive axes push up, negative push down.
        """
        positive = self.axes["joy"] + self.axes["trust"] * 0.5
        negative = (self.axes["sadness"] + self.axes["anger"] + self.axes["anxiety"]
                    + self.axes["disgust"]) * 0.5
        base = self.baseline_mood
        mood = base + positive * 4.0 - negative * 4.0
        return max(1.0, min(10.0, mood))

    def get_mood_description(self):
        """Text description of current mood state."""
        dominant = self.get_dominant(2)
        if not dominant:
            return "Calm/neutral"

        descriptions = {
            "joy": "happy/joyful",
            "sadness": "sad/melancholy",
            "anger": "angry/frustrated",
            "anxiety": "anxious/worried",
            "trust": "trusting/open",
            "disgust": "disgusted/repulsed",
            "attachment": "longing/attached",
        }

        parts = []
        for axis, val in dominant:
            if val >= 0.7:
                parts.append(f"Very {descriptions.get(axis, axis)}")
            elif val >= 0.4:
                parts.append(descriptions.get(axis, axis).capitalize())
            elif val >= 0.15:
                parts.append(f"Slightly {descriptions.get(axis, axis)}")

        if not parts:
            mood = self.get_mood_scalar()
            if mood >= 7:
                return "Good mood"
            elif mood >= 5:
                return "Calm/neutral"
            else:
                return "A bit down"

        return ", ".join(parts)

    def get_active_emotion_labels(self):
        """Get list of currently active emotion labels (for legacy compat)."""
        labels = []
        label_map = {
            "joy": "开心", "sadness": "难过", "anger": "愤怒",
            "anxiety": "焦虑", "trust": "信任", "disgust": "抗拒",
            "attachment": "依恋",
        }
        for axis, val in self.axes.items():
            if val >= 0.2:
                labels.append(label_map.get(axis, axis))
        return labels

    def to_dict(self):
        return {
            "axes": {k: round(v, 4) for k, v in self.axes.items()},
            "half_lives": self.half_lives,
            "baseline_mood": self.baseline_mood,
            "last_decay_time": self.last_decay_time,
            "stimulus_history": self.stimulus_history[-30:],
            # Legacy compat
            "current_mood": round(self.get_mood_scalar(), 2),
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(
            decay_rates=d.get("half_lives"),
            baseline_mood=d.get("baseline_mood", 6.0)
        )
        if "axes" in d:
            for axis in cls.AXES:
                obj.axes[axis] = d["axes"].get(axis, 0.0)
        elif "current_mood" in d:
            # Migrate from old scalar format
            mood = d.get("current_mood", 6.0)
            if mood >= 6:
                obj.axes["joy"] = min(0.5, (mood - 5) * 0.1)
            else:
                obj.axes["sadness"] = min(0.5, (5 - mood) * 0.1)
        obj.last_decay_time = d.get("last_decay_time", time.time())
        obj.stimulus_history = d.get("stimulus_history", [])
        return obj


# ============================================================
# TensionDetector — detects compound/contradictory emotional states
# ============================================================

class TensionDetector:
    """
    Detects compound emotional states from conflicting high axes.
    These tension states override normal behavior patterns.
    """

    # (pattern_name, required_axes as {axis: min_value}, label_cn, behavioral_note)
    TENSION_PATTERNS = [
        {
            "name": "love_hate",
            "require": {"attachment": 0.6, "anger": 0.5},
            "label_cn": "又爱又恨",
            "behavior": "Messages alternate between warmth and hostility; contradictory statements; push-pull pattern",
            "lang_override": {"segmentation_boost": 2, "contradiction_mode": True},
        },
        {
            "name": "anxious_joy",
            "require": {"joy": 0.5, "anxiety": 0.5},
            "label_cn": "患得患失",
            "behavior": "Happy but terrified it won't last; seeks reassurance; reads into small signals",
            "lang_override": {"question_frequency_boost": True},
        },
        {
            "name": "grievance",
            "require": {"anger": 0.5, "sadness": 0.5},
            "label_cn": "委屈",
            "behavior": "Feels wronged; wants acknowledgment not solutions; may passive-aggressive",
            "lang_override": {"passive_aggressive": True},
        },
        {
            "name": "frozen_attachment",
            "require": {"attachment": 0.6, "anxiety": 0.6},
            "label_cn": "害怕失去",
            "behavior": "Deeply attached but paralyzed by fear; over-analyzes responses; clingy or withdrawing",
            "lang_override": {},
        },
        {
            "name": "bitter_trust",
            "require": {"trust": 0.4, "disgust": 0.4},
            "label_cn": "信任裂痕",
            "behavior": "Wants to trust but feels betrayed; tests the other person; guarded vulnerability",
            "lang_override": {"testing_behavior": True},
        },
    ]

    # Special aggregate states
    AGGREGATE_STATES = [
        {
            "name": "numb",
            "condition": "low_energy",  # all axes < 0.15
            "label_cn": "麻木",
            "behavior": "Flat affect, minimal response, emotionally shutdown",
        },
        {
            "name": "overwhelmed",
            "condition": "high_energy",  # total energy >= 3.5
            "label_cn": "情绪过载",
            "behavior": "Chaotic, may shut down or lash out, messages become erratic",
        },
    ]

    @staticmethod
    def detect(emotion_vector):
        """
        Detect all active tension states.
        Returns list of active tension dicts.
        """
        active = []
        axes = emotion_vector.axes

        # Check pattern-based tensions
        for pattern in TensionDetector.TENSION_PATTERNS:
            match = True
            for axis, min_val in pattern["require"].items():
                if axes.get(axis, 0) < min_val:
                    match = False
                    break
            if match:
                active.append({
                    "name": pattern["name"],
                    "label_cn": pattern["label_cn"],
                    "behavior": pattern["behavior"],
                    "lang_override": pattern.get("lang_override", {}),
                })

        # Check aggregate states
        total = emotion_vector.total_energy()
        all_low = all(v < 0.15 for v in axes.values())

        if all_low and total < 0.5:
            for state in TensionDetector.AGGREGATE_STATES:
                if state["condition"] == "low_energy":
                    active.append({
                        "name": state["name"],
                        "label_cn": state["label_cn"],
                        "behavior": state["behavior"],
                        "lang_override": {},
                    })

        if total >= 3.5:
            for state in TensionDetector.AGGREGATE_STATES:
                if state["condition"] == "high_energy":
                    active.append({
                        "name": state["name"],
                        "label_cn": state["label_cn"],
                        "behavior": state["behavior"],
                        "lang_override": {},
                    })

        return active


# ============================================================
# PressureAccumulator — hidden pressure buildup + burst mechanics
# ============================================================

class PressureAccumulator:
    """
    Tracks hidden pressure from repeated small stimuli.
    When pressure crosses threshold → non-linear emotional burst.

    Channels: no_reply, criticized, controlled, ignored, boundary_pushed
    """

    CHANNELS = ["no_reply", "criticized", "controlled", "ignored", "boundary_pushed"]

    # Burst direction based on attachment style category
    BURST_PROFILES = {
        "anxious": {
            # Anxious types → explosive anger or desperate clinging
            "no_reply": {"anger": 0.6, "anxiety": 0.4, "attachment": 0.3},
            "criticized": {"sadness": 0.5, "anger": 0.3},
            "controlled": {"anger": 0.5, "anxiety": 0.3},
            "ignored": {"sadness": 0.4, "anger": 0.4, "attachment": 0.3},
            "boundary_pushed": {"anger": 0.3, "anxiety": 0.4},
        },
        "avoidant": {
            # Avoidant types → withdrawal, emotional shutdown
            "no_reply": {"disgust": 0.2, "trust": -0.3},
            "criticized": {"anger": 0.3, "trust": -0.2},
            "controlled": {"anger": 0.5, "disgust": 0.3},
            "ignored": {},  # avoidants don't mind being ignored much
            "boundary_pushed": {"disgust": 0.4, "anger": 0.3},
        },
        "secure": {
            # Secure types → proportional response
            "no_reply": {"sadness": 0.2, "anxiety": 0.1},
            "criticized": {"sadness": 0.2, "anger": 0.2},
            "controlled": {"anger": 0.3},
            "ignored": {"sadness": 0.2},
            "boundary_pushed": {"anger": 0.3, "disgust": 0.2},
        },
        "fearful": {
            # Fearful-avoidant → chaotic, self-destructive
            "no_reply": {"anger": 0.4, "sadness": 0.4, "attachment": 0.3},
            "criticized": {"sadness": 0.5, "disgust": 0.2, "anger": 0.2},
            "controlled": {"anger": 0.4, "anxiety": 0.4},
            "ignored": {"sadness": 0.5, "attachment": 0.3},
            "boundary_pushed": {"anger": 0.3, "disgust": 0.3, "anxiety": 0.3},
        },
    }

    # Map attachment styles to burst profile categories
    ATTACHMENT_TO_PROFILE = {
        "安全·依恋": "secure", "安全·松弛自洽": "secure", "安全·主动滋养": "secure",
        "焦虑·依恋": "anxious", "焦虑·讨好牺牲": "anxious", "焦虑·情绪化施压": "anxious",
        "回避·安全": "avoidant", "回避·焦虑": "avoidant", "回避·依恋": "avoidant",
        "回避·理想化远方": "avoidant",
        "恐惧回避·推拉矛盾": "fearful", "恐惧回避·自毁测试": "fearful",
    }

    def __init__(self, attachment_style="安全·依恋"):
        self.pressure = {ch: 0.0 for ch in self.CHANNELS}
        self.threshold = {ch: 1.0 for ch in self.CHANNELS}  # burst threshold
        self.burst_count = {ch: 0 for ch in self.CHANNELS}
        self.last_tick_time = time.time()
        self.burst_history = []

        # Determine burst profile
        self.profile_key = self.ATTACHMENT_TO_PROFILE.get(attachment_style, "secure")

    def record(self, channel, amount=0.15):
        """
        Record a pressure event on a channel.
        amount: how much pressure to add (default 0.15, range 0.05-0.5)
        """
        if channel not in self.pressure:
            return None

        self.pressure[channel] = min(2.0, self.pressure[channel] + amount)

        # Check for burst
        if self.pressure[channel] >= self.threshold[channel]:
            return self._burst(channel)
        return None

    def _burst(self, channel):
        """
        Pressure burst — returns emotion changes dict to apply.
        """
        profile = self.BURST_PROFILES.get(self.profile_key, self.BURST_PROFILES["secure"])
        emotion_changes = dict(profile.get(channel, {}))

        # Scale by how much over threshold
        overshoot = self.pressure[channel] / self.threshold[channel]
        emotion_changes = {k: v * overshoot for k, v in emotion_changes.items()}

        # Reset pressure
        self.pressure[channel] = 0.0

        # Sensitization: lower threshold by 10% (min 0.5)
        self.threshold[channel] = max(0.5, self.threshold[channel] * 0.9)
        self.burst_count[channel] += 1

        self.burst_history.append({
            "channel": channel,
            "time": time.time(),
            "profile": self.profile_key,
            "overshoot": round(overshoot, 2),
        })
        if len(self.burst_history) > 30:
            self.burst_history = self.burst_history[-30:]

        return {
            "burst": True,
            "channel": channel,
            "emotion_changes": emotion_changes,
            "burst_count": self.burst_count[channel],
        }

    def tick(self, elapsed_minutes=None):
        """
        Natural pressure decay. Call periodically.
        Pressure decays slowly (half-life ~120 virtual minutes).
        """
        now = time.time()
        if elapsed_minutes is None:
            elapsed_minutes = (now - self.last_tick_time) / 60.0
        self.last_tick_time = now

        if elapsed_minutes < 1:
            return

        decay_factor = math.pow(0.5, elapsed_minutes / 120.0)
        for ch in self.CHANNELS:
            self.pressure[ch] *= decay_factor
            if self.pressure[ch] < 0.01:
                self.pressure[ch] = 0.0

    def get_active_pressures(self):
        """Get channels with significant pressure buildup."""
        return {ch: round(v, 3) for ch, v in self.pressure.items() if v >= 0.1}

    def to_dict(self):
        return {
            "pressure": {k: round(v, 4) for k, v in self.pressure.items()},
            "threshold": {k: round(v, 3) for k, v in self.threshold.items()},
            "burst_count": self.burst_count,
            "profile_key": self.profile_key,
            "last_tick_time": self.last_tick_time,
            "burst_history": self.burst_history[-10:],
        }

    @classmethod
    def from_dict(cls, d, attachment_style="安全·依恋"):
        obj = cls(attachment_style)
        obj.pressure = d.get("pressure", {ch: 0.0 for ch in cls.CHANNELS})
        obj.threshold = d.get("threshold", {ch: 1.0 for ch in cls.CHANNELS})
        obj.burst_count = d.get("burst_count", {ch: 0 for ch in cls.CHANNELS})
        obj.profile_key = d.get("profile_key", "secure")
        obj.last_tick_time = d.get("last_tick_time", time.time())
        obj.burst_history = d.get("burst_history", [])
        return obj


# ============================================================
# AffinitySystem (unchanged from v2.0)
# ============================================================

class AffinitySystem:
    """
    Affinity System
    - affinity: normal affinity (0-100, initial 65)
    - special_affinity: special affinity/attachment slot (0-100, initial 65)
    - affinity_log: affinity change log
    """

    def __init__(self, initial=65, special_initial=65):
        self.affinity = initial
        self.special_affinity = special_initial
        self.affinity_log = []

    def modify_affinity(self, delta, reason="", category="normal"):
        if category == "special":
            old = self.special_affinity
            self.special_affinity = max(0, min(100, self.special_affinity + delta))
            self.affinity_log.append({
                "time": time.time(), "type": "special", "delta": delta,
                "from": old, "to": self.special_affinity, "reason": reason,
            })
        else:
            old = self.affinity
            self.affinity = max(0, min(100, self.affinity + delta))
            self.affinity_log.append({
                "time": time.time(), "type": "normal", "delta": delta,
                "from": old, "to": self.affinity, "reason": reason,
            })
        if len(self.affinity_log) > 100:
            self.affinity_log = self.affinity_log[-100:]

    def get_affinity_level(self):
        a = self.affinity
        if a >= 90: return "Very high (deep trust)"
        elif a >= 80: return "High (close/likes)"
        elif a >= 70: return "Medium-high (interested)"
        elif a >= 60: return "Medium (neutral/new)"
        elif a >= 40: return "Low (distant)"
        else: return "Very low (dislike)"

    def is_dark_line_unlocked(self):
        return self.affinity >= 85

    def to_dict(self):
        return {
            "affinity": self.affinity,
            "special_affinity": self.special_affinity,
            "affinity_log": self.affinity_log[-30:],
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls(d.get("affinity", 65), d.get("special_affinity", 65))
        obj.affinity_log = d.get("affinity_log", [])
        return obj


# ============================================================
# MemorySystem (unchanged from v2.0)
# ============================================================

class MemorySystem:
    """
    Memory System
    - short_term: short-term conversation memory (recent N turns)
    - long_term: long-term memory (important events/high emotion memories)
    - emotional_memory: conditional reflex emotion memory (triggers)
    - semantic_memory: cognitive summary of user
    """

    def __init__(self, max_short=30, max_long=100):
        self.short_term = []
        self.long_term = []
        self.emotional_memory = []
        self.semantic_memory = {}
        self.max_short = max_short
        self.max_long = max_long

    def add_conversation(self, role, content):
        self.short_term.append({
            "role": role, "content": content, "time": time.time(),
        })
        if len(self.short_term) > self.max_short:
            self.short_term = self.short_term[-self.max_short:]

    def consolidate_memory(self, event, emotion_label, intensity, context=""):
        self.long_term.append({
            "event": event, "emotion": emotion_label, "intensity": intensity,
            "context": context, "time": time.time(), "recall_count": 0,
            "distortion": 0.0,
        })
        if len(self.long_term) > self.max_long:
            self.long_term.sort(key=lambda x: abs(x["intensity"]), reverse=True)
            self.long_term = self.long_term[:self.max_long]

    def add_trigger(self, trigger_pattern, response_pattern, origin_event):
        self.emotional_memory.append({
            "trigger": trigger_pattern,
            "response": response_pattern,
            "origin": origin_event,
            "times_triggered": 0,
        })

    def update_semantic(self, key, value):
        self.semantic_memory[key] = {"value": value, "updated_at": time.time()}

    def get_recent_context(self, n=10):
        return self.short_term[-n:]

    def get_relevant_long_term(self, keywords=None, n=5):
        if not keywords:
            return self.long_term[-n:]
        scored = []
        for mem in self.long_term:
            score = sum(1 for kw in keywords if kw in mem.get("event", "") or kw in mem.get("context", ""))
            if score > 0:
                scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:n]]

    def check_triggers(self, input_text):
        triggered = []
        for trigger in self.emotional_memory:
            if trigger["trigger"] in input_text:
                trigger["times_triggered"] += 1
                triggered.append(trigger)
        return triggered

    def to_dict(self):
        return {
            "short_term": self.short_term,
            "long_term": self.long_term,
            "emotional_memory": self.emotional_memory,
            "semantic_memory": self.semantic_memory,
        }

    @classmethod
    def from_dict(cls, d, max_short=30, max_long=100):
        obj = cls(max_short, max_long)
        obj.short_term = d.get("short_term", [])
        obj.long_term = d.get("long_term", [])
        obj.emotional_memory = d.get("emotional_memory", [])
        obj.semantic_memory = d.get("semantic_memory", {})
        return obj


# ============================================================
# StorylineSystem (unchanged from v2.0)
# ============================================================

class StorylineSystem:
    """Storyline System - manages character's daily life progression"""

    def __init__(self):
        self.storyline = []
        self.current_day = 0
        self.current_time_slot = "afternoon"
        self.emotion_stage = ""

    def set_storyline(self, storyline_data):
        self.storyline = storyline_data

    def get_today(self):
        if 0 <= self.current_day < len(self.storyline):
            return self.storyline[self.current_day]
        return None

    def get_current_events(self):
        today = self.get_today()
        if not today:
            return []
        time_order = ["morning", "afternoon", "evening", "night"]
        current_idx = time_order.index(self.current_time_slot) if self.current_time_slot in time_order else 0
        events = []
        for event in today.get("events", []):
            event_slot = event.get("time_slot", "morning")
            if event_slot in time_order:
                if time_order.index(event_slot) <= current_idx:
                    events.append(event)
        return events

    def advance_time(self):
        slots = ["morning", "afternoon", "evening", "night"]
        idx = slots.index(self.current_time_slot) if self.current_time_slot in slots else 0
        if idx < len(slots) - 1:
            self.current_time_slot = slots[idx + 1]
        else:
            self.current_time_slot = "morning"
            self.current_day += 1

    def to_dict(self):
        return {
            "storyline": self.storyline,
            "current_day": self.current_day,
            "current_time_slot": self.current_time_slot,
            "emotion_stage": self.emotion_stage,
        }

    @classmethod
    def from_dict(cls, d):
        obj = cls()
        obj.storyline = d.get("storyline", [])
        obj.current_day = d.get("current_day", 0)
        obj.current_time_slot = d.get("current_time_slot", "afternoon")
        obj.emotion_stage = d.get("emotion_stage", "")
        return obj


# ============================================================
# CharacterState — aggregates all subsystems
# ============================================================

class CharacterState:
    """
    Complete Character State - aggregates all subsystems.
    v2.5: Added RelationshipJudge for metacognitive evaluation.
    """

    def __init__(self, character_data):
        self.character = character_data
        personality = character_data.get("性格维度", {})
        derived = character_data.get("衍生属性", {})

        # Build per-axis decay rates from personality
        decay_rates = derived.get("emotion_decay_rates", None)
        baseline = 6.0

        # EmotionVector (replaces old EmotionSystem)
        self.emotion = EmotionVector(decay_rates=decay_rates, baseline_mood=baseline)

        # PressureAccumulator
        attachment_style = personality.get("依恋模式", "安全·依恋")
        self.pressure = PressureAccumulator(attachment_style)

        # Other systems (unchanged)
        self.affinity = AffinitySystem()
        self.memory = MemorySystem()
        self.storyline = StorylineSystem()

        # RelationshipJudge (NEW v2.5) — metacognitive evaluation
        from relationship_judge import RelationshipJudge
        self.judge = RelationshipJudge(self)

        # Interaction statistics
        self.interaction_count = 0
        self.session_start_time = time.time()
        self.last_interaction_time = time.time()

    def process_input(self, user_input):
        """
        Update state before processing user input.
        Returns a state snapshot for the dialogue engine.
        """
        now = time.time()

        # 1. Natural emotion decay
        self.emotion.decay()

        # 2. Pressure tick (natural pressure decay)
        self.pressure.tick()

        # 3. Check triggers
        triggers = self.memory.check_triggers(user_input)

        # 4. Record conversation
        if user_input:
            self.memory.add_conversation("user", user_input)

        # 5. Update interaction statistics
        self.interaction_count += 1
        self.last_interaction_time = now

        # 6. Detect tension states
        tensions = TensionDetector.detect(self.emotion)

        # 7. Build state snapshot
        snapshot = {
            # Emotion vector
            "emotion_axes": {k: round(v, 3) for k, v in self.emotion.axes.items()},
            "dominant_emotions": self.emotion.get_dominant(3),
            "total_energy": round(self.emotion.total_energy(), 2),
            "tension_states": tensions,

            # Legacy compat
            "mood": self.emotion.get_mood_scalar(),
            "mood_description": self.emotion.get_mood_description(),
            "active_emotions": self.emotion.get_active_emotion_labels(),

            # Pressure
            "active_pressures": self.pressure.get_active_pressures(),

            # Affinity
            "affinity": self.affinity.affinity,
            "affinity_level": self.affinity.get_affinity_level(),
            "special_affinity": self.affinity.special_affinity,
            "dark_line_unlocked": self.affinity.is_dark_line_unlocked(),

            # Memory & context
            "triggered_memories": triggers,
            "recent_context": self.memory.get_recent_context(10),
            "semantic_memory": self.memory.semantic_memory,

            # Storyline
            "today_events": self.storyline.get_current_events(),
            "time_slot": self.storyline.current_time_slot,
            "day": self.storyline.current_day,
            "emotion_stage": self.storyline.emotion_stage,

            # Stats
            "interaction_count": self.interaction_count,

            # Relationship judgments (v3.0 — no goal field)
            "relationship_label": self.judge.relationship_label,
            "user_speculation": self.judge.user_speculation,
        }

        return snapshot

    def process_output(self, char_response, emotion_delta=0, emotion_label="",
                       emotion_changes=None, affinity_delta=0, special_affinity_delta=0,
                       memory_note="", semantic_updates=None):
        """
        Update state after character output.
        Supports both old format (emotion_delta/label) and new format (emotion_changes dict).
        """
        # Record character response
        self.memory.add_conversation("character", char_response)

        # Update emotion — prefer new multi-axis format
        if emotion_changes and isinstance(emotion_changes, dict):
            self.emotion.apply_stimulus(emotion_changes, reason=memory_note)
        elif emotion_delta != 0 and emotion_label:
            # Legacy single-scalar format
            self.emotion.apply_legacy_stimulus(emotion_label, emotion_delta, reason=memory_note)

        # Update affinity
        if affinity_delta != 0:
            self.affinity.modify_affinity(affinity_delta, reason=memory_note, category="normal")
        if special_affinity_delta != 0:
            self.affinity.modify_affinity(special_affinity_delta, reason=memory_note, category="special")

        # Consolidate important memories
        energy = sum(abs(v) for v in (emotion_changes or {}).values()) if emotion_changes else abs(emotion_delta)
        if memory_note and energy >= 0.3:
            self.memory.consolidate_memory(
                event=memory_note,
                emotion_label=emotion_label or str(list((emotion_changes or {}).keys())[:2]),
                intensity=energy,
            )

        # Update semantic memory (cap at 5 per turn to prevent LLM spam)
        if semantic_updates:
            items = list(semantic_updates.items())
            if len(items) > 5:
                print(f"[State] WARNING: semantic_updates has {len(items)} entries, capping at 5")
                items = items[:5]
            for key, value in items:
                self.memory.update_semantic(key, value)

    def save(self, state_path=None, memory_path=None, storyline_path=None):
        """Persist state to storage"""
        from config import STATE_FILE, MEMORY_FILE, STORYLINE_FILE

        state_path = state_path or STATE_FILE
        memory_path = memory_path or MEMORY_FILE
        storyline_path = storyline_path or STORYLINE_FILE

        os.makedirs(os.path.dirname(state_path), exist_ok=True)

        state_data = {
            "emotion": self.emotion.to_dict(),
            "pressure": self.pressure.to_dict(),
            "affinity": self.affinity.to_dict(),
            "judge": self.judge.to_dict(),
            "interaction_count": self.interaction_count,
            "session_start_time": self.session_start_time,
            "last_interaction_time": self.last_interaction_time,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state_data, f, ensure_ascii=False, indent=2)

        with open(memory_path, "w", encoding="utf-8") as f:
            json.dump(self.memory.to_dict(), f, ensure_ascii=False, indent=2)

        with open(storyline_path, "w", encoding="utf-8") as f:
            json.dump(self.storyline.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, state_path=None, memory_path=None, storyline_path=None):
        """Restore state from files"""
        from config import STATE_FILE, MEMORY_FILE, STORYLINE_FILE

        state_path = state_path or STATE_FILE
        memory_path = memory_path or MEMORY_FILE
        storyline_path = storyline_path or STORYLINE_FILE

        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load emotion (handles migration from old format)
            self.emotion = EmotionVector.from_dict(data.get("emotion", {}))

            # Load pressure (if exists)
            if "pressure" in data:
                attachment = self.character.get("性格维度", {}).get("依恋模式", "安全·依恋")
                self.pressure = PressureAccumulator.from_dict(data["pressure"], attachment)

            self.affinity = AffinitySystem.from_dict(data.get("affinity", {}))

            # Load judge state (v2.5)
            if "judge" in data:
                from relationship_judge import RelationshipJudge
                self.judge = RelationshipJudge.from_dict(data["judge"], self)

            self.interaction_count = data.get("interaction_count", 0)
            self.session_start_time = data.get("session_start_time", time.time())
            self.last_interaction_time = data.get("last_interaction_time", time.time())

        if os.path.exists(memory_path):
            with open(memory_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.memory = MemorySystem.from_dict(data)

        if os.path.exists(storyline_path):
            with open(storyline_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.storyline = StorylineSystem.from_dict(data)

    def _get_prompt_stage(self):
        """Determine prompt unlocking stage (1=Surface, 2=Familiar, 3=Deep)."""
        affinity = self.affinity.affinity
        dark = self.affinity.is_dark_line_unlocked()
        if dark or affinity >= 85:
            return 3
        elif affinity >= 70 or self.interaction_count >= 30:
            return 2
        else:
            return 1

    def get_status_summary(self):
        """Get status summary (for debugging/display)"""
        tensions = TensionDetector.detect(self.emotion)
        tension_labels = [t["label_cn"] for t in tensions]

        dominant = self.emotion.get_dominant(3)
        dominant_str = ", ".join(f"{ax}={val:.2f}" for ax, val in dominant) if dominant else "none"

        stage = self._get_prompt_stage()
        stage_names = {1: "Surface", 2: "Familiar", 3: "Deep"}

        return {
            "mood": f"{self.emotion.get_mood_scalar():.1f}/10 ({self.emotion.get_mood_description()})",
            "emotion_axes": {k: round(v, 2) for k, v in self.emotion.axes.items() if v >= 0.05},
            "dominant": dominant_str,
            "tensions": tension_labels if tension_labels else ["none"],
            "active_emotions": self.emotion.get_active_emotion_labels(),
            "pressures": self.pressure.get_active_pressures(),
            "affinity": f"{self.affinity.affinity}/100 ({self.affinity.get_affinity_level()})",
            "special_affinity": f"{self.affinity.special_affinity}/100",
            "dark_line_unlocked": self.affinity.is_dark_line_unlocked(),
            "turns": self.interaction_count,
            "story_day": f"Day {self.storyline.current_day + 1} {self.storyline.current_time_slot}",
            "memory_count": f"short {len(self.memory.short_term)} | long {len(self.memory.long_term)}",
            "relationship_label": self.judge.relationship_label,
            "user_speculation": self.judge.user_speculation,
            "prompt_stage": f"{stage} ({stage_names[stage]})",
        }
