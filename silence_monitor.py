"""
Silence Monitor — Detects user silence and drives emotional reactions.

The core problem this solves: without it, the character's emotions only update
when the user sends a message. In reality, being ignored/waiting for a reply
causes anxiety, frustration, sadness, or boredom that builds over time.

The SilenceMonitor runs on every polling tick (every ~3 seconds) and:
  1. Tracks virtual minutes since the user last sent a message
  2. Applies graduated emotional effects based on silence duration
  3. Feeds "no_reply" pressure into the PressureAccumulator
  4. Triggers proactive messages expressing the character's reaction to silence

The silence reaction style is personality-driven:
  - Anxious attachment → anxiety spikes early, then anger
  - Avoidant attachment → mild relief initially, then slight disgust at neediness
  - Fearful-avoidant → oscillates between anxiety and detachment
  - Secure attachment → calm for longer, mild curiosity, then gentle sadness
"""
import time
import math


class SilenceMonitor:
    """
    Monitors user silence and generates emotional/pressure effects.
    Designed to be called on every proactive-event polling tick.
    """

    # Silence phases (virtual minutes thresholds)
    # Phase 0: 0-10 min  → normal, no effect
    # Phase 1: 10-30 min → mild (slight anxiety/curiosity)
    # Phase 2: 30-90 min → moderate (anxiety, sadness, or anger building)
    # Phase 3: 90-240 min → strong (pressure accumulation, proactive triggers)
    # Phase 4: 240+ min  → severe (emotional burst risk, withdrawal possible)

    PHASE_THRESHOLDS = [0, 10, 30, 90, 240]

    # Hard limits: max proactive messages per silence episode, per personality
    # Most real people do NOT keep messaging when ignored.
    # Only anxious types with extreme emotion might send 1 follow-up.
    SILENCE_MSG_LIMITS = {
        "secure":  0,    # Secure people don't chase — they wait
        "avoidant": 0,   # Avoidant people withdraw — they never chase
        "anxious": 1,    # Anxious people might send 1 follow-up, but not more
        "fearful": 1,    # Fearful-avoidant might send 1, then retreat
    }

    # Minimum emotion energy required to trigger a silence proactive message
    # Even anxious types need to be emotionally charged enough to act
    SILENCE_MSG_ENERGY_THRESHOLD = {
        "secure":  99.0,   # Effectively never (hard blocked by limit=0 anyway)
        "avoidant": 99.0,  # Effectively never
        "anxious": 1.5,    # Needs moderate emotional charge
        "fearful": 2.0,    # Needs higher emotional charge (more inhibited)
    }

    # Minimum phase required before a proactive message CAN fire
    SILENCE_MSG_MIN_PHASE = {
        "secure": 99,    # Never
        "avoidant": 99,  # Never
        "anxious": 3,    # Only at 90+ virtual minutes of silence
        "fearful": 3,    # Only at 90+ virtual minutes of silence
    }

    # Reaction profiles per attachment category
    # Each phase maps to: emotion_changes dict, pressure_amount, inner_thought
    # NOTE: trigger_chance removed — replaced by hard limits above
    PROFILES = {
        "anxious": {
            # Anxious types react fast and escalate quickly
            "phase_1": {
                "emotion": {"anxiety": 0.03},
                "pressure": 0.02,
                "inner": "还没回...应该在忙吧",
            },
            "phase_2": {
                "emotion": {"anxiety": 0.06, "sadness": 0.03, "attachment": 0.02},
                "pressure": 0.05,
                "inner": "为什么不回我...是不是说错了什么",
            },
            "phase_3": {
                "emotion": {"anxiety": 0.08, "anger": 0.05, "sadness": 0.04, "attachment": 0.03},
                "pressure": 0.10,
                "inner": "又不回消息...每次都这样",
            },
            "phase_4": {
                "emotion": {"anger": 0.08, "sadness": 0.06, "disgust": 0.03, "trust": -0.04},
                "pressure": 0.15,
                "inner": "算了，可能就是不想理我吧",
            },
        },
        "avoidant": {
            # Avoidant types are slow to react, may feel relief initially
            "phase_1": {
                "emotion": {},  # no reaction
                "pressure": 0.0,
                "inner": "没回就没回",
            },
            "phase_2": {
                "emotion": {"joy": 0.01},  # slight relief at not being demanded upon
                "pressure": 0.01,
                "inner": "也好，安静一会儿",
            },
            "phase_3": {
                "emotion": {"sadness": 0.02, "anxiety": 0.01},
                "pressure": 0.03,
                "inner": "...好像很久没消息了",
            },
            "phase_4": {
                "emotion": {"sadness": 0.03, "disgust": 0.02, "trust": -0.02},
                "pressure": 0.05,
                "inner": "果然不该太投入",
            },
        },
        "secure": {
            # Secure types are patient, react proportionally
            "phase_1": {
                "emotion": {},
                "pressure": 0.0,
                "inner": "可能在忙",
            },
            "phase_2": {
                "emotion": {"anxiety": 0.01},
                "pressure": 0.01,
                "inner": "还没回，应该有事吧",
            },
            "phase_3": {
                "emotion": {"sadness": 0.03, "anxiety": 0.02},
                "pressure": 0.04,
                "inner": "好久没回了，有点想聊天",
            },
            "phase_4": {
                "emotion": {"sadness": 0.04, "anxiety": 0.02, "trust": -0.01},
                "pressure": 0.06,
                "inner": "是不是发生什么事了...",
            },
        },
        "fearful": {
            # Fearful-avoidant oscillates between clinging and pulling away
            "phase_1": {
                "emotion": {"anxiety": 0.04},
                "pressure": 0.02,
                "inner": "没回...我就知道",
            },
            "phase_2": {
                "emotion": {"anxiety": 0.05, "anger": 0.03, "attachment": 0.03},
                "pressure": 0.06,
                "inner": "又来了，每次都不回",
            },
            "phase_3": {
                "emotion": {"anger": 0.04, "sadness": 0.05, "disgust": 0.03, "attachment": 0.02},
                "pressure": 0.12,
                "inner": "无所谓了...不对，还是很在意",
            },
            "phase_4": {
                "emotion": {"sadness": 0.06, "anger": 0.04, "disgust": 0.04, "trust": -0.05},
                "pressure": 0.15,
                "inner": "我不该主动的",
            },
        },
    }

    # Map attachment styles to profile keys (same mapping as PressureAccumulator)
    ATTACHMENT_MAP = {
        "安全·依恋": "secure", "安全·松弛自洽": "secure", "安全·主动滋养": "secure",
        "焦虑·依恋": "anxious", "焦虑·讨好牺牲": "anxious", "焦虑·情绪化施压": "anxious",
        "回避·安全": "avoidant", "回避·焦虑": "avoidant", "回避·依恋": "avoidant",
        "回避·理想化远方": "avoidant",
        "恐惧回避·推拉矛盾": "fearful", "恐惧回避·自毁测试": "fearful",
    }

    def __init__(self, character_state, time_controller):
        self.state = character_state
        self.time_ctrl = time_controller

        # Determine personality profile
        attachment = character_state.character.get("性格维度", {}).get("依恋模式", "安全·依恋")
        self.profile_key = self.ATTACHMENT_MAP.get(attachment, "secure")

        # Tracking
        self.last_user_message_real = time.time()
        self.last_user_message_virtual = None  # set when first message arrives
        self._last_phase_applied = 0  # track which phase we last applied
        self._last_tick_time = time.time()
        self._silence_tick_count = 0  # how many ticks since last user message

        # Hard-limit tracking for proactive silence messages
        self._silence_messages_sent = 0  # count of proactive messages sent this silence episode
        self._silence_msg_limit = self.SILENCE_MSG_LIMITS.get(self.profile_key, 0)
        self._silence_energy_threshold = self.SILENCE_MSG_ENERGY_THRESHOLD.get(self.profile_key, 99.0)
        self._silence_min_phase = self.SILENCE_MSG_MIN_PHASE.get(self.profile_key, 99)

    def on_user_message(self):
        """Call this whenever the user sends a message. Resets silence tracking."""
        self.last_user_message_real = time.time()
        if self.time_ctrl:
            self.last_user_message_virtual = self.time_ctrl.get_virtual_now()
        self._last_phase_applied = 0
        self._silence_tick_count = 0
        self._silence_messages_sent = 0  # reset message budget for new silence episode

    def tick(self):
        """
        Called on every polling cycle (~3 seconds).
        Checks silence duration and applies emotional effects.

        Returns:
          None if no proactive message should be sent, or
          dict with silence event info for proactive message generation.
        """
        now_real = time.time()

        # Don't tick too fast
        if now_real - self._last_tick_time < 2.0:
            return None
        self._last_tick_time = now_real

        # Calculate virtual minutes of silence
        if self.time_ctrl and self.last_user_message_virtual:
            virtual_now = self.time_ctrl.get_virtual_now()
            delta = (virtual_now - self.last_user_message_virtual).total_seconds() / 60.0
            silence_vminutes = max(0, delta)
        else:
            # Fallback: use real time with speed multiplier
            real_elapsed = now_real - self.last_user_message_real
            speed = self.time_ctrl.speed_multiplier if self.time_ctrl else 1.0
            silence_vminutes = (real_elapsed * speed) / 60.0

        # Determine current phase
        current_phase = 0
        for i, threshold in enumerate(self.PHASE_THRESHOLDS):
            if silence_vminutes >= threshold:
                current_phase = i

        # No effect in phase 0
        if current_phase == 0:
            return None

        # Only apply effects when entering a new phase (not every tick)
        # But also apply gradual effects within a phase every ~30 virtual minutes
        self._silence_tick_count += 1
        should_apply = False

        if current_phase > self._last_phase_applied:
            # New phase entered
            should_apply = True
            self._last_phase_applied = current_phase

        elif self._silence_tick_count % 10 == 0:
            # Gradual effect: apply reduced emotion every ~30 seconds of real time
            # (10 ticks * 3 seconds = 30 seconds)
            should_apply = True

        if not should_apply:
            return None

        # Get phase config
        profile = self.PROFILES.get(self.profile_key, self.PROFILES["secure"])
        phase_key = f"phase_{current_phase}"
        phase_config = profile.get(phase_key, {})

        emotion_changes = phase_config.get("emotion", {})
        pressure_amount = phase_config.get("pressure", 0)
        trigger_chance = phase_config.get("trigger_chance", 0)
        inner_thought = phase_config.get("inner", "")

        # Apply emotional effects (scaled down for gradual application)
        if emotion_changes:
            # Scale: full strength on phase transition, 30% on gradual ticks
            scale = 1.0 if current_phase > (self._last_phase_applied - 1) else 0.3
            scaled = {k: v * scale for k, v in emotion_changes.items()}
            self.state.emotion.apply_stimulus(scaled, reason=f"silence_phase_{current_phase}")

        # Apply pressure
        if pressure_amount > 0:
            burst_result = self.state.pressure.record("no_reply", pressure_amount)
            if burst_result and burst_result.get("burst"):
                # Pressure burst happened! Apply burst emotions
                burst_emotions = burst_result.get("emotion_changes", {})
                if burst_emotions:
                    self.state.emotion.apply_stimulus(
                        burst_emotions, reason="silence_pressure_burst"
                    )
                    print(f"[Silence] Pressure BURST on no_reply! Profile={self.profile_key}, "
                          f"burst_count={burst_result.get('burst_count')}")

        # Check if we should trigger a proactive message (HARD LIMITS)
        # Most real people do NOT keep texting when ignored.
        # Only certain personality types, under high emotion, send at most 1 follow-up.
        if self._should_send_silence_message(current_phase, silence_vminutes):
            self._silence_messages_sent += 1
            print(f"[Silence] Proactive message #{self._silence_messages_sent} "
                  f"(limit={self._silence_msg_limit}, profile={self.profile_key}, "
                  f"phase={current_phase}, energy={self.state.emotion.total_energy():.2f})")
            return {
                "type": "silence_reaction",
                "phase": current_phase,
                "silence_minutes": round(silence_vminutes, 1),
                "profile": self.profile_key,
                "inner_thought": inner_thought,
                "mood_value": self.state.emotion.get_mood_scalar(),
                "active_emotions": self.state.emotion.get_active_emotion_labels(),
                "dominant": self.state.emotion.get_dominant(3),
            }

        return None

    def _should_send_silence_message(self, current_phase, silence_vminutes):
        """
        Hard programmatic check for whether to trigger a silence proactive message.

        Rules:
        1. Must not have exceeded per-personality message limit for this silence episode
        2. Must be at or above the minimum phase for this personality
        3. Must have total emotion energy above the personality threshold
        4. Only triggers on phase transitions (not gradual ticks)

        This replaces the old probabilistic trigger_chance system.
        Real people don't keep messaging when ignored — at most they send ONE
        follow-up, and only if they're emotionally charged enough.
        """
        # Rule 1: Hard message count limit
        if self._silence_messages_sent >= self._silence_msg_limit:
            return False

        # Rule 2: Must be at minimum phase
        if current_phase < self._silence_min_phase:
            return False

        # Rule 3: Must be on a phase transition (not a gradual tick)
        # We only trigger when entering a new phase, not on every tick
        if current_phase <= (self._last_phase_applied - 1):
            return False

        # Rule 4: Emotion energy threshold
        total_energy = self.state.emotion.total_energy()
        if total_energy < self._silence_energy_threshold:
            return False

        return True

    def get_silence_status(self):
        """Get current silence tracking info (for debug display)."""
        now_real = time.time()
        real_elapsed = now_real - self.last_user_message_real

        if self.time_ctrl and self.last_user_message_virtual:
            virtual_now = self.time_ctrl.get_virtual_now()
            delta = (virtual_now - self.last_user_message_virtual).total_seconds() / 60.0
            silence_vmin = max(0, delta)
        else:
            speed = self.time_ctrl.speed_multiplier if self.time_ctrl else 1.0
            silence_vmin = (real_elapsed * speed) / 60.0

        current_phase = 0
        for i, threshold in enumerate(self.PHASE_THRESHOLDS):
            if silence_vmin >= threshold:
                current_phase = i

        return {
            "silence_real_sec": round(real_elapsed, 1),
            "silence_virtual_min": round(silence_vmin, 1),
            "phase": current_phase,
            "profile": self.profile_key,
            "no_reply_pressure": round(self.state.pressure.pressure.get("no_reply", 0), 3),
            "messages_sent": self._silence_messages_sent,
            "message_limit": self._silence_msg_limit,
            "energy_threshold": self._silence_energy_threshold,
            "current_energy": round(self.state.emotion.total_energy(), 2),
        }
