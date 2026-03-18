"""
Proactive Event System - Character initiates conversations based on storyline and state.
Monitors virtual time, detects story events, generates proactive messages via LLM.

v2.3: Integrated SilenceMonitor for no-reply emotional reactions.
"""
import time
import json
import re
from collections import deque
from silence_monitor import SilenceMonitor


class ProactiveEventSystem:
    """
    Monitors the character's storyline + state and triggers proactive messages.

    Proactive message types:
    1. STORYLINE: A storyline event just happened, character shares it
    2. MOOD: Character's mood crossed a threshold, they reach out
    3. ROUTINE: Character sends casual messages at certain times (morning greeting etc.)
    4. SILENCE: Character reacts to user not replying (anxiety, frustration, sadness)
    5. FOLLOW_UP: Character follows up on a previous conversation topic
    """

    # Minimum real seconds between proactive messages (to avoid spam)
    MIN_INTERVAL_REAL_SEC = 10
    # Minimum virtual minutes between proactive messages
    MIN_INTERVAL_VIRTUAL_MIN = 30

    def __init__(self, character_state, time_controller):
        self.state = character_state
        self.time_ctrl = time_controller
        self.pending_messages = deque(maxlen=20)  # Queue of pending proactive messages
        self.sent_history = []  # Track what was already sent

        # Silence monitor (NEW)
        self.silence_monitor = SilenceMonitor(character_state, time_controller)

        # Track which events have been processed
        self._processed_events = set()  # "day_slot" keys that we already handled
        self._last_check_real = 0
        self._last_proactive_real = 0
        self._last_proactive_virtual = None

    def check_and_generate(self, conversation_engine):
        """
        Main tick function. Called periodically by the server.
        Checks for events that should trigger proactive messages.
        Returns list of proactive messages ready to send, or empty list.
        """
        now_real = time.time()

        # Rate limit checks (don't check too frequently)
        if now_real - self._last_check_real < 2.0:
            return []
        self._last_check_real = now_real

        # Don't send if too soon after last proactive message
        if now_real - self._last_proactive_real < self.MIN_INTERVAL_REAL_SEC:
            return []

        # Collect triggered events
        events = []

        # 1. Check time transitions (storyline & routine DISABLED — kept for time tracking only)
        transitions = self.time_ctrl.check_transitions()

        if transitions["day_changed"]:
            # Still track day progression for time display
            self.state.storyline.current_day = transitions["new_day"]
            self.state.storyline.current_time_slot = self.time_ctrl.get_time_slot()

        if transitions["slot_changed"]:
            self.state.storyline.current_time_slot = self.time_ctrl.get_time_slot()

        # 2. Silence monitor tick — check if user silence should trigger reactions
        silence_event = self.silence_monitor.tick()
        if silence_event:
            events.append(silence_event)

        # 3. Check mood-based proactive triggers
        mood_event = self._check_mood_trigger()
        if mood_event:
            events.append(mood_event)

        # 4. If we have events, run deep evaluation gate then generate
        result_messages = []
        for event in events:
            msg = self._deliberate_and_generate(event, conversation_engine)
            if msg:
                result_messages.append(msg)
                self._last_proactive_real = now_real
                self._last_proactive_virtual = self.time_ctrl.get_virtual_now()

        return result_messages

    def _deliberate_and_generate(self, event, conversation_engine):
        """
        Deep evaluation gate: before sending any proactive message, the character
        steps back and considers whether to send it, based on:
        - Her relationship judgment (label, goal, speculation about user)
        - Conversation history
        - Whether the user has been silent (and for how long)

        This models the real human process: "I want to message them... but wait,
        should I? What's our relationship? What do I actually want?"
        """
        judge = self.state.judge

        # Determine silence context
        silence_minutes = 0
        silence_status = self.silence_monitor.get_silence_status()
        if silence_status:
            silence_minutes = silence_status.get("silence_virtual_min", 0)

        # For silence reactions, always consider it a silence context
        if event.get("type") == "silence_reaction":
            silence_minutes = max(silence_minutes, event.get("silence_minutes", 30))

        # Run the deep evaluation: updates the 3 status fields + returns send decision
        decision = judge.evaluate_for_proactive(
            conversation_engine.conversation_history,
            silence_minutes=silence_minutes,
        )

        # Check the decision
        should_send = decision.get("should_send", True)
        tone = decision.get("tone", "自然")
        reasoning = decision.get("reasoning", "")

        if not should_send:
            print(f"[Proactive] Deep eval BLOCKED {event['type']} message: {reasoning}")
            # Even though we don't send, the internal emotion changes from silence
            # still apply (already handled by silence_monitor.tick)

            # If user hasn't responded, amplify the emotional impact of not sending
            if silence_minutes > 0 and event.get("type") == "silence_reaction":
                self._amplify_silence_emotion(event)

            return None

        print(f"[Proactive] Deep eval APPROVED {event['type']} message, tone={tone}")

        # Pass the tone to the message generator so it can use it
        event["_deliberation_tone"] = tone
        event["_deliberation_reasoning"] = reasoning
        event["_llm_judge_raw"] = decision.get("llm_judge_raw", "")

        return self._generate_proactive_message(event, conversation_engine)

    def _amplify_silence_emotion(self, event):
        """
        When the character decides NOT to send a message despite wanting to,
        the suppressed impulse causes additional internal emotional buildup.
        This is the "I wanted to text but held back" effect.
        """
        phase = event.get("phase", 1)
        profile = event.get("profile", "secure")

        # Suppression amplifies anxiety and sadness
        suppression_changes = {}
        if profile in ("anxious", "fearful"):
            suppression_changes = {
                "anxiety": 0.03 * phase,
                "sadness": 0.02 * phase,
            }
        elif profile == "avoidant":
            suppression_changes = {
                "disgust": 0.01 * phase,  # mild self-disgust for caring
            }

        if suppression_changes:
            self.state.emotion.apply_stimulus(
                suppression_changes,
                reason=f"想发消息但忍住了(phase {phase})"
            )
            print(f"[Proactive] Suppression emotion amplified: {suppression_changes}")

    def _check_storyline_event(self, day, slot):
        """Check if there's a storyline event for this day+slot that we haven't processed."""
        event_key = f"d{day}_{slot}"
        if event_key in self._processed_events:
            return None

        if day >= len(self.state.storyline.storyline):
            return None

        today_data = self.state.storyline.storyline[day]
        if not today_data:
            return None

        # Find event matching this slot
        for event in today_data.get("events", []):
            if event.get("time_slot") == slot:
                self._processed_events.add(event_key)

                # Apply mood impact from storyline (uses legacy compat method)
                impact = event.get("mood_impact", 0)
                label = event.get("mood_label", "")
                if impact != 0 and label:
                    self.state.emotion.apply_legacy_stimulus(
                        label, impact, reason=event.get("event", "storyline event")
                    )

                # Update storyline state
                self.state.storyline.current_time_slot = slot
                self.state.storyline.current_day = day
                if today_data.get("emotion_stage"):
                    self.state.storyline.emotion_stage = today_data["emotion_stage"]

                return {
                    "type": "storyline",
                    "event_text": event.get("event", ""),
                    "mood_label": label,
                    "mood_impact": impact,
                    "day": day,
                    "slot": slot,
                    "snapshot": today_data.get("snapshot", ""),
                    "hidden_trigger": today_data.get("hidden_trigger", {}),
                }

        return None

    def _check_routine_message(self, new_slot, day):
        """Check if the character should send a routine greeting/message."""
        affinity = self.state.affinity.affinity

        # Only send routine messages at higher affinity
        if affinity < 70:
            return None

        routine_key = f"routine_d{day}_{new_slot}"
        if routine_key in self._processed_events:
            return None

        # Morning greeting (if affinity >= 75)
        if new_slot == "morning" and affinity >= 75:
            self._processed_events.add(routine_key)
            return {
                "type": "routine",
                "routine_kind": "morning_greeting",
                "slot": new_slot,
                "day": day,
            }

        # Evening check-in (if affinity >= 80)
        if new_slot == "evening" and affinity >= 80:
            self._processed_events.add(routine_key)
            return {
                "type": "routine",
                "routine_kind": "evening_checkin",
                "slot": new_slot,
                "day": day,
            }

        # Night goodbye (if affinity >= 85)
        if new_slot == "night" and affinity >= 85:
            self._processed_events.add(routine_key)
            return {
                "type": "routine",
                "routine_kind": "good_night",
                "slot": new_slot,
                "day": day,
            }

        return None

    def _check_mood_trigger(self):
        """Check if mood is extreme enough to trigger proactive outreach."""
        # Use mood scalar from EmotionVector for compatibility
        mood = self.state.emotion.get_mood_scalar()
        affinity = self.state.affinity.affinity

        # Only proactively share feelings at moderate+ affinity
        if affinity < 72:
            return None

        # Prevent duplicate mood triggers
        mood_key = f"mood_{int(mood)}_{int(time.time() / 300)}"  # 5-min windows
        if mood_key in self._processed_events:
            return None

        # Very happy -> share excitement
        if mood >= 8.5:
            self._processed_events.add(mood_key)
            return {
                "type": "mood",
                "mood_kind": "very_happy",
                "mood_value": mood,
                "active_emotions": self.state.emotion.get_active_emotion_labels(),
            }

        # Very sad -> seek comfort
        if mood <= 3.0 and affinity >= 80:
            self._processed_events.add(mood_key)
            return {
                "type": "mood",
                "mood_kind": "very_sad",
                "mood_value": mood,
                "active_emotions": self.state.emotion.get_active_emotion_labels(),
            }

        return None

    def _generate_proactive_message(self, event, conversation_engine):
        """
        Use LLM to generate a proactive message based on the event.
        Returns a dict with unified format: {"reply", "messages", "status", "inner_thought",
        "proactive", "event_type", "time_display", "storyline_event"}.
        """
        from conversation_engine import (call_gemini, parse_llm_response, build_system_prompt,
                                         format_reply_messages, apply_parsed_output)

        char = self.state.character
        basic = char.get("基础信息", {})
        name = basic.get("名字", "Character")

        # Build context for the proactive message
        snapshot = self.state.process_input("")  # Get current state snapshot (empty input)
        system_prompt = build_system_prompt(char, snapshot)

        # Build the proactive instruction
        if event["type"] == "storyline":
            user_prompt = f"""[系统指令 - 主动消息触发]
现在是第{event['day']+1}天的{event['slot']}。你刚刚经历了这件事：
「{event['event_text']}」
这让你的情绪{event['mood_label']}（影响：{event['mood_impact']:+.1f}）。

请根据你的性格和当前心情，主动发一条消息给对方。
注意：
- 你是主动发消息的，不是在回复对方
- 根据好感度决定分享的深度（好感低就简短/日常，好感高可以分享感受）
- 不要直接说"我今天发生了XXX"，要自然，像真人发消息一样
- 可以是分享、抱怨、求安慰、炫耀、随口聊天等

请用标准JSON格式回复。"""

        elif event["type"] == "routine":
            routine_prompts = {
                "morning_greeting": "现在是早上。你想给对方发一条早安消息。根据你的性格，自然地打招呼。",
                "evening_checkin": "现在是傍晚。你想随便跟对方聊聊。根据你的性格，自然地搭话。",
                "good_night": "现在是深夜了。你想跟对方说晚安。根据你的性格，自然地说。",
            }
            kind = event.get("routine_kind", "morning_greeting")
            user_prompt = f"""[系统指令 - 主动消息触发]
{routine_prompts.get(kind, routine_prompts['morning_greeting'])}

注意：
- 你是主动发消息的，不是在回复对方
- 简短自然，不要太刻意
- 根据你的语言指纹和性格来决定风格

请用标准JSON格式回复。"""

        elif event["type"] == "silence_reaction":
            phase = event.get("phase", 1)
            minutes = event.get("silence_minutes", 30)
            inner = event.get("inner_thought", "")
            dominant = event.get("dominant", [])
            dominant_str = ", ".join(f"{ax}({val:.2f})" for ax, val in dominant)

            # Build silence-specific prompt based on phase intensity
            if phase <= 2:
                intensity_hint = "轻微的——可能只是一条试探性的消息，或者一个表情符号，或者分享个无关的东西来看对方是否回复"
            elif phase == 3:
                intensity_hint = "明显的——你可能会直接或间接表达不满/失落，或者用冷淡的语气暗示你注意到对方没回"
            else:
                intensity_hint = "强烈的——你可能已经从焦虑变成愤怒或心寒，回复可能很冲、很冷淡、或者故意表现得无所谓"

            user_prompt = f"""[系统指令 - 沉默反应触发]
对方已经{minutes:.0f}分钟没有回复你的消息了。

你的内心此刻：「{inner}」
你当前主导情绪：{dominant_str or '混杂'}
沉默阶段：第{phase}阶段（{intensity_hint}）

请根据你的性格、依恋模式和当前情绪，主动发一条消息。

重要规则：
- 真实的人不会说"你怎么不回我"（太直白），而是会用各种间接方式表达
- 可能的方式：发一个"?"、分享无关的内容试探、突然冷淡、抱怨某件事（实际是在迁怒）、故意说让对方焦虑的话、或者真的什么都不发只是内心波动
- 如果你的性格是回避型，你可能反而更加沉默，只是内心有变化
- 如果你决定不发消息（比如回避型在低阶段），回复"[Read]"

请用标准JSON格式回复。"""

        elif event["type"] == "mood":
            if event["mood_kind"] == "very_happy":
                user_prompt = f"""[系统指令 - 主动消息触发]
你现在心情特别好（情绪值：{event['mood_value']:.1f}/10），活跃情绪：{', '.join(event.get('active_emotions', []))}。
你想跟对方分享你的好心情。

注意：
- 你是主动发消息的
- 自然表达，不要太夸张
- 根据性格决定分享方式

请用标准JSON格式回复。"""
            else:
                user_prompt = f"""[系统指令 - 主动消息触发]
你现在心情很不好（情绪值：{event['mood_value']:.1f}/10），活跃情绪：{', '.join(event.get('active_emotions', []))}。
你想找对方聊聊，或者只是想有人陪。

注意：
- 你是主动发消息的
- 根据你的依恋模式决定求助方式（可能直接说难过，可能旁敲侧击，可能只是发个省略号）
- 不要太戏剧化

请用标准JSON格式回复。"""
        else:
            return None

        # Inject deliberation tone if available (from _deliberate_and_generate)
        deliberation_tone = event.get("_deliberation_tone", "")
        deliberation_reason = event.get("_deliberation_reasoning", "")
        if deliberation_tone:
            user_prompt += f"""

【深层思考后的决定】
你仔细想了想，决定用「{deliberation_tone}」的语气发这条消息。
原因：{deliberation_reason}
请让你的回复体现这个语气。"""

        # Call LLM
        messages = [{"role": "user", "content": user_prompt}]
        raw = call_gemini(messages, system_instruction=system_prompt, max_tokens=2048, thinking_budget=0)

        if raw.startswith("[System:"):
            print(f"[Proactive] LLM call failed: {raw[:100]}")
            return None

        parsed = parse_llm_response(raw)

        # Update state (shared utility — same path as ConversationEngine.chat)
        apply_parsed_output(self.state, parsed)

        # Add PARSED REPLY to conversation history (not raw JSON!)
        # Storing raw JSON would cause the LLM to echo JSON format in future turns.
        conversation_engine.conversation_history.append({
            "role": "model",
            "content": parsed["reply"],
        })

        # Record inner_thought for RelationshipJudge speculation fuel
        inner = parsed.get("inner_thought", "")
        if inner:
            self.state.judge.record_inner_thought(inner)

        # Format messages (shared utility — same logic as ConversationEngine.chat)
        reply_text = parsed["reply"]
        msg_list = format_reply_messages(reply_text)
        if msg_list is None:
            return None  # [Read] or [Typing...] — skip non-messages

        self.state.save()

        # Build unified response format
        # Core fields match ConversationEngine.chat(): reply, messages, status, inner_thought
        # Extra proactive fields: proactive, event_type, time_display, storyline_event
        result = {
            "reply": reply_text,
            "messages": msg_list,
            "status": self.state.get_status_summary(),
            "inner_thought": parsed.get("inner_thought", ""),
            "llm_raw": raw or "",
            "llm_judge_raw": event.get("_llm_judge_raw", ""),
            "proactive": True,
            "event_type": event["type"],
            "time_display": self.time_ctrl.get_display_date(),
        }

        if event["type"] == "storyline":
            result["storyline_event"] = event.get("event_text", "")

        print(f"[Proactive] Generated {event['type']} message: {reply_text[:60]}...")
        self.sent_history.append({
            "type": event["type"],
            "time": time.time(),
            "virtual_time": self.time_ctrl.get_virtual_now().isoformat(),
            "reply_preview": reply_text[:80],
        })

        return result

    def get_pending(self, conversation_engine):
        """
        Get all pending proactive messages.
        Called by the polling endpoint.
        """
        new_messages = self.check_and_generate(conversation_engine)
        for msg in new_messages:
            self.pending_messages.append(msg)

        # Return and clear pending
        result = list(self.pending_messages)
        self.pending_messages.clear()
        return result

    def to_dict(self):
        return {
            "processed_events": list(self._processed_events),
            "sent_history": self.sent_history[-20:],
        }

    def notify_user_message(self):
        """Call this whenever the user sends a message, to reset silence tracking."""
        self.silence_monitor.on_user_message()

    def get_silence_status(self):
        """Get silence tracking info for debug display."""
        return self.silence_monitor.get_silence_status()

    @classmethod
    def restore(cls, data, character_state, time_controller):
        obj = cls(character_state, time_controller)
        obj._processed_events = set(data.get("processed_events", []))
        obj.sent_history = data.get("sent_history", [])
        return obj
