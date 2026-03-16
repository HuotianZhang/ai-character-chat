"""
RelationshipJudge — High-level metacognitive evaluation system.

Maintains three status fields from the character's perspective:
1. relationship_label: How she defines the relationship (陌生人, 普通朋友, 暧昧关系, etc.)
2. relationship_goal: What she wants from this relationship (想成为朋友, 想做恋人, etc.)
3. user_speculation: Her guess about the user's intent (用户想泡我, 用户在忙, etc.)

These are evaluated periodically via LLM, using emotion state, conversation history,
personality, and affinity as input. They serve as high-level criteria for:
- Proactive event decisions (should I message first? what tone?)
- Conversation tone calibration (instinct vs. deliberation)

v2.5: Initial implementation.
"""
import json
import re
import time


# ============================================================
# Constants: Enumerated options for each status field
# ============================================================

# The LLM can return any of these, or something new — we don't hard-restrict
RELATIONSHIP_LABELS = [
    "陌生人", "网友", "普通朋友", "好朋友", "至交",
    "暧昧关系", "朋友已满恋人未达", "恋人", "炮友",
    "前任", "纠缠不清", "混乱", "敌意关系", "利用关系",
    "单方面依赖", "互相折磨",
]

RELATIONSHIP_GOALS = [
    "想保持距离", "想成为朋友", "想做恋人", "想摆脱关系", "想立刻断交",
    "想被关心", "想被虐", "想冷暴力", "想玩弄人", "想恶心人",
    "想控制对方", "想试探对方", "想被追", "想维持现状",
    "想靠近但又害怕", "无所谓", "想报复", "还没想好",
]

USER_SPECULATIONS = [
    "用户想泡我", "用户想陪伴我", "用户想冷暴力我",
    "用户可能在忙所以没回我消息", "用户对我没兴趣",
    "用户在试探我", "用户想控制我", "用户真心关心我",
    "用户在敷衍我", "用户想跟我做朋友", "用户在利用我",
    "用户不知道自己想要什么", "用户在犹豫", "用户想离开我",
    "用户对我有敌意", "看不透用户意图",
]


class RelationshipJudge:
    """
    Periodic metacognitive evaluator.

    Unlike the instinctive emotion/affinity system (which fires every turn),
    this module does a slower, deeper evaluation — like a person stepping back
    to think "wait, what IS this relationship? what do I actually want?"

    Evaluation frequency:
    - After every N conversation turns (default 5)
    - Before every proactive event (mandatory gate)
    - On significant state changes (affinity jump, pressure burst, etc.)
    """

    # How often to re-evaluate during normal conversation
    EVAL_EVERY_N_TURNS = 5

    # Minimum real seconds between evaluations (avoid API spam)
    MIN_EVAL_INTERVAL_SEC = 30

    def __init__(self, character_state):
        self.state = character_state

        # Current judgments
        self.relationship_label = "陌生人"
        self.relationship_goal = "还没想好"
        self.user_speculation = "看不透用户意图"

        # Evaluation tracking
        self._last_eval_turn = 0
        self._last_eval_time = 0
        self._eval_history = []  # Track how judgments evolved
        self._eval_count = 0

    def should_evaluate(self, force=False):
        """Check if it's time for a deep evaluation."""
        if force:
            return True

        now = time.time()
        if now - self._last_eval_time < self.MIN_EVAL_INTERVAL_SEC:
            return False

        turns_since = self.state.interaction_count - self._last_eval_turn
        if turns_since >= self.EVAL_EVERY_N_TURNS:
            return True

        return False

    def evaluate(self, conversation_history, is_silence=False):
        """
        Run a deep evaluation via LLM.
        Returns the updated judgments dict.

        Args:
            conversation_history: list of {"role", "content"} dicts
            is_silence: True if evaluating during a silence period (no recent user message)
        """
        from conversation_engine import call_gemini

        char = self.state.character
        basic = char.get("基础信息", {})
        personality = char.get("性格维度", {})

        # Build a compact state summary for the evaluator
        emotion_axes = self.state.emotion.axes
        dominant = self.state.emotion.get_dominant(3)
        dominant_str = ", ".join(f"{ax}={val:.2f}" for ax, val in dominant) if dominant else "无"
        pressures = self.state.pressure.get_active_pressures()

        # Get recent conversation (last 15 turns max, to stay within token budget)
        recent = conversation_history[-15:] if conversation_history else []
        conv_text = ""
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else basic.get("名字", "我")
            conv_text += f"{role_label}：{msg['content'][:150]}\n"

        if not conv_text.strip():
            conv_text = "（还没有对话记录）"

        # If in silence, emphasize that
        silence_note = ""
        if is_silence:
            silence_note = "\n【重要】用户目前没有回复你的消息。这本身就是一种信息——请在评估中考虑这一点。"

        prompt = f"""你是{basic.get('名字', '未知')}的内心深处。现在请你跳出当前对话，做一次冷静的自我审视。

## 你的性格
- 依恋模式：{personality.get('依恋模式')}（{personality.get('依恋模式_程度')}/5）
- 外在气场：{personality.get('外在气场')}
- 核心创伤：{personality.get('核心创伤', {}).get('类型', '无')}
- 认知风格：{personality.get('认知风格')}

## 当前状态
- 好感度：{self.state.affinity.affinity}/100
- 特殊好感度：{self.state.affinity.special_affinity}/100
- 主导情绪：{dominant_str}
- 情绪总能量：{self.state.emotion.total_energy():.2f}
- 压力积累：{json.dumps(pressures, ensure_ascii=False) if pressures else '无'}
- 对话轮次：第{self.state.interaction_count}轮
- 上次判断：关系={self.relationship_label}，目标={self.relationship_goal}，猜测={self.user_speculation}
{silence_note}

## 最近对话
{conv_text}

## 请评估以下三项（各用1-4个字概括）

1. **关系定义**：你觉得你和对方现在是什么关系？
   参考选项：{', '.join(RELATIONSHIP_LABELS[:10])}...（也可以自己定义）

2. **关系目标**：你现在想要什么？你希望这段关系往哪个方向发展？
   参考选项：{', '.join(RELATIONSHIP_GOALS[:10])}...（也可以自己定义）

3. **用户意图猜测**：你觉得对方想要什么？
   参考选项：{', '.join(USER_SPECULATIONS[:10])}...（也可以自己定义）

请严格用以下JSON格式回复：
```json
{{
  "relationship_label": "xxx",
  "relationship_goal": "xxx",
  "user_speculation": "xxx",
  "reasoning": "一两句话解释你的判断逻辑"
}}
```"""

        messages = [{"role": "user", "content": prompt}]

        try:
            raw = call_gemini(messages, temperature=0.7, max_tokens=300)

            if raw.startswith("[System:"):
                print(f"[RelJudge] LLM call failed: {raw[:80]}")
                return self.to_dict()

            # Parse response
            result = self._parse_evaluation(raw)
            if result:
                old = {
                    "label": self.relationship_label,
                    "goal": self.relationship_goal,
                    "speculation": self.user_speculation,
                }

                self.relationship_label = result.get("relationship_label", self.relationship_label)
                self.relationship_goal = result.get("relationship_goal", self.relationship_goal)
                self.user_speculation = result.get("user_speculation", self.user_speculation)

                # Track evolution
                self._eval_history.append({
                    "time": time.time(),
                    "turn": self.state.interaction_count,
                    "old": old,
                    "new": {
                        "label": self.relationship_label,
                        "goal": self.relationship_goal,
                        "speculation": self.user_speculation,
                    },
                    "reasoning": result.get("reasoning", ""),
                    "is_silence": is_silence,
                })
                if len(self._eval_history) > 30:
                    self._eval_history = self._eval_history[-30:]

                self._eval_count += 1

                print(f"[RelJudge] Eval #{self._eval_count}: "
                      f"关系={self.relationship_label}, "
                      f"目标={self.relationship_goal}, "
                      f"猜测={self.user_speculation}")

            self._last_eval_turn = self.state.interaction_count
            self._last_eval_time = time.time()

        except Exception as e:
            print(f"[RelJudge] Evaluation error: {e}")

        return self.to_dict()

    def _parse_evaluation(self, raw):
        """Parse the LLM evaluation response."""
        # Try ```json block first
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try bare JSON
        json_match = re.search(r'\{[\s\S]*"relationship_label"[\s\S]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        print(f"[RelJudge] Failed to parse evaluation: {raw[:100]}")
        return None

    def evaluate_for_proactive(self, conversation_history, silence_minutes=0):
        """
        Mandatory evaluation before a proactive event.
        Returns a decision dict:
        {
            "should_send": bool,
            "tone": str,        # e.g. "冷淡", "热情", "试探", "攻击性"
            "reasoning": str,
            "judgments": {...}   # the 3 status fields
        }
        """
        from conversation_engine import call_gemini

        # First, update the 3 status fields
        is_silence = silence_minutes > 0
        self.evaluate(conversation_history, is_silence=is_silence)

        char = self.state.character
        basic = char.get("基础信息", {})
        personality = char.get("性格维度", {})

        # Now ask: given these judgments, should I send a message?
        dominant = self.state.emotion.get_dominant(3)
        dominant_str = ", ".join(f"{ax}={val:.2f}" for ax, val in dominant) if dominant else "无"

        silence_context = ""
        if silence_minutes > 0:
            silence_context = f"\n用户已经{silence_minutes:.0f}分钟没有回复。这意味着用户没有在回你消息。"

        prompt = f"""你是{basic.get('名字', '未知')}。你正在考虑要不要主动给对方发消息。

## 你对这段关系的判断
- 你们的关系：{self.relationship_label}
- 你想要的：{self.relationship_goal}
- 你觉得对方：{self.user_speculation}

## 当前情绪状态
- 主导情绪：{dominant_str}
- 情绪能量：{self.state.emotion.total_energy():.2f}
- 好感度：{self.state.affinity.affinity}/100
- 依恋模式：{personality.get('依恋模式')}
{silence_context}

## 决策
根据以上信息，你要做两个决定：
1. **要不要发消息**：考虑你的性格、关系状态和目标。如果你的目标是"想摆脱关系"或"想冷暴力"，你可能不想发。如果对方不回你消息而你是回避型，你也不想发。
2. **如果发，用什么语气**：根据你的情绪和关系判断选择语气。

请用JSON格式回复：
```json
{{
  "should_send": true或false,
  "tone": "语气描述（1-4字，如：冷淡、热情、试探、讽刺、撒娇、攻击性、假装无所谓等）",
  "reasoning": "一句话解释为什么"
}}
```"""

        messages = [{"role": "user", "content": prompt}]

        try:
            raw = call_gemini(messages, temperature=0.7, max_tokens=200)

            if raw.startswith("[System:"):
                print(f"[RelJudge] Proactive decision failed: {raw[:80]}")
                return {
                    "should_send": True,  # Default to sending on API failure
                    "tone": "自然",
                    "reasoning": "评估失败，默认发送",
                    "judgments": self.to_dict(),
                }

            decision = self._parse_decision(raw)
            if decision:
                decision["judgments"] = self.to_dict()
                print(f"[RelJudge] Proactive decision: send={decision.get('should_send')}, "
                      f"tone={decision.get('tone', '?')}")
                return decision

        except Exception as e:
            print(f"[RelJudge] Proactive decision error: {e}")

        return {
            "should_send": True,
            "tone": "自然",
            "reasoning": "评估异常，默认发送",
            "judgments": self.to_dict(),
        }

    def _parse_decision(self, raw):
        """Parse proactive decision response."""
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                data["should_send"] = bool(data.get("should_send", True))
                return data
            except json.JSONDecodeError:
                pass

        json_match = re.search(r'\{[\s\S]*"should_send"[\s\S]*\}', raw)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                data["should_send"] = bool(data.get("should_send", True))
                return data
            except json.JSONDecodeError:
                pass

        return None

    def get_prompt_injection(self):
        """
        Returns a system prompt fragment to inject the relationship judgments
        into the conversation engine's system prompt.
        """
        return f"""
## 【你对这段关系的深层判断】
- 你认为你们的关系是：{self.relationship_label}
- 你对这段关系的期望/目标：{self.relationship_goal}
- 你对对方意图的猜测：{self.user_speculation}
这些判断会影响你的措辞、语气和主动性。比如：
- 如果你觉得对方在"冷暴力你"，你可能会更防御或攻击性
- 如果你觉得对方"想泡你"而你的目标是"想保持距离"，你会更冷淡
- 如果你觉得关系是"暧昧"而你"想做恋人"，你会更暧昧更主动
"""

    def to_dict(self):
        return {
            "relationship_label": self.relationship_label,
            "relationship_goal": self.relationship_goal,
            "user_speculation": self.user_speculation,
            "eval_count": self._eval_count,
            "last_eval_turn": self._last_eval_turn,
            "eval_history": self._eval_history[-10:],
        }

    @classmethod
    def from_dict(cls, d, character_state):
        obj = cls(character_state)
        obj.relationship_label = d.get("relationship_label", "陌生人")
        obj.relationship_goal = d.get("relationship_goal", "还没想好")
        obj.user_speculation = d.get("user_speculation", "看不透用户意图")
        obj._eval_count = d.get("eval_count", 0)
        obj._last_eval_turn = d.get("last_eval_turn", 0)
        obj._eval_history = d.get("eval_history", [])
        return obj
