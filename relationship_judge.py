"""
RelationshipJudge — High-level metacognitive evaluation system.

Maintains two status fields from the character's perspective:
1. relationship_label: How she defines the relationship (陌生人, 普通朋友, 暧昧关系, etc.)
   - Driven by affinity, personality openness, and long-term memory
   - Conservative personalities require high affinity + explicit user confirmation
   - Open personalities shift labels based on affinity thresholds alone
   - Some personalities change labels chaotically
2. user_speculation: Her guess about the user's intent (用户想泡我, 用户在忙, etc.)
   - Driven by inner_thought outputs, personality, and memory
   - Deliberately aggressive/paranoid guessing based on personality traits

v3.0: Removed relationship_goal. Rewrote label logic (affinity+personality-driven).
      Rewrote speculation logic (inner_thought-driven, more aggressive).
      Hardened output format to prevent frontend errors.
"""
import json
import re
import time


# ============================================================
# Constants
# ============================================================

RELATIONSHIP_LABELS = [
    "陌生人", "网友", "普通朋友", "好朋友", "至交",
    "暧昧关系", "朋友已满恋人未达", "恋人", "炮友",
    "前任", "纠缠不清", "混乱", "敌意关系", "利用关系",
    "单方面依赖", "互相折磨",
]

USER_SPECULATIONS = [
    "用户想泡我", "用户想陪伴我", "用户想冷暴力我",
    "用户可能在忙所以没回我消息", "用户对我没兴趣",
    "用户在试探我", "用户想控制我", "用户真心关心我",
    "用户在敷衍我", "用户想跟我做朋友", "用户在利用我",
    "用户不知道自己想要什么", "用户在犹豫", "用户想离开我",
    "用户对我有敌意", "看不透用户意图",
]

# Personality openness categories — determines how easily relationship_label shifts
# "conservative": needs high affinity + explicit mutual confirmation to change label
# "open": changes label based on affinity thresholds alone
# "chaotic": label can shift unpredictably, sometimes contradicting affinity
PERSONALITY_OPENNESS_MAP = {
    # Outer aura → openness
    "温柔治愈型": "open",
    "甜美元气少女型": "open",
    "冷艳疏离型": "conservative",
    "文艺忧郁型": "conservative",
    "野性叛逆型": "chaotic",
    "知性优雅型": "conservative",
    "天真烂漫型": "open",
    "端庄大气型": "conservative",
    "鬼马精灵型": "chaotic",
    "厌世慵懒型": "chaotic",
}

# Attachment style → speculation aggressiveness
# "paranoid": assumes worst-case, projects past trauma
# "projective": reads too much into small cues
# "balanced": roughly accurate but biased by mood
SPECULATION_STYLE_MAP = {
    "安全·依恋": "balanced", "安全·松弛自洽": "balanced", "安全·主动滋养": "balanced",
    "焦虑·依恋": "paranoid", "焦虑·讨好牺牲": "projective", "焦虑·情绪化施压": "paranoid",
    "回避·安全": "balanced", "回避·焦虑": "projective", "回避·依恋": "projective",
    "回避·理想化远方": "projective",
    "恐惧回避·推拉矛盾": "paranoid", "恐惧回避·自毁测试": "paranoid",
}


class RelationshipJudge:
    """
    Periodic metacognitive evaluator.

    Unlike the instinctive emotion/affinity system (which fires every turn),
    this module does a slower, deeper evaluation — like a person stepping back
    to think "wait, what IS this relationship? what does the user actually want?"

    v3.0: Two fields only (label + speculation). No relationship_goal.
    """

    EVAL_EVERY_N_TURNS = 5
    MIN_EVAL_INTERVAL_SEC = 30

    def __init__(self, character_state):
        self.state = character_state

        # Current judgments (2 fields only)
        self.relationship_label = "陌生人"
        self.user_speculation = "看不透用户意图"

        # Determine personality traits for label/speculation logic
        char = character_state.character if hasattr(character_state, 'character') else {}
        personality = char.get("性格维度", {})
        aura = personality.get("外在气场", "")
        attachment = personality.get("依恋模式", "安全·依恋")

        self.openness = PERSONALITY_OPENNESS_MAP.get(aura, "balanced")
        self.speculation_style = SPECULATION_STYLE_MAP.get(attachment, "balanced")

        # Evaluation tracking
        self._last_eval_turn = 0
        self._last_eval_time = 0
        self._eval_history = []
        self._eval_count = 0

        # Store recent inner_thoughts for speculation fuel
        self._recent_inner_thoughts = []

        # Transient debug fields (not persisted)
        self._last_eval_raw = ""
        self._last_judge_raw = ""

    def record_inner_thought(self, thought):
        """Record an inner_thought from LLM output for speculation reference."""
        if thought and thought.strip():
            self._recent_inner_thoughts.append(thought.strip())
            if len(self._recent_inner_thoughts) > 20:
                self._recent_inner_thoughts = self._recent_inner_thoughts[-20:]

    def should_evaluate(self, force=False):
        """Check if it's time for a deep evaluation."""
        if force:
            return True
        now = time.time()
        if now - self._last_eval_time < self.MIN_EVAL_INTERVAL_SEC:
            return False
        turns_since = self.state.interaction_count - self._last_eval_turn
        return turns_since >= self.EVAL_EVERY_N_TURNS

    def evaluate(self, conversation_history, is_silence=False):
        """
        Run a deep evaluation via LLM.
        Updates relationship_label and user_speculation.
        """
        from conversation_engine import call_gemini, _extract_json_object

        char = self.state.character
        basic = char.get("基础信息", {})
        personality = char.get("性格维度", {})

        # State summary
        dominant = self.state.emotion.get_dominant(3)
        dominant_str = ", ".join(f"{ax}={val:.2f}" for ax, val in dominant) if dominant else "无"
        pressures = self.state.pressure.get_active_pressures()
        affinity = self.state.affinity.affinity
        special_affinity = self.state.affinity.special_affinity

        # Recent conversation (last 15 turns)
        recent = conversation_history[-15:] if conversation_history else []
        conv_text = ""
        for msg in recent:
            role_label = "用户" if msg["role"] == "user" else basic.get("名字", "我")
            conv_text += f"{role_label}：{msg['content'][:150]}\n"
        if not conv_text.strip():
            conv_text = "（还没有对话记录）"

        # Long-term memory summary
        long_term = self.state.memory.long_term[-10:] if hasattr(self.state, 'memory') else []
        memory_text = ""
        for mem in long_term:
            content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
            if content:
                memory_text += f"- {content[:80]}\n"
        if not memory_text:
            memory_text = "（无长期记忆）"

        # Inner thoughts for speculation
        inner_text = "\n".join(f"- {t[:80]}" for t in self._recent_inner_thoughts[-10:])
        if not inner_text:
            inner_text = "（无近期内心独白）"

        silence_note = ""
        if is_silence:
            silence_note = "\n【注意】用户当前没有回复消息。"

        # Personality-specific label instruction
        if self.openness == "conservative":
            label_instruction = f"""你是保守型性格。关系标签变化规则：
- 好感<70: 只能是陌生人或网友
- 好感70-80: 可以升级到普通朋友，但需要对方有明确的友好表示
- 好感80-90: 可以是好朋友，但暧昧/恋人必须对方先表态
- 好感90+: 只有对方明确表白且你接受了才能是恋人
- 关系标签一旦确定不会轻易变化"""
        elif self.openness == "open":
            label_instruction = f"""你是开放型性格。关系标签变化规则：
- 好感<65: 陌生人或网友
- 好感65-75: 自然过渡到普通朋友
- 好感75-85: 会主动感觉到暧昧气氛，标签可能是暧昧关系
- 好感85+: 心里已经把对方当很亲近的人了
- 标签随心情和互动质量流动"""
        else:  # chaotic
            label_instruction = f"""你是情绪化/混乱型性格。关系标签变化规则：
- 标签可能因为一句话突然变化（比如上一秒是好朋友下一秒变成敌意关系）
- 情绪好时可能高估关系，情绪差时可能低估
- 矛盾标签是正常的（比如又爱又恨=纠缠不清）
- 不稳定本身就是你的特征"""

        # Speculation style instruction
        if self.speculation_style == "paranoid":
            spec_instruction = """你的揣测风格是【偏执型】：
- 倾向于往最坏的方向猜测用户意图
- 用户不回消息→一定是在冷暴力/不在乎你
- 用户太好了→可能是在骗你/有目的
- 过去的创伤经历会强烈投射到当前判断上
- 小细节会被放大解读"""
        elif self.speculation_style == "projective":
            spec_instruction = """你的揣测风格是【投射型】：
- 会把自己的感受投射给用户（你焦虑→觉得用户也在纠结）
- 容易过度解读用户的措辞和回复速度
- 对暧昧信号特别敏感，可能在没有的地方看到暗示
- 不确定时倾向于犹豫和反复推测"""
        else:
            spec_instruction = """你的揣测风格是【相对理性】：
- 大体能客观判断，但会受当前情绪影响
- 情绪好时倾向于正面解读
- 情绪差时会稍微消极
- 不会过度脑补但也不是完全不猜"""

        prompt = f"""你是{basic.get('名字', '未知')}的内心深处。做一次冷静的自我审视。

## 你的性格
- 依恋模式：{personality.get('依恋模式')}（{personality.get('依恋模式_程度')}/5）
- 外在气场：{personality.get('外在气场')}
- 核心创伤：{personality.get('核心创伤', {}).get('类型', '无')}

## 当前状态
- 好感度：{affinity}/100 | 特殊好感度：{special_affinity}/100
- 主导情绪：{dominant_str} | 总能量：{self.state.emotion.total_energy():.2f}
- 压力：{json.dumps(pressures, ensure_ascii=False) if pressures else '无'}
- 对话轮次：第{self.state.interaction_count}轮
- 上次判断：关系={self.relationship_label}，猜测={self.user_speculation}
{silence_note}

## 长期记忆
{memory_text}

## 你最近的内心独白
{inner_text}

## 最近对话
{conv_text}

## 关系标签评估规则
{label_instruction}

## 用户意图揣测规则
{spec_instruction}

请评估以下两项（各用1-4个字概括）：

1. **关系定义**：你和对方现在是什么关系？（好感度{affinity}，参考上述规则）
   参考选项：{', '.join(RELATIONSHIP_LABELS[:10])}...

2. **用户意图猜测**：结合你的内心独白和揣测风格，对方到底想干嘛？
   参考选项：{', '.join(USER_SPECULATIONS[:10])}...

严格用以下JSON格式回复，不要有其他文字：
```json
{{
  "relationship_label": "xxx",
  "user_speculation": "xxx",
  "reasoning": "一句话解释"
}}
```"""

        messages = [{"role": "user", "content": prompt}]

        try:
            raw = call_gemini(messages, temperature=0.7, max_tokens=512, thinking_budget=0)
            self._last_eval_raw = raw or ""

            if raw.startswith("[System:"):
                print(f"[RelJudge] LLM call failed: {raw[:80]}")
                return self.to_dict()

            # Parse with robust extraction
            result = self._parse_evaluation(raw)
            if result:
                old = {
                    "label": self.relationship_label,
                    "speculation": self.user_speculation,
                }

                self.relationship_label = str(result.get("relationship_label", self.relationship_label))[:20]
                self.user_speculation = str(result.get("user_speculation", self.user_speculation))[:30]

                self._eval_history.append({
                    "time": time.time(),
                    "turn": self.state.interaction_count,
                    "old": old,
                    "new": {
                        "label": self.relationship_label,
                        "speculation": self.user_speculation,
                    },
                    "reasoning": str(result.get("reasoning", ""))[:100],
                    "is_silence": is_silence,
                })
                if len(self._eval_history) > 30:
                    self._eval_history = self._eval_history[-30:]

                self._eval_count += 1
                print(f"[RelJudge] Eval #{self._eval_count}: "
                      f"关系={self.relationship_label}, "
                      f"猜测={self.user_speculation}")

            self._last_eval_turn = self.state.interaction_count
            self._last_eval_time = time.time()

        except Exception as e:
            print(f"[RelJudge] Evaluation error: {e}")

        return self.to_dict()

    def _parse_evaluation(self, raw):
        """Parse the LLM evaluation response with robust fallback."""
        from conversation_engine import _extract_json_object

        # Strategy 1: ```json block
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "relationship_label" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 2: bracket-counting extraction
        extracted = _extract_json_object(raw, "relationship_label")
        if extracted:
            return extracted

        # Strategy 3: bare JSON regex
        json_match = re.search(r'\{[^}]*"relationship_label"[^}]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        print(f"[RelJudge] Failed to parse evaluation: {raw[:100]}")
        return None

    def evaluate_for_proactive(self, conversation_history, silence_minutes=0):
        """
        Mandatory evaluation before a proactive event.
        Returns a decision dict with guaranteed valid format:
        {
            "should_send": bool,
            "tone": str,
            "reasoning": str,
            "judgments": {...}
        }
        """
        from conversation_engine import call_gemini, _extract_json_object

        # First, update the 2 status fields
        is_silence = silence_minutes > 0
        self.evaluate(conversation_history, is_silence=is_silence)

        char = self.state.character
        basic = char.get("基础信息", {})
        personality = char.get("性格维度", {})

        dominant = self.state.emotion.get_dominant(3)
        dominant_str = ", ".join(f"{ax}={val:.2f}" for ax, val in dominant) if dominant else "无"

        silence_context = ""
        if silence_minutes > 0:
            silence_context = f"\n用户已经{silence_minutes:.0f}分钟没有回复。"

        prompt = f"""你是{basic.get('名字', '未知')}。你正在考虑要不要主动给对方发消息。

## 当前判断
- 关系：{self.relationship_label}
- 你觉得对方：{self.user_speculation}

## 状态
- 情绪：{dominant_str}（能量{self.state.emotion.total_energy():.2f}）
- 好感度：{self.state.affinity.affinity}/100
- 依恋模式：{personality.get('依恋模式')}
{silence_context}

## 决策
1. 要不要发消息？（考虑你的关系判断和对用户的猜测）
2. 如果发，用什么语气？

用JSON格式回复：
```json
{{
  "should_send": true,
  "tone": "语气（1-4字）",
  "reasoning": "一句话原因"
}}
```"""

        messages = [{"role": "user", "content": prompt}]

        # Default safe result — always returns valid format
        safe_default = {
            "should_send": True,
            "tone": "自然",
            "reasoning": "默认发送",
            "judgments": self.to_dict(),
            "llm_judge_raw": "",
        }

        try:
            raw = call_gemini(messages, temperature=0.7, max_tokens=512, thinking_budget=0)
            self._last_judge_raw = raw or ""

            if raw.startswith("[System:"):
                print(f"[RelJudge] Proactive decision failed: {raw[:80]}")
                safe_default["llm_judge_raw"] = raw or ""
                return safe_default

            decision = self._parse_decision(raw)
            if decision:
                # Ensure all required fields exist with correct types
                decision["should_send"] = bool(decision.get("should_send", True))
                decision["tone"] = str(decision.get("tone", "自然"))[:20]
                decision["reasoning"] = str(decision.get("reasoning", ""))[:100]
                decision["judgments"] = self.to_dict()
                decision["llm_judge_raw"] = raw or ""
                print(f"[RelJudge] Proactive decision: send={decision['should_send']}, "
                      f"tone={decision['tone']}")
                return decision

        except Exception as e:
            print(f"[RelJudge] Proactive decision error: {e}")

        return safe_default

    def _parse_decision(self, raw):
        """Parse proactive decision response with robust fallback."""
        from conversation_engine import _extract_json_object

        # Strategy 1: ```json block
        json_match = re.search(r'```json\s*(.*?)\s*```', raw, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if isinstance(data, dict) and "should_send" in data:
                    return data
            except (json.JSONDecodeError, ValueError):
                pass

        # Strategy 2: bracket-counting extraction
        extracted = _extract_json_object(raw, "should_send")
        if extracted:
            return extracted

        # Strategy 3: bare regex
        json_match = re.search(r'\{[^}]*"should_send"[^}]*\}', raw)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except (json.JSONDecodeError, ValueError):
                pass

        print(f"[RelJudge] Failed to parse decision: {raw[:100]}")
        return None

    def get_prompt_injection(self):
        """Returns a system prompt fragment for conversation engine."""
        return f"""## 【你对这段关系的判断】
- 你认为你们的关系是：{self.relationship_label}
- 你对对方意图的猜测：{self.user_speculation}
这些判断影响你的措辞和态度。"""

    def to_dict(self):
        return {
            "relationship_label": self.relationship_label,
            "user_speculation": self.user_speculation,
            "eval_count": self._eval_count,
            "last_eval_turn": self._last_eval_turn,
            "eval_history": self._eval_history[-10:],
            "openness": self.openness,
            "speculation_style": self.speculation_style,
        }

    @classmethod
    def from_dict(cls, d, character_state):
        obj = cls(character_state)
        obj.relationship_label = d.get("relationship_label", "陌生人")
        obj.user_speculation = d.get("user_speculation", "看不透用户意图")
        obj._eval_count = d.get("eval_count", 0)
        obj._last_eval_turn = d.get("last_eval_turn", 0)
        obj._eval_history = d.get("eval_history", [])
        return obj
