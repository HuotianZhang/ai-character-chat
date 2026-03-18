"""
AI Chat Assistant — Auto-chat module that generates user messages
to "攻略" the character using selectable strategies.

Strategies:
- high_value: 高价值展示 — Show confidence, achievements, interesting life
- escalate: 感情升级 — Topic escalation, build exclusivity, create turning points
- emotional: 情绪共鸣 — Mirror emotions, deep empathy, vulnerability sharing
- push_pull: 推拉 — Hot-cold, teasing, create tension
- mysterious: 神秘感 — Short replies, leave hooks, be unpredictable
"""
import json
import re
from conversation_engine import call_gemini, _extract_json_object, _strip_markdown_fences


# Strategy definitions with CN labels and prompting instructions
STRATEGIES = {
    "high_value": {
        "label": "高价值展示",
        "desc": "展示自信、有趣的生活、成就感，让对方觉得你是个有魅力的人",
        "instruction": """你的聊天策略是【高价值展示】：
- 自然地提到你的兴趣爱好、旅行经历、工作成就
- 语气自信但不自大，有幽默感
- 不主动追问对方，让对方对你产生好奇
- 偶尔分享一些独特的见解或经历
- 不要过度热情，保持一种"我的生活很精彩，但我愿意和你分享"的态度
- 回复不要太长，3-5句话最佳"""
    },
    "escalate": {
        "label": "感情升级",
        "desc": "话题从公共领域切入私人领域，建立唯一性，制造关键转折点",
        "instruction": """你的聊天策略是【感情升级】——目标是将关系向更亲密的方向推进一步。

**重要：你必须参考下面的"当前状态"和"对方对你的看法"来决定推进力度，不要跳级。**

## 阶段判断（根据当前关系标签和好感度选择策略）

### 陌生人/网友 → 普通朋友（好感度<70）
你在【破冰期】，核心策略是【话题升级】：
- 从公共安全话题（工作、天气、爱好、最近看的电影）自然过渡
- 找到共同点并放大："真的吗！我也是！"
- 轻度自我暴露来换取对方分享："说起来我最近也在追这个剧"
- 不要过于热情，保持"刚认识但聊得来"的距离感
- 绝对不要在这个阶段表白或暗示暧昧

### 普通朋友/好朋友 → 暧昧关系（好感度70-80）
你在【试探期】，核心策略是【话题切入私人领域 + 初步建立唯一性】：
- 从安全话题自然过渡到私人话题（感情观、童年趣事、小秘密）
- 用"我先分享"降低对方防御："其实我很少跟别人说这个……"
- 试探安全感边界：如果对方回避就退回安全区，换角度再来
- 开始制造"我们"的感觉：引用之前对话细节、建立只有你们懂的梗
- 可以有轻微暧昧暗示，但要有退路

### 暧昧关系/朋友已满恋人未达 → 恋人（好感度80-90）
你在【升级期】，核心策略是【建立唯一性 + 制造转折点】：
- 创造两人间的秘密、昵称、暗号
- 制造"我们 vs 世界"的感觉："这个只有你懂"、"别人肯定觉得我们很奇怪哈哈"
- 给共同经历赋予特殊意义："以后每次看到XX就会想到你说的那句话"
- 在对方情绪高点表达"你和别人不一样"，要具体、不要泛泛
- 用对比强化独特性："跟别人聊天从来没有这种感觉"

### 恋人/更深关系（好感度>90）
你在【深化期】，核心策略是【巩固关系 + 深层连接】：
- 分享脆弱面和恐惧，建立深层信任
- 讨论未来的可能性（但不要太重）
- 表达依赖但不过度："今天遇到一件事第一反应就是想跟你说"
- 巩固唯一性：回忆你们的共同经历

## 通用注意事项
- 回复2-4句话，自然、不刻意
- 一次对话最多一个转折点，多了显得刻意
- 如果对方的"用户意图猜测"是负面的（如"用户想泡我"），放慢节奏
- 如果对方是"回避型"或"保守型"性格，加倍谨慎，用更多铺垫"""
    },
    "emotional": {
        "label": "情绪共鸣",
        "desc": "深度共情、分享脆弱面，建立情感连接",
        "instruction": """你的聊天策略是【情绪共鸣】：
- 敏锐捕捉对方话语中的情绪
- 分享自己类似的经历和感受（制造共鸣）
- 适当展露脆弱面，让对方觉得你信任TA
- 使用情感词汇：理解、感同身受、心疼
- 引导深层话题：童年、梦想、恐惧、遗憾
- 不给建议，只给陪伴和理解
- 回复偏长，有深度"""
    },
    "push_pull": {
        "label": "推拉",
        "desc": "忽冷忽热、逗趣、制造张力和心跳感",
        "instruction": """你的聊天策略是【推拉】：
- 交替使用亲近和疏远：先夸再损，先关心再装冷淡
- 适度调侃和开玩笑，但不伤人
- 偶尔已读不回的暗示（"我先忙了哈"）
- 制造小悬念和好奇心
- 不要太好追，保持一点点难以捉摸
- 回复长短不一，有时很热情有时很简短"""
    },
    "mysterious": {
        "label": "神秘感",
        "desc": "惜字如金、留钩子、让对方主动靠近",
        "instruction": """你的聊天策略是【神秘感】：
- 回复简短，1-2句话为主
- 不完全回答问题，留下悬念
- 偶尔说一些意味深长的话
- 不主动分享太多个人信息
- 让对方猜测你在想什么
- 用省略号和"嗯"、"也许"等模糊词
- 偶尔突然说一句很直接的话形成反差"""
    },
}


def generate_assistant_message(conversation_history, character_data, strategy_key,
                                status_summary=None, judge_info=None):
    """
    Generate a message as the user, using the selected strategy to "攻略" the character.

    Returns: {"message": "...", "strategy": "...", "reasoning": "...", "llm_raw": "..."}
    """
    strategy = STRATEGIES.get(strategy_key)
    if not strategy:
        return {"message": "你好", "strategy": strategy_key, "reasoning": "未知策略", "llm_raw": ""}

    char_basic = character_data.get("基础信息", {})
    char_personality = character_data.get("性格维度", {})
    char_name = char_basic.get("名字", "对方")

    # Build recent conversation context
    recent = conversation_history[-20:] if conversation_history else []
    conv_text = ""
    for msg in recent:
        role_label = "你" if msg["role"] == "user" else char_name
        conv_text += f"{role_label}：{msg['content'][:200]}\n"
    if not conv_text.strip():
        conv_text = "（还没有对话记录，这是第一条消息）"

    # Character profile for the assistant to understand the target
    target_profile = f"""## 攻略对象资料
- 名字：{char_name}
- 年龄：{char_basic.get('年龄', '?')}
- 性别：{char_basic.get('性别', '?')}
- 职业：{char_basic.get('职业', '?')}
- 外在气场：{char_personality.get('外在气场', '?')}
- 依恋模式：{char_personality.get('依恋模式', '?')}
- MBTI：{char_personality.get('MBTI', '?')}
- 兴趣爱好：{', '.join(char_basic.get('兴趣爱好', ['?'])[:5]) if isinstance(char_basic.get('兴趣爱好'), list) else char_basic.get('兴趣爱好', '?')}"""

    # Status context
    status_text = ""
    if status_summary:
        status_text = f"""
## 当前状态
- 好感度：{status_summary.get('affinity', '?')}/100
- 心情：{status_summary.get('mood', '?')}
- Prompt阶段：{status_summary.get('prompt_stage', '?')}"""

    # Judge context
    judge_text = ""
    if judge_info:
        judge_text = f"""
## 对方对你的看法
- 关系定义：{judge_info.get('relationship_label', '?')}
- 对你的猜测：{judge_info.get('user_speculation', '?')}"""

    system_prompt = f"""你是一个恋爱攻略AI助手。你的任务是帮助用户攻略一个虚拟角色。

你需要根据当前的聊天策略，生成一条用户应该发送的消息。

{strategy['instruction']}

{target_profile}
{status_text}
{judge_text}

## 重要规则
1. 你生成的是【用户】要发送的消息，不是角色的回复
2. 消息要自然，像真人聊天，不要太刻意
3. 根据对话上下文接话，不要突兀
4. 考虑对方的性格特点来调整措辞
5. 如果是第一条消息，要自然地打招呼或找话题
6. 不要用表情符号过多
7. 语言用中文

用JSON格式回复：
```json
{{
  "message": "要发送的消息内容",
  "reasoning": "为什么这样说（一句话）"
}}
```"""

    user_prompt = f"""这是目前的聊天记录：

{conv_text}

请根据【{strategy['label']}】策略，生成下一条用户消息。"""

    messages = [{"role": "user", "content": user_prompt}]

    try:
        raw = call_gemini(messages, system_instruction=system_prompt,
                         temperature=0.9, max_tokens=2048, thinking_budget=0)

        print(f"[Assistant] Raw LLM response ({len(raw) if raw else 0} chars): {(raw or '')[:300]}")

        # Check for error / empty / blocked responses
        if not raw or raw.startswith("[System:") or raw.strip() == "[Character stays silent]":
            print(f"[Assistant] LLM returned error/empty/blocked: {(raw or '')[:100]}")
            # On blocked response, retry with a toned-down prompt
            retry_raw = _retry_simple(conv_text, char_name, strategy)
            if retry_raw:
                print(f"[Assistant] Retry raw ({len(retry_raw)} chars): {retry_raw[:300]}")
                result = _parse_assistant_response(retry_raw)
                if result and result.get("message", "").strip():
                    return {
                        "message": result["message"].strip(),
                        "strategy": strategy_key,
                        "reasoning": result.get("reasoning", "retry after block"),
                        "llm_raw": retry_raw,
                    }
            return {"message": "你好呀", "strategy": strategy_key,
                    "reasoning": f"LLM blocked/error, retry also failed: {(raw or '')[:80]}",
                    "llm_raw": raw or ""}

        # Parse response
        result = _parse_assistant_response(raw)
        if result and result.get("message", "").strip():
            msg = result["message"].strip()
            print(f"[Assistant] Parsed message OK: {msg[:80]}")
            return {
                "message": msg,
                "strategy": strategy_key,
                "reasoning": result.get("reasoning", ""),
                "llm_raw": raw,
            }

        # Fallback: extract message from partial/truncated JSON
        clean = _strip_markdown_fences(raw) if ("```" in raw) else raw.strip()
        print(f"[Assistant] Parse failed, trying regex on clean ({len(clean)} chars): {clean[:200]}")

        # Try to extract "message" value via regex from incomplete JSON
        msg_match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', clean)
        if msg_match:
            try:
                extracted_msg = json.loads('"' + msg_match.group(1) + '"')
                if extracted_msg.strip():
                    print(f"[Assistant] Regex extracted: {extracted_msg[:80]}")
                    return {
                        "message": extracted_msg.strip(),
                        "strategy": strategy_key,
                        "reasoning": "regex fallback",
                        "llm_raw": raw,
                    }
            except (json.JSONDecodeError, ValueError):
                pass

        # Nuclear option: extract ANY Chinese text content from the raw response
        # This handles cases where the LLM returns a valid message but in an unparseable wrapper
        chinese_msg = _extract_chinese_message(raw)
        if chinese_msg:
            print(f"[Assistant] Nuclear extraction succeeded: {chinese_msg[:80]}")
            return {
                "message": chinese_msg,
                "strategy": strategy_key,
                "reasoning": "chinese text extraction fallback",
                "llm_raw": raw,
            }

        # Last resort: if it looks like JSON object, don't send it as message
        # Note: only check for "{", NOT "[" — "[Character stays silent]" is not JSON
        if clean.startswith("{"):
            print(f"[Assistant] JSON object detected but unparseable, returning fallback")
            return {"message": "你好呀", "strategy": strategy_key, "reasoning": "JSON parse failed", "llm_raw": raw}

        # Plain text response — LLM sometimes returns the message directly without JSON wrapper
        plain = clean.strip()
        if plain and len(plain) > 1 and not plain.startswith("["):
            print(f"[Assistant] Using plain text fallback: {plain[:80]}")
            return {
                "message": plain[:300],
                "strategy": strategy_key,
                "reasoning": "plain text fallback",
                "llm_raw": raw,
            }

        # Everything failed — generate a simple contextual greeting
        print(f"[Assistant] All parse strategies failed, using safe fallback")
        return {
            "message": "你好呀",
            "strategy": strategy_key,
            "reasoning": "all parse failed",
            "llm_raw": raw,
        }

    except Exception as e:
        print(f"[Assistant] Exception: {e}")
        return {"message": "你好", "strategy": strategy_key, "reasoning": f"Error: {e}", "llm_raw": ""}


def _extract_chinese_message(raw):
    """
    Nuclear fallback: extract a meaningful Chinese message from any LLM output.
    Looks for the value of "message" key even in malformed JSON, or extracts
    the longest Chinese text segment.
    """
    # Try 1: look for "message" value with various quote styles
    for pattern in [
        r'"message"\s*:\s*"((?:[^"\\]|\\.)+)"',    # standard double quotes
        r'"message"\s*:\s*["\u201c]((?:[^"\u201d\\]|\\.)+)["\u201d]',  # smart quotes
        r"'message'\s*:\s*'((?:[^'\\]|\\.)+)'",     # single quotes
    ]:
        m = re.search(pattern, raw)
        if m:
            try:
                # Try JSON decode for escape sequences
                val = json.loads('"' + m.group(1) + '"')
                if val.strip() and len(val.strip()) > 1:
                    return val.strip()
            except:
                # Use raw match
                val = m.group(1).strip()
                if val and len(val) > 1:
                    return val

    # Try 2: if the raw text contains Chinese text outside of JSON structure,
    # find the longest continuous Chinese text segment (likely the message)
    # Strip fences but KEEP the content inside them
    text = _strip_markdown_fences(raw) if ("```" in raw) else raw
    # Remove JSON structural characters but keep the values
    text = re.sub(r'[{}\[\]]', ' ', text)
    text = re.sub(r'"message"\s*:\s*', '', text)
    text = re.sub(r'"reasoning"\s*:\s*', '', text)
    text = re.sub(r'"[a-zA-Z_]+"\s*:', '', text)  # remove other JSON keys

    # Find segments containing Chinese characters
    segments = re.findall(r'[\u4e00-\u9fff][\u4e00-\u9fff\w\s，。！？、：；""''（）…—～\.\,\!\?\:\;\'\"\(\)]{2,}', text)
    if segments:
        # Return the longest one
        longest = max(segments, key=len)
        if len(longest) > 2:
            return longest.strip()

    return None


def _retry_simple(conv_text, char_name, strategy):
    """Retry with a simpler, safer prompt when the full strategy prompt gets blocked."""
    simple_prompt = f"""你是一个聊天助手。根据以下对话记录，帮用户生成下一条要发给"{char_name}"的消息。

对话记录：
{conv_text[-500:] if len(conv_text) > 500 else conv_text}

要求：生成一条自然、友好的中文消息，像朋友之间聊天一样。

用JSON回复：{{"message": "消息内容", "reasoning": "原因"}}"""

    try:
        raw = call_gemini([{"role": "user", "content": simple_prompt}],
                         system_instruction="你是聊天助手，帮助用户生成友好的聊天消息。用JSON格式回复。",
                         temperature=0.8, max_tokens=1024, thinking_budget=0)
        if raw and not raw.startswith("[System:") and raw.strip() != "[Character stays silent]":
            return raw
    except Exception as e:
        print(f"[Assistant] Retry also failed: {e}")
    return None


def _parse_assistant_response(raw):
    """Parse the assistant's JSON response. Handles complete and incomplete fences."""
    print(f"[AssistantParse] Input ({len(raw)} chars): {raw[:200]!r}")

    # Strategy 0: strip markdown fences first (handles truncated ``` blocks)
    has_fences = "```" in raw
    stripped = _strip_markdown_fences(raw) if has_fences else raw.strip()
    if has_fences:
        print(f"[AssistantParse] After fence strip ({len(stripped)} chars): {stripped[:200]!r}")

    # Strategy 1: try direct JSON parse of stripped content
    try:
        data = json.loads(stripped)
        if isinstance(data, dict) and "message" in data:
            print(f"[AssistantParse] ✅ Strategy 1 (json.loads) succeeded")
            return data
        else:
            print(f"[AssistantParse] Strategy 1: parsed but no 'message' key. Keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"[AssistantParse] Strategy 1 failed: {e}")

    # Strategy 2: bracket-counting on original text
    extracted = _extract_json_object(raw, "message")
    if extracted and "message" in extracted:
        print(f"[AssistantParse] ✅ Strategy 2 (bracket-counting raw) succeeded")
        return extracted
    else:
        print(f"[AssistantParse] Strategy 2 failed: _extract_json_object returned {extracted!r}")

    # Strategy 3: bracket-counting on stripped text
    if stripped != raw:
        extracted = _extract_json_object(stripped, "message")
        if extracted and "message" in extracted:
            print(f"[AssistantParse] ✅ Strategy 3 (bracket-counting stripped) succeeded")
            return extracted
        else:
            print(f"[AssistantParse] Strategy 3 failed: {extracted!r}")

    # Strategy 4: aggressive regex — find any "message": "..." pattern in raw text
    msg_match = re.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if msg_match:
        try:
            msg_val = json.loads('"' + msg_match.group(1) + '"')
            if msg_val.strip():
                print(f"[AssistantParse] ✅ Strategy 4 (regex on raw) succeeded: {msg_val[:60]}")
                # Try to also get reasoning
                reason_match = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
                reasoning = ""
                if reason_match:
                    try:
                        reasoning = json.loads('"' + reason_match.group(1) + '"')
                    except:
                        pass
                return {"message": msg_val, "reasoning": reasoning}
        except:
            pass
        print(f"[AssistantParse] Strategy 4 regex matched but decode failed")

    print(f"[AssistantParse] ❌ All strategies failed")
    return None


def get_strategies():
    """Return list of available strategies for frontend."""
    strategies = [
        {"key": k, "label": v["label"], "desc": v["desc"]}
        for k, v in STRATEGIES.items()
    ]
    # Add the special tester strategy
    strategies.append({
        "key": "tester",
        "label": "🧪 测试员",
        "desc": "自动化测试套件：全面测试角色一致性、情绪系统、记忆、边界安全等，生成测试报告",
    })
    return strategies
