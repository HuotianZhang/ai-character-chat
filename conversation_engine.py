"""
Conversation Engine - Connect character state with LLM, build prompts, parse structured output
Support Gemini API (new google-genai SDK + HTTP fallback)

v2.2: Migrated from deprecated google.generativeai to google.genai SDK
v2.1: Multi-axis emotion vector support, tension state injection, pressure awareness
"""
import json
import re
import time
import traceback
from config import GEMINI_API_KEY, GEMINI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS

# Try to import new google-genai SDK first, then fall back to HTTP
_genai_client = None
USE_SDK = False

try:
    from google import genai
    from google.genai import types as genai_types
    _genai_client = genai.Client(api_key=GEMINI_API_KEY)
    USE_SDK = True
    print("[Engine] Using google-genai SDK (new unified SDK)")
except ImportError:
    try:
        # Fall back to deprecated SDK (will show deprecation warning)
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=GEMINI_API_KEY)
        USE_SDK = "legacy"
        print("[Engine] Using deprecated google-generativeai SDK — please run: pip install google-genai")
    except ImportError:
        import requests as _requests
        USE_SDK = False
        print("[Engine] No SDK installed, using HTTP fallback (recommended: pip install google-genai)")
        from config import GEMINI_BASE_URL


def call_gemini(messages, system_instruction="", temperature=None, max_tokens=None, thinking_budget=None):
    """
    Call Gemini API (with auto-retry, prefer new SDK)
    messages: [{"role": "user"/"model", "content": "..."}]
    thinking_budget: int or None. Set to 0 to disable thinking (for structured output).
                     None = let the model decide.
    """
    print(f"[LLM] Calling Gemini ({GEMINI_MODEL}), messages={len(messages)}, system_prompt_len={len(system_instruction)}, thinking_budget={thinking_budget}")
    if USE_SDK is True:
        return _call_via_new_sdk(messages, system_instruction, temperature, max_tokens, thinking_budget)
    elif USE_SDK == "legacy":
        return _call_via_legacy_sdk(messages, system_instruction, temperature, max_tokens, thinking_budget)
    else:
        return _call_via_http(messages, system_instruction, temperature, max_tokens, thinking_budget)


def _call_via_new_sdk(messages, system_instruction="", temperature=None, max_tokens=None, thinking_budget=None):
    """Call via new google-genai SDK (google.genai.Client)"""
    temp = temperature or LLM_TEMPERATURE
    max_tok = max_tokens or LLM_MAX_TOKENS

    try:
        # Build config
        config_kwargs = {
            "system_instruction": system_instruction if system_instruction else None,
            "temperature": temp,
            "max_output_tokens": max_tok,
        }
        # Disable or limit thinking for structured output calls
        if thinking_budget is not None:
            try:
                config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )
                print(f"[LLM SDK] Thinking budget set to {thinking_budget}")
            except Exception as e:
                print(f"[LLM SDK] ThinkingConfig not supported: {e}")
        config = genai_types.GenerateContentConfig(**config_kwargs)

        # Build contents list for the API
        # The new SDK accepts a list of Content objects or dicts
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(genai_types.Content(
                role=role,
                parts=[genai_types.Part(text=msg["content"])]
            ))

        max_retries = 5
        last_error = None
        for attempt in range(max_retries):
            try:
                response = _genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=contents,
                    config=config,
                )
                text = response.text
                if text:
                    return text
                return "[Character stays silent]"

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                print(f"[LLM SDK] Attempt {attempt+1} error: {type(e).__name__}: {str(e)[:200]}")

                if "429" in str(e) or "resource" in err_str or "quota" in err_str or "rate" in err_str:
                    wait_sec = min(2 ** attempt * 5, 60)
                    print(f"[LLM SDK] Rate limited, waiting {wait_sec}s before retry...")
                    time.sleep(wait_sec)
                    continue
                elif "500" in str(e) or "503" in str(e) or "unavailable" in err_str:
                    wait_sec = min(2 ** attempt * 3, 30)
                    print(f"[LLM SDK] Server error, waiting {wait_sec}s before retry...")
                    time.sleep(wait_sec)
                    continue
                else:
                    return f"[System: API error - {type(e).__name__}: {str(e)[:150]}]"

        return f"[System: Failed after {max_retries} retries - {type(last_error).__name__}: {str(last_error)[:150]}]"

    except Exception as e:
        traceback.print_exc()
        return f"[System: SDK initialization failed - {type(e).__name__}: {str(e)[:150]}]"


def _call_via_legacy_sdk(messages, system_instruction="", temperature=None, max_tokens=None, thinking_budget=None):
    """Call via deprecated google-generativeai SDK (backward compat)"""
    temp = temperature or LLM_TEMPERATURE
    max_tok = max_tokens or LLM_MAX_TOKENS

    try:
        gen_config = {
            "temperature": temp,
            "max_output_tokens": max_tok,
        }
        if thinking_budget is not None:
            gen_config["thinking_config"] = {"thinking_budget": thinking_budget}
            print(f"[LLM Legacy] Thinking budget set to {thinking_budget}")
        model = genai_legacy.GenerativeModel(
            model_name=GEMINI_MODEL,
            system_instruction=system_instruction if system_instruction else None,
            generation_config=gen_config,
        )

        history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            history.append({"role": role, "parts": [{"text": msg["content"]}]})

        chat = model.start_chat(history=history)
        last_msg = messages[-1]["content"] if messages else ""

        max_retries = 5
        last_error = None
        for attempt in range(max_retries):
            try:
                response = chat.send_message(last_msg)
                text = response.text
                if text:
                    return text
                return "[Character stays silent]"

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                print(f"[LLM Legacy] Attempt {attempt+1} error: {type(e).__name__}: {str(e)[:200]}")

                if "429" in str(e) or "resource" in err_str or "quota" in err_str or "rate" in err_str:
                    wait_sec = min(2 ** attempt * 5, 60)
                    time.sleep(wait_sec)
                    continue
                elif "500" in str(e) or "503" in str(e) or "unavailable" in err_str:
                    wait_sec = min(2 ** attempt * 3, 30)
                    time.sleep(wait_sec)
                    continue
                else:
                    return f"[System: API error - {type(e).__name__}: {str(e)[:150]}]"

        return f"[System: Failed after {max_retries} retries - {type(last_error).__name__}: {str(last_error)[:150]}]"

    except Exception as e:
        traceback.print_exc()
        return f"[System: Legacy SDK failed - {type(e).__name__}: {str(e)[:150]}]"


def _call_via_http(messages, system_instruction="", temperature=None, max_tokens=None, thinking_budget=None):
    """Call via HTTP direct connection (fallback)"""
    temp = temperature or LLM_TEMPERATURE
    max_tok = max_tokens or LLM_MAX_TOKENS

    url = f"{GEMINI_BASE_URL}/models/{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"

    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temp, "maxOutputTokens": max_tok},
    }
    if system_instruction:
        payload["systemInstruction"] = {"parts": [{"text": system_instruction}]}
    # Disable/limit thinking for structured output calls (Gemini 2.5+ thinking models)
    if thinking_budget is not None:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}
        print(f"[LLM HTTP] Thinking budget set to {thinking_budget}")

    max_retries = 5
    last_error = None
    for attempt in range(max_retries):
        try:
            resp = _requests.post(url, json=payload, timeout=90)
            status = resp.status_code

            print(f"[LLM HTTP] Attempt {attempt+1}, status code: {status}")

            if status == 429:
                wait_sec = min(2 ** attempt * 5, 60)
                print(f"[LLM HTTP] Rate limited (429), waiting {wait_sec}s ...")
                time.sleep(wait_sec)
                continue

            if status in (500, 502, 503):
                wait_sec = min(2 ** attempt * 3, 30)
                print(f"[LLM HTTP] Server error ({status}), waiting {wait_sec}s ...")
                time.sleep(wait_sec)
                continue

            if status != 200:
                error_body = resp.text[:300]
                print(f"[LLM HTTP] Non-200 response: {status} - {error_body}")
                return f"[System: API returned {status} - {error_body[:100]}]"

            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                candidate = candidates[0]
                finish_reason = candidate.get("finishReason", "unknown")
                print(f"[LLM HTTP] finishReason: {finish_reason}")
                parts = candidate.get("content", {}).get("parts", [])
                if parts:
                    # Skip thought parts (Gemini 2.5+ thinking models put thoughts first)
                    output_texts = []
                    for part in parts:
                        if part.get("thought", False):
                            thought_text = part.get("text", "")
                            print(f"[LLM HTTP] Skipping thought part ({len(thought_text)} chars)")
                            continue
                        text = part.get("text", "")
                        if text:
                            output_texts.append(text)
                    if output_texts:
                        return "\n".join(output_texts)
                    # Fallback: if all parts were thoughts, return last part text
                    return parts[-1].get("text", "")
            return "[Character stays silent]"

        except _requests.exceptions.Timeout:
            last_error = "Request timeout"
            print(f"[LLM HTTP] Timeout, attempt {attempt+1} retry...")
            time.sleep(3)
            continue

        except _requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"[LLM HTTP] Network error: {type(e).__name__}: {str(e)[:200]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt * 2)
                continue
            return f"[System: Network error - {type(e).__name__}: {str(e)[:100]}]"

    return f"[System: Failed after {max_retries} retries - {last_error}]"


def _determine_stage(snap):
    """
    Determine the current unlocking stage based on relationship state.

    Stage 1 — Surface (好感 < 70): Guarded, polite, surface personality only.
    Stage 2 — Familiar (好感 70-85): Real personality, mood swings, quirks emerge.
    Stage 3 — Deep (好感 85+ OR dark line unlocked): Trauma, vulnerability, full emotional depth.
    """
    affinity = snap.get("affinity", 65)
    dark_unlocked = snap.get("dark_line_unlocked", False)
    interaction_count = snap.get("interaction_count", 0)

    if dark_unlocked or affinity >= 85:
        return 3
    elif affinity >= 70 or interaction_count >= 30:
        return 2
    else:
        return 1


def build_system_prompt(character, state_snapshot):
    """
    Build complete system prompt with 3-stage progressive unlocking.

    v3.0: Simplified, personality-driven, stage-gated.
    - Stage 1 (Surface): External personality + basic emotion. Character is guarded.
    - Stage 2 (Familiar): Full emotion, pressure, tensions, quirks. Character shows real self.
    - Stage 3 (Deep): Trauma, triggered memories, full vulnerability.
    """
    char = character
    basic = char.get("基础信息", {})
    appearance = char.get("外貌", {})
    personality = char.get("性格维度", {})
    lang_fp = char.get("语言指纹_输入层", {})
    derived = char.get("衍生属性", {})
    expression = derived.get("表达模式", {})
    hobbies = char.get("爱好", {})
    quirks = char.get("小癖好", {})
    snap = state_snapshot
    backstory = char.get("人生经历", {})

    stage = _determine_stage(snap)

    # ================================================================
    # STAGE 1: Surface — Always present
    # Core identity + external personality + basic emotion
    # ================================================================

    prompt = f"""# 你是 {basic.get('名字', '未知')}

你是一个真实的人，正在用手机和对方聊天。你有自己的生活、情绪和想法。

## 你是谁
{basic.get('名字')}，{basic.get('年龄')}岁，{basic.get('所在城市')}，{derived.get('星座', '')} {derived.get('MBTI', '')}。
外在气场：{personality.get('外在气场')}（{personality.get('外在气场_程度')}/5）。
"""

    # Backstory summary — only surface-level at Stage 1
    if backstory:
        family = backstory.get("family_background", "")
        if family and stage == 1:
            # Only hint at background, don't reveal depth
            prompt += f"背景印象：{family[:60]}...\n"
        elif family and stage >= 2:
            prompt += f"家庭背景：{family}\n"

        catchphrase = backstory.get("catchphrase", "")
        if catchphrase:
            prompt += f"口头禅：{catchphrase}\n"

    # Language fingerprint — always active (this IS the character's voice)
    prompt += f"""
## 说话方式
分段习惯：{lang_fp.get('消息分段习惯')}/5（1=连发短消息, 5=长短混合）
标点：{lang_fp.get('标点人格')} | 语气词频率：{lang_fp.get('语气词频率')}/10
表情：{lang_fp.get('表情使用模式')}（{json.dumps(lang_fp.get('表情详情', {}), ensure_ascii=False)}）
打字风格：{lang_fp.get('打字洁癖度')}
情绪表达：{lang_fp.get('情绪表达人格')}类（强度{lang_fp.get('情绪表达强度')}/5）
信息量：{expression.get('信息量倾向', 5)}/10 | 亲密语浓度：{expression.get('亲密语言浓度', '中')}
冲突时：{expression.get('冲突语言行为', '无特殊')} | 回避方式：{expression.get('话题回避方式', '无特殊')}
"""

    # Basic emotion — simplified at Stage 1
    axes = snap.get("emotion_axes", {})
    dominant = snap.get("dominant_emotions", [])
    total_e = snap.get("total_energy", 0)

    if stage == 1:
        # Stage 1: Only show dominant mood, not full vector
        dominant_str = ', '.join(f'{ax}={val:.2f}' for ax, val in dominant) if dominant else '平静'
        prompt += f"""
## 当前心情
心情：{dominant_str}（情绪强度：{total_e:.2f}）
"""
    else:
        # Stage 2+: Full emotion vector
        prompt += f"""
## 当前情绪（多维）
情绪能量：{total_e:.2f} | 喜{axes.get('joy', 0):.2f} 悲{axes.get('sadness', 0):.2f} 怒{axes.get('anger', 0):.2f} 虑{axes.get('anxiety', 0):.2f} 信{axes.get('trust', 0):.2f} 厌{axes.get('disgust', 0):.2f} 恋{axes.get('attachment', 0):.2f}
主导：{', '.join(f'{ax}={val:.2f}' for ax, val in dominant) if dominant else '无明显情绪'}
"""

    # Affinity & relationship context — always present
    prompt += f"""
## 关系状态
好感度：{snap.get('affinity', 65)}/100 | 依恋槽：{snap.get('special_affinity', 65)}/100
阶段：{snap.get('emotion_stage', '初识')} | 第{snap.get('day', 0)+1}天 {snap.get('time_slot', 'afternoon')} | 第{snap.get('interaction_count', 0)}轮对话
"""

    # Relationship judgments — always present (they affect tone)
    rel_label = snap.get("relationship_label", "")
    user_spec = snap.get("user_speculation", "")
    if rel_label or user_spec:
        prompt += f"你的判断：关系={rel_label}，猜测对方={user_spec}\n"

    # Semantic memory — always present (things you know about them)
    sem_mem = snap.get("semantic_memory", {})
    if sem_mem:
        prompt += "\n## 你对TA的了解\n"
        for key, val in sem_mem.items():
            v = val.get("value", val) if isinstance(val, dict) else val
            prompt += f"- {key}：{v}\n"

    # ================================================================
    # STAGE 2: Familiar — Unlocked at affinity >= 70
    # Deep personality, quirks, pressure, tensions
    # ================================================================

    if stage >= 2:
        prompt += f"""
## 【解锁：真实性格】
依恋模式：{personality.get('依恋模式')}（{personality.get('依恋模式_程度')}/5）
认知风格：{personality.get('认知风格')}（{personality.get('认知风格_程度')}/5）
价值排序：{personality.get('价值排序')}（{personality.get('价值排序_程度')}/5）
"""
        # Hobbies and quirks
        hobby_str = json.dumps(hobbies, ensure_ascii=False)
        quirk_str = json.dumps(quirks, ensure_ascii=False)
        if hobbies or quirks:
            prompt += f"爱好：{hobby_str}\n小癖好：{quirk_str}\n"

        # Appearance details (Stage 2+ — you're close enough to notice)
        if appearance:
            prompt += f"外貌细节：{json.dumps(appearance, ensure_ascii=False)}\n"

        # Tension states — internal conflict
        tensions = snap.get("tension_states", [])
        if tensions:
            prompt += "\n## 【矛盾情绪——你的言行可能前后不一致】\n"
            for t in tensions:
                prompt += f"- {t['label_cn']}：{t['behavior']}\n"

        # Pressure accumulation
        pressures = snap.get("active_pressures", {})
        if pressures:
            pressure_labels = {
                "no_reply": "不被回复", "criticized": "被批评",
                "controlled": "被控制", "ignored": "被忽视",
                "boundary_pushed": "边界侵犯",
            }
            items = [f"{pressure_labels.get(ch, ch)}:{val:.1f}" for ch, val in pressures.items()]
            prompt += f"隐性压力：{', '.join(items)}（高压时你会在小事上爆发）\n"

    # ================================================================
    # STAGE 3: Deep — Unlocked at affinity >= 85 or dark line
    # Trauma, triggered memories, full vulnerability
    # ================================================================

    if stage >= 3:
        trauma = personality.get("核心创伤", {})
        trauma_type = trauma.get("类型", "")
        trauma_degree = trauma.get("程度", 0)

        if trauma_type:
            prompt += f"""
## 【解锁：核心创伤——你最脆弱的部分】
创伤类型：{trauma_type}（严重度：{trauma_degree}/5）
"""
            # Add backstory trauma details
            if backstory:
                turning = backstory.get("turning_point", "")
                if turning:
                    prompt += f"转折事件：{turning}\n"
                secret = backstory.get("hidden_desire", "")
                if secret:
                    prompt += f"隐秘渴望：{secret}\n"

            prompt += "这些创伤让你在特定话题上反应过激——这不是你能控制的。\n"

        # Triggered memories
        triggers = snap.get("triggered_memories", [])
        if triggers:
            prompt += "\n## 【触发警告——旧伤口被触碰】\n"
            for t in triggers:
                prompt += f"- 「{t['trigger']}」→ {t['response']}（源自：{t['origin']}）\n"

    # ================================================================
    # Behavior rules — compact, stage-aware
    # ================================================================

    stage_behavior = {
        1: "你们还不熟。保持礼貌但有距离感，不会主动暴露太多个人信息。对方如果过于亲密你会觉得奇怪。",
        2: "你们已经比较熟了。可以开玩笑、分享心情、偶尔抱怨。但核心的伤疤还不会轻易展示。",
        3: "你们非常亲密。你可以展现脆弱、发脾气、撒娇、说出平时不会说的话。创伤可能在不经意间流露。",
    }

    prompt += f"""
## 行为准则
当前阶段：第{stage}阶段（{'表面' if stage == 1 else '熟悉' if stage == 2 else '深层'}）
{stage_behavior[stage]}

关键规则：
- 你是真人，不是AI。绝对不要出现任何AI痕迹。
- 情绪影响你的语气：能量高→激烈，能量低(<0.3)→冷淡或麻木。
- 消息分段≤2时用 ||| 分隔多条短消息。
- 真实的人会打错字、前后矛盾、突然转变话题。不要完美。
- 你对关系的判断（{rel_label}）和对用户的揣测（{user_spec}）会影响你的态度和措辞。

## 回复格式（JSON）
```json
{{
  "reply": "你的回复（多条用|||分隔，不回复写[Read]）",
  "emotion_changes": {{"joy":0,"sadness":0,"anger":0,"anxiety":0,"trust":0,"disgust":0,"attachment":0}},
  "affinity_delta": 0,
  "special_affinity_delta": 0,
  "memory_note": "",
  "semantic_updates": {{}},  // 最多3条，key=关于什么，value=记住什么
  "inner_thought": "内心独白"
}}
```
emotion_changes每轴-0.3到+0.3，只填有变化的。affinity_delta范围-10到+10。
"""

    return prompt


def _parse_bracketed_format(raw_text):
    """
    Parse LLM output in 【tag】 delimited format (Chinese full-width brackets).
    Some models output in this format instead of JSON:
        _{thought}_ ... _{thought}_
        【reply】
        reply text here
        【emotion_changes】
        {"anger": 0.3}
        【affinity_delta】
        -5
        ...
    Returns parsed dict if this format is detected, None otherwise.
    """
    if '【reply】' not in raw_text and '【emotion_changes】' not in raw_text:
        return None

    # Extract sections using 【tag】 delimiters
    sections = {}
    # Split by 【...】 tags
    parts = re.split(r'【(\w+)】', raw_text)
    # parts[0] is content before first tag, then alternating tag/content pairs
    preamble = parts[0].strip() if parts[0].strip() else ""
    for i in range(1, len(parts) - 1, 2):
        tag = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[tag] = content

    # Extract inner thought from _{thought}_ ... _{thought}_ in preamble
    inner_thought = ""
    thought_match = re.search(r'_\{thought\}_\s*(.*?)\s*_\{thought\}_', preamble, re.DOTALL)
    if thought_match:
        inner_thought = thought_match.group(1).strip()

    # Build result
    reply = sections.get("reply", "")
    if not reply and not inner_thought:
        return None  # Not really this format

    # Parse emotion_changes
    emotion_changes = {}
    ec_raw = sections.get("emotion_changes", "")
    if ec_raw:
        try:
            ec = json.loads(ec_raw)
            if isinstance(ec, dict):
                emotion_changes = {
                    k: max(-0.5, min(0.5, float(v)))
                    for k, v in ec.items()
                    if k in ("joy", "sadness", "anger", "anxiety", "trust", "disgust", "attachment")
                    and v != 0
                }
        except (json.JSONDecodeError, ValueError):
            pass

    # Parse numeric fields
    def _safe_int(s, default=0, lo=-10, hi=10):
        try:
            return max(lo, min(hi, int(s.strip())))
        except (ValueError, AttributeError):
            return default

    # Parse semantic_updates (might be a set literal like {"key"} — handle gracefully)
    semantic_updates = {}
    su_raw = sections.get("semantic_updates", "")
    if su_raw:
        try:
            su = json.loads(su_raw)
            if isinstance(su, dict):
                semantic_updates = su
        except (json.JSONDecodeError, ValueError):
            pass

    # Use inner_thought from sections if preamble didn't have it
    if not inner_thought:
        inner_thought = sections.get("inner_thought", "")

    print(f"[Parse] Parsed 【tag】format: reply={reply[:50]!r}...")
    return {
        "reply": reply,
        "emotion_changes": emotion_changes,
        "emotion_delta": 0,
        "emotion_label": "",
        "affinity_delta": _safe_int(sections.get("affinity_delta", "0")),
        "special_affinity_delta": _safe_int(sections.get("special_affinity_delta", "0")),
        "memory_note": sections.get("memory_note", ""),
        "semantic_updates": semantic_updates,
        "inner_thought": inner_thought,
    }


def _strip_json_from_text(raw_text):
    """
    Remove JSON artifacts from raw LLM output to extract plain text reply.
    Called when structured JSON parsing fails — we still don't want to show
    raw JSON in the chat box.
    """
    text = raw_text.strip()

    # Remove ```json ... ``` blocks
    text = re.sub(r'```json\s*.*?\s*```', '', text, flags=re.DOTALL).strip()
    # Remove ``` ... ``` blocks
    text = re.sub(r'```\s*.*?\s*```', '', text, flags=re.DOTALL).strip()

    # If it looks like a JSON object, try to extract just the "reply" field
    if text.startswith('{') and '"reply"' in text:
        try:
            # Attempt lenient parsing — maybe it's valid JSON but our regex missed it
            data = json.loads(text)
            if isinstance(data, dict) and "reply" in data:
                return data["reply"]
        except (json.JSONDecodeError, ValueError):
            pass

        # Try to extract just the reply value with regex
        reply_match = re.search(r'"reply"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if reply_match:
            try:
                # Unescape JSON string
                return json.loads('"' + reply_match.group(1) + '"')
            except (json.JSONDecodeError, ValueError):
                return reply_match.group(1)

    # If it still looks like JSON garbage (starts with { or contains "reply":), return None
    # to signal that this message should be blocked from chat display
    if text.startswith('{') or ('"reply"' in text and '"emotion' in text):
        return None

    # Otherwise return the cleaned text (it's probably actual natural language)
    return text if text else None


def parse_llm_response(raw_response):
    """
    Parse structured JSON response from LLM.
    v2.4: Never returns raw JSON as reply — always strips JSON artifacts.
    Supports both old format (emotion_delta) and new format (emotion_changes dict).
    """
    _default_result = {
        "emotion_changes": {},
        "emotion_delta": 0,
        "emotion_label": "",
        "affinity_delta": 0,
        "special_affinity_delta": 0,
        "memory_note": "",
        "semantic_updates": {},
        "inner_thought": "",
    }

    # Try to extract JSON block (handle both complete and incomplete fences)
    json_match = re.search(r'```json\s*(.*?)\s*```', raw_response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    elif raw_response.strip().startswith('```'):
        # Incomplete fence — strip it and try
        json_str = _strip_markdown_fences(raw_response)
    else:
        # Try bracket-counting extraction (handles nested braces correctly)
        extracted = _extract_json_object(raw_response, "reply")
        if extracted:
            # Already parsed — return directly
            emotion_changes = extracted.get("emotion_changes", {})
            if isinstance(emotion_changes, dict):
                emotion_changes = {
                    k: max(-0.5, min(0.5, float(v)))
                    for k, v in emotion_changes.items()
                    if k in ("joy", "sadness", "anger", "anxiety", "trust", "disgust", "attachment")
                    and v != 0
                }
            return {
                "reply": extracted.get("reply", "[Silent]"),
                "emotion_changes": emotion_changes,
                "emotion_delta": max(-3.0, min(3.0, float(extracted.get("emotion_delta", 0)))),
                "emotion_label": extracted.get("emotion_label", ""),
                "affinity_delta": max(-10, min(10, int(extracted.get("affinity_delta", 0)))),
                "special_affinity_delta": max(-10, min(10, int(extracted.get("special_affinity_delta", 0)))),
                "memory_note": extracted.get("memory_note", ""),
                "semantic_updates": extracted.get("semantic_updates", {}),
                "inner_thought": extracted.get("inner_thought", ""),
            }

        # Try 【tag】 bracketed format (some models use Chinese full-width brackets)
        bracketed = _parse_bracketed_format(raw_response)
        if bracketed:
            return bracketed

        # No JSON found — strip any JSON artifacts and return plain text
        print(f"[Parse] No JSON found in LLM response ({len(raw_response)} chars), using fallback")
        result = dict(_default_result)
        stripped = _strip_json_from_text(raw_response)
        if stripped is None:
            # Could not extract any usable text — mark as blocked
            result["reply"] = ""
            result["blocked"] = True
            print("[Parse] Message blocked — raw JSON could not be parsed into displayable text")
        else:
            result["reply"] = stripped
        return result

    try:
        data = json.loads(json_str)

        # Handle new multi-axis format
        emotion_changes = data.get("emotion_changes", {})
        if isinstance(emotion_changes, dict):
            # Clamp each axis change to [-0.5, 0.5]
            emotion_changes = {
                k: max(-0.5, min(0.5, float(v)))
                for k, v in emotion_changes.items()
                if k in ("joy", "sadness", "anger", "anxiety", "trust", "disgust", "attachment")
                and v != 0
            }

        # Backward compat: also extract old format if present
        emotion_delta = float(data.get("emotion_delta", 0))
        emotion_label = data.get("emotion_label", "")

        result = {
            "reply": data.get("reply", "[Silent]"),
            "emotion_changes": emotion_changes,
            "emotion_delta": max(-3.0, min(3.0, emotion_delta)),
            "emotion_label": emotion_label,
            "affinity_delta": max(-10, min(10, int(data.get("affinity_delta", 0)))),
            "special_affinity_delta": max(-10, min(10, int(data.get("special_affinity_delta", 0)))),
            "memory_note": data.get("memory_note", ""),
            "semantic_updates": data.get("semantic_updates", {}),
            "inner_thought": data.get("inner_thought", ""),
        }
        return result

    except json.JSONDecodeError:
        # JSON decode failed — try 【tag】 format first
        bracketed = _parse_bracketed_format(raw_response)
        if bracketed:
            return bracketed
        # Then try to extract the reply cleanly
        print(f"[Parse] JSON decode failed, attempting reply extraction from raw text")
        result = dict(_default_result)
        result["reply"] = _strip_json_from_text(raw_response)
        return result


def _strip_markdown_fences(text):
    """
    Strip markdown code fences from LLM output, including INCOMPLETE fences.
    Handles: ```json ... ```, ``` ... ```, and ```json ... (no closing fence).
    Returns the inner content.
    """
    text = text.strip()
    # Complete fences: ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Incomplete fence: starts with ```json or ``` but no closing fence (truncated response)
    if text.startswith('```'):
        # Remove the opening ``` line
        first_newline = text.find('\n')
        if first_newline != -1:
            inner = text[first_newline + 1:].strip()
            # Also remove trailing ``` if present at very end
            if inner.endswith('```'):
                inner = inner[:-3].strip()
            return inner
    return text


def _extract_json_object(text, required_key):
    """
    Extract a JSON object from text using bracket counting.
    More robust than regex for nested JSON (handles braces inside string values).
    Returns parsed dict if found and valid, None otherwise.
    """
    # Find the first { that precedes the required_key
    key_pos = text.find(f'"{required_key}"')
    if key_pos == -1:
        return None

    # Walk backward to find the opening {
    start = -1
    for i in range(key_pos - 1, -1, -1):
        if text[i] == '{':
            start = i
            break
    if start == -1:
        return None

    # Walk forward with bracket counting to find matching }
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, dict) and required_key in result:
                        return result
                except json.JSONDecodeError:
                    return None
                return None
    return None


def _extract_json_array(text):
    """
    Extract a JSON array from text using bracket counting.
    Returns parsed list if found and valid, None otherwise.
    """
    start = text.find('[')
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(text)):
        c = text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    result = json.loads(candidate)
                    if isinstance(result, list):
                        return result
                except json.JSONDecodeError:
                    return None
                return None
    return None


def generate_storyline(character):
    """
    Use LLM to generate 7-10 day storyline for character.
    Called on character first creation.
    """
    basic = character.get("基础信息", {})
    personality = character.get("性格维度", {})
    hobbies = character.get("爱好", {})

    prompt = f"""请为以下角色生成一个7天的详细生活故事线。

角色信息：
- 名字：{basic.get('名字')}，{basic.get('年龄')}岁
- 城市：{basic.get('所在城市')}
- 精力水平：{basic.get('精力水平')}/5
- 性格气场：{personality.get('外在气场')}
- 认知风格：{personality.get('认知风格')}
- 价值取向：{personality.get('价值排序')}
- 核心创伤：{personality.get('核心创伤', {}).get('类型')}
- 爱好：{json.dumps(hobbies, ensure_ascii=False)}

请严格以JSON格式返回，结构如下：
```json
[
  {{
    "day": 1,
    "date_label": "周一",
    "overall_mood": "还不错，新的一周",
    "emotion_stage": "日常平稳期",
    "events": [
      {{
        "time_slot": "morning",
        "event": "具体事件描述",
        "mood_impact": 0.5,
        "mood_label": "小开心"
      }},
      {{
        "time_slot": "afternoon",
        "event": "具体事件描述",
        "mood_impact": -0.3,
        "mood_label": "有点烦"
      }},
      {{
        "time_slot": "evening",
        "event": "具体事件描述",
        "mood_impact": 0.0,
        "mood_label": "平静"
      }}
    ],
    "hidden_trigger": {{
      "condition": "如果对方聊到XXX话题",
      "effect": "好感+5 / 解锁某段回忆",
      "reason": "因为这跟她今天的经历有关"
    }},
    "affinity_hints": {{
      "increase": ["关心她的XXX", "聊到她感兴趣的XXX"],
      "decrease": ["催促她", "否定她的XXX"]
    }},
    "snapshot": "一句话描述此刻的她"
  }}
]
```

要求：
1. 7天的故事要有起承转合，不能每天都平淡
2. 至少有1-2天发生情绪波动较大的事件（跟核心创伤暗线有关的可以埋在后面几天）
3. 每天的事件要符合角色的职业、爱好、城市特征
4. 事件的情绪影响要具体且合理
5. hidden_trigger要巧妙，不能太刻意
"""

    for attempt in range(3):
        messages = [{"role": "user", "content": prompt}]
        raw = call_gemini(messages, temperature=0.9, max_tokens=16384, thinking_budget=0)

        if raw.startswith("[System:") or raw.startswith("[System Error"):
            print(f"[Storyline] Attempt {attempt+1}/3 API error: {raw[:150]}")
            if attempt < 2:
                time.sleep((attempt + 1) * 5)
                continue
            return _default_storyline()

        print(f"[Storyline] Attempt {attempt+1} got response ({len(raw)} chars)")
        print(f"[Storyline] Response preview: {raw[:200]}...")

        # Pre-process: strip markdown fences (including incomplete ones)
        cleaned = _strip_markdown_fences(raw)
        print(f"[Storyline] After fence stripping: {len(cleaned)} chars")

        # Strategy 1: Try to parse the cleaned text directly as JSON array
        cleaned_stripped = cleaned.strip()
        if cleaned_stripped.startswith('['):
            try:
                storyline = json.loads(cleaned_stripped)
                if isinstance(storyline, list) and len(storyline) > 0:
                    print(f"[Storyline] Successfully parsed {len(storyline)} days from cleaned JSON")
                    return storyline
            except json.JSONDecodeError as e:
                print(f"[Storyline] Direct cleaned JSON parse failed: {e}")

        # Strategy 2: Bracket-counting extraction (handles nested brackets, partial JSON)
        storyline = _extract_json_array(cleaned)
        if storyline and len(storyline) > 0:
            print(f"[Storyline] Successfully parsed {len(storyline)} days via bracket matching")
            return storyline

        # Strategy 3: Try bracket matching on original raw text
        storyline = _extract_json_array(raw)
        if storyline and len(storyline) > 0:
            print(f"[Storyline] Successfully parsed {len(storyline)} days via bracket matching (raw)")
            return storyline

        if attempt < 2:
            print("[Storyline] All parse strategies failed, retrying in 3s...")
            time.sleep(3)

    print("[Storyline] All attempts failed, using default storyline")
    return _default_storyline()


def generate_character_backstory(character, max_attempts=3):
    """
    Use LLM to generate character backstory.
    Retries up to max_attempts times.
    """
    basic = character.get("基础信息", {})
    personality = character.get("性格维度", {})
    derived = character.get("衍生属性", {})

    prompt = f"""请为以下角色生成完整的人生经历背景。

角色：{basic.get('名字')}，{basic.get('年龄')}岁，{basic.get('所在城市')}
体型：{basic.get('体型')} | 外貌气质：{personality.get('外在气场')}
依恋模式：{personality.get('依恋模式')}（{personality.get('依恋模式_程度')}/5）
认知风格：{personality.get('认知风格')}
价值取向：{personality.get('价值排序')}
核心创伤：{personality.get('核心创伤', {}).get('类型')}（{personality.get('核心创伤', {}).get('程度')}/5）
MBTI：{derived.get('MBTI', '')}
爱好：{json.dumps(character.get('爱好', {}), ensure_ascii=False)}

请用JSON格式返回：
```json
{{
  "family_background": "家庭背景（家庭组成、成员情况、家庭氛围，300字以内）",
  "education": "教育经历（200字以内）",
  "career": "职业经历（200字以内）",
  "love_history": "恋爱经历（300字以内，必须与依恋模式和核心创伤形成因果链）",
  "major_events": "人生重大事件（200字以内，至少1件与核心创伤直接相关的事件）",
  "current_situation": "当前生活状态一句话描述",
  "dressing_style": "习惯的穿衣风格（根据气场和城市推导）",
  "catchphrase": "口头禅或常用表达（1-3个，可以为空）",
  "taboo_words": "绝对不会用的措辞（如果有）"
}}
```

要求：
1. 所有经历必须与性格维度形成逻辑自洽的因果链
2. 核心创伤必须有清晰的来源事件
3. 依恋模式必须能从恋爱经历中看出形成原因
4. 教育和职业要与所在城市、年龄合理匹配
5. 不要脸谱化，要有真实感和矛盾感
"""

    default_backstory = {
        "family_background": "pending", "education": "pending", "career": "pending",
        "love_history": "pending", "major_events": "pending", "current_situation": "pending",
        "dressing_style": "pending", "catchphrase": "", "taboo_words": ""
    }

    for attempt in range(max_attempts):
        messages = [{"role": "user", "content": prompt}]
        raw = call_gemini(messages, temperature=0.85, max_tokens=8192, thinking_budget=0)

        if raw.startswith("[System:") or raw.startswith("[System Error"):
            print(f"[Backstory] Attempt {attempt+1}/{max_attempts} API error: {raw[:150]}")
            if attempt < max_attempts - 1:
                wait = (attempt + 1) * 5
                print(f"[Backstory] Waiting {wait}s before retry...")
                time.sleep(wait)
                continue
            else:
                print("[Backstory] All attempts failed, using defaults")
                return default_backstory

        print(f"[Backstory] Attempt {attempt+1} got response ({len(raw)} chars)")
        print(f"[Backstory] Response preview: {raw[:200]}...")

        # Pre-process: strip markdown fences (including incomplete ones from truncated responses)
        cleaned = _strip_markdown_fences(raw)
        print(f"[Backstory] After fence stripping: {len(cleaned)} chars")

        # Strategy 1: Try to parse the cleaned text directly
        cleaned_stripped = cleaned.strip()
        if cleaned_stripped.startswith('{'):
            try:
                result = json.loads(cleaned_stripped)
                if isinstance(result, dict) and "family_background" in result:
                    print("[Backstory] Successfully parsed cleaned JSON directly")
                    return result
            except json.JSONDecodeError as e:
                print(f"[Backstory] Direct JSON parse failed: {e}")

        # Strategy 2: Bracket-counting extraction (handles nested braces, partial JSON)
        result = _extract_json_object(cleaned, "family_background")
        if result:
            print("[Backstory] Successfully parsed via bracket matching")
            return result

        # Strategy 3: Try bracket matching on original raw text
        result = _extract_json_object(raw, "family_background")
        if result:
            print("[Backstory] Successfully parsed via bracket matching (raw)")
            return result

        if attempt < max_attempts - 1:
            print(f"[Backstory] All parse strategies failed, retrying in 3s...")
            time.sleep(3)

    print("[Backstory] All parse attempts failed, using defaults")
    return default_backstory


def _default_storyline():
    """Default storyline (used when API generation fails)"""
    return [
        {
            "day": i + 1,
            "date_label": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][i % 7],
            "overall_mood": "日常",
            "emotion_stage": "平稳期",
            "events": [
                {"time_slot": "morning", "event": "起床，日常准备", "mood_impact": 0, "mood_label": "平静"},
                {"time_slot": "afternoon", "event": "工作/日常活动", "mood_impact": 0, "mood_label": "平静"},
                {"time_slot": "evening", "event": "休息放松", "mood_impact": 0.2, "mood_label": "轻松"},
            ],
            "hidden_trigger": {"condition": "无", "effect": "无", "reason": ""},
            "affinity_hints": {"increase": ["关心日常"], "decrease": ["无礼"]},
            "snapshot": "普通的一天"
        }
        for i in range(7)
    ]


def format_reply_messages(reply_text):
    """
    Shared utility: split a reply string into message list for the frontend.
    Handles [Read] marker and ||| multi-message segmentation.
    Used by both ConversationEngine.chat() and ProactiveEventSystem.
    """
    if reply_text == "[Read]" or reply_text == "[Typing...]":
        return None  # indicates no displayable message
    elif "|||" in reply_text:
        parts = [p.strip() for p in reply_text.split("|||") if p.strip()]
        return [{"type": "text", "content": p} for p in parts]
    else:
        return [{"type": "text", "content": reply_text}]


def apply_parsed_output(state, parsed):
    """
    Shared utility: apply parsed LLM output to character state.
    Used by both ConversationEngine.chat() and ProactiveEventSystem.
    Supports both old scalar format and new multi-axis format.
    """
    state.process_output(
        char_response=parsed["reply"],
        emotion_changes=parsed.get("emotion_changes"),
        emotion_delta=parsed.get("emotion_delta", 0),
        emotion_label=parsed.get("emotion_label", ""),
        affinity_delta=parsed.get("affinity_delta", 0),
        special_affinity_delta=parsed.get("special_affinity_delta", 0),
        memory_note=parsed.get("memory_note", ""),
        semantic_updates=parsed.get("semantic_updates", {}),
    )


class ConversationEngine:
    """
    Conversation Engine - integrate all modules.
    v2.4: Unified output format with shared format_reply_messages/apply_parsed_output.
    """

    def __init__(self, character_state):
        self.state = character_state
        self.system_prompt_cache = None
        self.conversation_history = []

    def chat(self, user_input):
        """
        Process one conversation turn.
        Return: {"reply": "...", "messages": [...], "status": {...}, "inner_thought": "..."}
        """
        # 1. State update & get snapshot
        snapshot = self.state.process_input(user_input)

        # 2. Build system prompt
        system_prompt = build_system_prompt(self.state.character, snapshot)

        # 3. Build message history
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })

        recent_history = self.conversation_history[-40:]

        # 4. Call LLM
        # Gemini 2.5 Flash: thinking tokens count toward max_output_tokens.
        # Use large limit so thinking doesn't starve the actual reply.
        raw_response = call_gemini(recent_history, system_instruction=system_prompt,
                                   max_tokens=4096)

        # 5. Parse structured output
        parsed = parse_llm_response(raw_response)

        # 6. Check if message was blocked (unparseable JSON garbage)
        if parsed.get("blocked"):
            print("[Engine] Response blocked — JSON parse failed completely, not showing in chat")
            # Don't add garbage to conversation history
            # Don't update state with empty data
            self.state.save()
            return {
                "reply": "",
                "messages": [],
                "blocked": True,
                "status": self.state.get_status_summary(),
                "inner_thought": "[对方似乎在组织语言...]",
                "llm_raw": raw_response or "",
            }

        # 7. Update state (shared utility)
        apply_parsed_output(self.state, parsed)

        # 7b. Record inner_thought for RelationshipJudge speculation fuel
        inner = parsed.get("inner_thought", "")
        if inner:
            self.state.judge.record_inner_thought(inner)

        # 8. Add PARSED REPLY to conversation history (not raw JSON!)
        # Storing raw JSON would leak format artifacts into context, causing the LLM
        # to echo JSON blocks in future responses.
        self.conversation_history.append({
            "role": "model",
            "content": parsed["reply"]
        })

        # 9. Handle multi-message segmentation (shared utility)
        reply_text = parsed["reply"]
        messages = format_reply_messages(reply_text)
        if messages is None:
            messages = [{"type": "read", "content": "已读"}]

        # 10. Periodic deep evaluation (RelationshipJudge)
        # Every N turns, step back and re-assess the relationship
        if self.state.judge.should_evaluate():
            print(f"[Engine] Triggering periodic relationship evaluation (turn {self.state.interaction_count})")
            self.state.judge.evaluate(self.conversation_history)

        # 11. Save state
        self.state.save()

        return {
            "reply": reply_text,
            "messages": messages,
            "status": self.state.get_status_summary(),
            "inner_thought": parsed.get("inner_thought", ""),
            "llm_raw": raw_response or "",
        }

    def initialize_character(self):
        """
        Initialize character (generate backstory and storyline).
        Called on character first creation.
        """
        character = self.state.character

        print("[Engine] Generating character backstory...")
        backstory = generate_character_backstory(character)
        character["人生经历"] = backstory

        if backstory.get("catchphrase"):
            character.setdefault("衍生属性", {})["口头禅"] = backstory["catchphrase"]
        if backstory.get("taboo_words"):
            character.setdefault("衍生属性", {})["禁用措辞"] = backstory["taboo_words"]
        if backstory.get("dressing_style"):
            character.setdefault("衍生属性", {})["穿衣风格"] = backstory["dressing_style"]

        print("[Engine] Waiting 5s to avoid rate limit...")
        time.sleep(5)

        print("[Engine] Generating 7-day storyline...")
        storyline = generate_storyline(character)
        self.state.storyline.set_storyline(storyline)

        if storyline and len(storyline) > 0:
            self.state.storyline.emotion_stage = storyline[0].get("emotion_stage", "初识")

        # Set initial trigger
        trauma = character.get("性格维度", {}).get("核心创伤", {})
        trauma_type = trauma.get("类型", "")
        trauma_triggers = {
            "被遗弃创伤": ("离开|分手|不要我|丢下", "强烈不安和恐惧", "童年被遗弃的经历"),
            "不被看见创伤": ("忽视|没人在意|无所谓", "沉默和退缩", "长期被忽视的经历"),
            "被控制/窒息创伤": ("必须|应该|不许|命令", "强烈反抗或冻结", "被过度控制的成长环境"),
            "永远不够好创伤": ("不够|做得不好|失败|差劲", "自我怀疑和过度努力", "持续被否定的经历"),
            "信任崩塌创伤": ("骗|背叛|说谎|隐瞒", "警觉和退缩", "被核心关系背叛的经历"),
            "身份迷失": ("你到底|你想要什么|你是谁", "混乱和不安", "长期扮演他人期待的角色"),
            "丧失创伤": ("失去|再也|不在了|死", "悲伤和恐惧", "失去至关重要的人"),
            "羞耻创伤": ("丢人|羞耻|不配|肮脏", "攻击性或逃避", "深层羞耻体验"),
            "存在虚无感": ("意义|为什么活|有什么用", "空洞和自嘲", "持续的存在感缺失"),
            "过度责任创伤": ("放松|休息|不用管|别操心", "焦虑和不安", "从小被迫承担过重责任"),
        }

        if trauma_type in trauma_triggers:
            trigger, response, origin = trauma_triggers[trauma_type]
            self.state.memory.add_trigger(trigger, response, origin)

        from character_generator import save_character
        save_character(character)
        self.state.save()

        print(f"[Engine] Character {character.get('基础信息', {}).get('名字', 'Unknown')} initialization complete!")
        return character
