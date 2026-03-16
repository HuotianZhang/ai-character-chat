# AI Character Chat System — Architecture Document

**Version**: 2.3 (Emotion Vector + Silence Monitor + Debug Tools)
**Last Updated**: 2026-03-16

---

## 1. System Overview

A pure-Python chat system that simulates a realistic AI character with emotions, memory, personality, and life progression. The character communicates through a WeChat-style web interface, driven by Gemini LLM with structured JSON output.

**Core Principle**: The character should feel like a real person texting on their phone — with contradictory emotions, daily routines, mood swings, and attachment patterns that emerge from their personality and trauma history.

---

## 2. Module Map

```
ai_character/
├── main.py                 # HTTP server, API routing, subsystem init
├── config.py               # Global configuration (API keys, paths, params)
├── character_generator.py  # Random character profile generation
├── character_state.py      # State engine (emotion, affinity, memory, storyline)
├── conversation_engine.py  # LLM integration, prompt building, response parsing
├── time_controller.py      # Virtual time system (speed, pause, jump)
├── proactive_events.py     # Character-initiated conversations
├── silence_monitor.py      # Detects user silence, drives emotional reactions
├── debug_tools.py          # Standalone testing: scenarios, comparison, interactive
├── templates/
│   └── chat.html           # WeChat-style chat UI + debug panel
├── data/                   # Persisted state (auto-created)
│   ├── character.json      # Character profile
│   ├── state.json          # Emotion + affinity state
│   ├── memory.json         # Memory system
│   ├── storyline.json      # Storyline progression
│   └── time_state.json     # Virtual time state
└── requirements.txt        # google-generativeai, requests
```

---

## 3. Data Flow

```
[User types message]
       │
       ▼
   main.py (POST /api/chat)
       │
       ▼
   CharacterState.process_input(message)
       ├── EmotionVector.decay()          ← natural emotion decay per axis
       ├── PressureAccumulator.tick()     ← check for pressure bursts
       ├── TensionDetector.detect()       ← identify compound emotional states
       ├── MemorySystem.check_triggers()  ← scan for trauma triggers
       └── build state snapshot
       │
       ▼
   ConversationEngine.chat(message)
       ├── build_system_prompt()          ← inject full character + state
       ├── call_gemini()                  ← LLM generates response as JSON
       ├── parse_llm_response()           ← extract structured output
       └── CharacterState.process_output()
            ├── EmotionVector.apply_stimulus()  ← multi-axis emotion changes
            ├── PressureAccumulator.record()    ← accumulate pressure
            ├── AffinitySystem.modify()
            └── MemorySystem.consolidate()
       │
       ▼
   JSON response → chat.html renders messages
```

---

## 4. Module Details

### 4.1 `config.py`
Global constants: Gemini API key/model, file paths, system parameters (memory limits, emotion baseline, affinity range, server host/port).

### 4.2 `character_generator.py` (644 lines)
Generates complete character profiles using weighted randomization:

- **Basic info**: Name (CN/EN pools), birthday (Gaussian), height (Gaussian), city (weighted by region), body type, energy level
- **Appearance**: Face shape, feature intensity, bone-flesh ratio, skin tone, age-feel, ethnicity look, hair (length/curl/style/color)
- **Personality dimensions** (5 axes, each with type + degree 1-5):
  - Attachment style (12 types: secure, anxious, avoidant, fearful-avoidant variants)
  - Outer aura (10 types)
  - Cognitive style (10 types)
  - Value priority (10 types)
  - Core trauma (10 types)
- **Language fingerprint**: Message segmentation (1-5), punctuation personality, interjection frequency, emoji patterns, typing style, A/B emotion expression type
- **Derived attributes**: Zodiac, MBTI (heuristic), expression rules, emotion decay rates
- **Blacklist validation**: Prevents contradictory personality combinations (e.g., "emotionally stable" + "anxious attachment")

### 4.3 `character_state.py` — The Living Core

#### EmotionVector (NEW — replaces old single-scalar EmotionSystem)
7 independent emotion axes, each 0.0–1.0, coexisting simultaneously:

| Axis | Description | Example triggers |
|------|-------------|-----------------|
| joy | Happiness, excitement | Compliments, good news, fun conversations |
| sadness | Grief, loss, melancholy | Being ignored, bad memories, disappointment |
| anger | Frustration, rage | Disrespect, boundary violations, triggers |
| anxiety | Worry, nervousness | Uncertainty, perceived rejection, pressure |
| trust | Safety, openness | Consistency, vulnerability shared, reliability |
| disgust | Revulsion, contempt | Boundary crossing, moral violations |
| attachment | Longing, dependency | Absence, intimacy, separation cues |

Key behaviors:
- Each axis decays independently toward 0 (per-axis decay rates derived from personality)
- Opposing axes DON'T cancel — both can be high simultaneously (enabling contradictory states)
- `get_dominant()` returns the 2-3 strongest axes for prompt injection
- `total_energy()` = sum of all axes (high energy = emotionally charged regardless of valence)

#### TensionDetector (NEW)
Detects compound emotional states when conflicting axes are both elevated:

| Pattern | Axes Required | Chinese Label | Behavioral Signature |
|---------|--------------|---------------|---------------------|
| love_hate | attachment≥0.6 AND anger≥0.5 | 又爱又恨 | Contradictory messages, push-pull |
| anxious_joy | joy≥0.5 AND anxiety≥0.5 | 患得患失 | Happy but afraid it won't last |
| grievance | anger≥0.5 AND sadness≥0.5 | 委屈 | Wronged, wants acknowledgment |
| numb | all axes < 0.2, total energy < 0.5 | 麻木 | Flat affect, minimal response |
| overwhelmed | total energy ≥ 4.0 | 情绪过载 | Chaotic, may shut down |

Each tension state carries language fingerprint overrides (e.g., love_hate → increased message segmentation, contradictory statements).

#### PressureAccumulator (NEW)
Hidden pressure values that build from repeated small stimuli:

- **Channels**: no_reply, criticized, controlled, ignored, boundary_pushed
- **Mechanics**: Pressure += amount per event, slow natural decay over time
- **Burst threshold**: When pressure ≥ 1.0, triggers non-linear emotional explosion
  - Burst direction depends on attachment style (anxious → anger explosion, avoidant → withdrawal/freeze)
  - Post-burst: pressure resets, threshold lowers by 10% (sensitization)
  - Burst injects large emotion vector stimulus
- **LLM awareness**: Active pressure levels injected into system prompt so character can reference building frustration

#### AffinitySystem (unchanged)
Dual-track affinity: normal (0-100) + special/attachment slot (0-100). Affinity level determines conversation intimacy depth. Dark line unlocks at ≥85.

#### MemorySystem (unchanged)
Four layers: short-term conversation (last 30 turns), long-term consolidated (high-emotion events), emotional triggers (trauma patterns), semantic memory (facts about user).

#### StorylineSystem (unchanged)
7-day storyline with per-slot events, emotion stages, hidden triggers, and affinity hints. Managed by TimeController for progression.

### 4.4 `conversation_engine.py`

#### `build_system_prompt()`
Massive prompt (~2500+ chars) injecting:
- Character identity, appearance, personality, language fingerprint
- **Emotion vector state** (all 7 axes with values)
- **Active tension states** with behavioral instructions
- **Pressure levels** for relevant channels
- Current affinity, triggered memories, semantic memory, storyline events
- Behavior rules (mood-driven responses, affinity-gated intimacy, language fingerprint enforcement)
- Structured JSON output format specification

#### `parse_llm_response()`
Extracts JSON from LLM output. New format includes multi-axis emotion changes:
```json
{
  "reply": "...",
  "emotion_changes": {"joy": 0.1, "anxiety": -0.05, "trust": 0.15},
  "affinity_delta": 0,
  "special_affinity_delta": 0,
  "memory_note": "",
  "semantic_updates": {},
  "inner_thought": ""
}
```

#### `call_gemini()`
SDK-first with HTTP fallback, exponential backoff retry (5 attempts), rate limit handling.

#### `generate_storyline()` / `generate_character_backstory()`
LLM-generated content with retry logic and default fallbacks.

### 4.5 `time_controller.py`
Virtual time decoupled from real time:
- **Speed multiplier**: 1x (realtime), 60x (1 sec = 1 min), 600x, 3600x (1 sec = 1 hour)
- **Time slots**: morning (6-12), afternoon (12-18), evening (18-22), night (22-6)
- **Controls**: pause/resume, jump to slot, jump forward N hours
- **Persistence**: Saves/loads virtual time anchor to `time_state.json`
- `check_transitions()` detects slot/day changes for event triggering

### 4.6 `proactive_events.py`
Character-initiated messages based on:
1. **Storyline events**: When virtual time reaches a slot with a story event, character shares it
2. **Routine messages**: Morning greeting (affinity≥75), evening check-in (≥80), good night (≥85)
3. **Mood triggers**: Very happy (mood≥8.5) shares excitement; very sad (mood≤3.0 + affinity≥80) seeks comfort

Rate-limited: min 10 real seconds between proactive messages. Each proactive message calls LLM with context-appropriate prompt.

### 4.7 `templates/chat.html`
WeChat-style single-page app:
- Message bubbles (user right, character left), typing indicators
- Time control bar: speed buttons, slot jump buttons, pause/resume
- Event polling (3-second interval) for proactive messages
- Debug panel: emotion axes, affinity, memory stats, storyline info

---

## 5. API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve chat.html |
| GET | `/api/status` | Character existence/load status + time |
| POST | `/api/init` | Create new or load existing character |
| POST | `/api/chat` | Send message, get character response |
| GET | `/api/character` | Full character info + state dump |
| GET | `/api/time` | Virtual time status |
| POST | `/api/time/speed` | Set time speed multiplier |
| POST | `/api/time/pause` | Pause virtual time |
| POST | `/api/time/resume` | Resume virtual time |
| POST | `/api/time/jump` | Jump to slot or forward N hours |
| POST | `/api/advance_time` | Legacy: advance to next time slot |
| GET | `/api/events` | Poll for proactive messages |

---

## 6. Character Generation Pipeline

```
generate_character()
  └── _generate_raw_character()     ← random rolls for all attributes
  └── check_blacklist()             ← validate no contradictory combos
  └── _derive_attributes()
       ├── get_zodiac()
       ├── calculate_emotion_decay_rate()   ← per-axis rates (NEW)
       ├── _derive_mbti()
       └── _derive_expression_rules()
```

Then on first load:
```
ConversationEngine.initialize_character()
  ├── generate_character_backstory()  ← LLM call (family/edu/career/love/trauma)
  ├── generate_storyline()            ← LLM call (7-day event schedule)
  └── set trauma triggers             ← keyword → emotional response mappings
```

---

## 7. Persistence Model

All state persists to JSON files in `data/`:
- `character.json`: Full character profile (generated once, updated with backstory)
- `state.json`: EmotionVector axes + PressureAccumulator + AffinitySystem + interaction stats
- `memory.json`: All 4 memory layers
- `storyline.json`: 7-day storyline + current position
- `time_state.json`: Virtual time anchor + speed + pause state

Save triggers: after every chat response, after time jumps, after proactive events.

---

## 8. LLM Integration Notes

- **Model**: Gemini 2.5 Flash (free tier: 10 RPM, 250K TPM, 250 RPD)
- **Temperature**: 0.85 for dialogue, 0.9 for storyline generation, 0.85 for backstory
- **Max tokens**: 1024 for chat, 3000-4096 for generation tasks
- **Retry**: 5 attempts with exponential backoff for 429/500/503
- **Structured output**: LLM returns JSON with reply + emotion changes + affinity + memory notes + inner thought
- **System prompt**: ~3000 chars, rebuilt every turn with fresh state snapshot

---

## 9. Changelog

### v2.4 (Current)
- Hard-limit proactive silence messages: replaced probabilistic `trigger_chance` with strict per-personality budgets
  - secure/avoidant: 0 messages (never chase when ignored — realistic)
  - anxious: max 1 message, only at phase 3+ (90+ min silence) AND emotion energy ≥ 1.5
  - fearful: max 1 message, only at phase 3+ AND emotion energy ≥ 2.0
  - Deterministic `_should_send_silence_message()` replaces random rolls
- Unified output format: proactive events and conversation engine now share:
  - `format_reply_messages()`: shared message splitting (|||, [Read])
  - `apply_parsed_output()`: shared state update from LLM response
  - Both return same core fields: `reply`, `messages`, `status`, `inner_thought`
  - Proactive adds: `proactive`, `event_type`, `time_display`, `storyline_event`

### v2.3
- Added SilenceMonitor: detects user silence, applies graduated emotion/pressure per personality
  - 5 silence phases (0-10-30-90-240 virtual minutes)
  - 4 personality profiles (anxious/avoidant/secure/fearful) with distinct reactions
  - Emotional effects continue even without proactive messages
  - Pressure accumulation on "no_reply" channel with burst mechanics
- Added debug_tools.py: standalone emotional realism testing
  - Interactive mode, scenario simulation, attachment style comparison
  - 6 built-in scenarios: repeated_no_reply, love_bomb, slow_ghost, criticism_spiral, push_pull, boundary_violation
- Added debug API endpoints: /api/debug/state, /api/debug/inject, /api/debug/simulate_silence
- Migrated from deprecated google.generativeai to google.genai SDK

### v2.1
- Replace single-scalar EmotionSystem with 7-axis EmotionVector
- Add TensionDetector for compound emotional states
- Add PressureAccumulator for non-linear emotion buildup
- Update system prompt for multi-axis emotion awareness
- Update LLM response format: `emotion_changes` dict replaces scalar `emotion_delta`

### v2.0
- Added TimeController (virtual time with speed control)
- Added ProactiveEventSystem (character-initiated messages)
- Rewrote main.py from Flask to pure Python stdlib HTTP server
- Added WeChat-style chat.html with time controls and debug panel

### v1.0
- Initial system: character generator, single-scalar emotion, affinity, memory, storyline
- Gemini API integration with SDK + HTTP fallback
- Basic chat interface
