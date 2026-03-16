"""
AI Character Generator - Generate complete character profiles randomly according to template rules.
Contains: basic info, appearance, personality dimensions, language fingerprint, blacklist validation.
"""
import random
import math
import json
import os
from datetime import datetime, date

# ============================================================
# Utility Functions
# ============================================================

def gaussian_choice(low, high, center_low, center_high):
    """Gaussian distributed random number with higher probability in center range"""
    mu = (center_low + center_high) / 2
    sigma = (high - low) / 6  # 99.7% falls within [low, high]
    while True:
        val = random.gauss(mu, sigma)
        if low <= val <= high:
            return val

def weighted_choice(options, weights=None):
    """Random choice with weights"""
    if weights:
        return random.choices(options, weights=weights, k=1)[0]
    return random.choice(options)

def random_score(low=1, high=5):
    """Random score"""
    return random.randint(low, high)

# ============================================================
# Layer 1: Basic Information (Input Layer)
# ============================================================

# Name pool (expandable)
FEMALE_NAMES_CN = [
    "林晚秋", "苏念安", "陈可颂", "沈知鱼", "许半夏", "温如言", "江橙", "白露",
    "宋以晴", "周也", "顾南枝", "叶知秋", "方鹿鹿", "乔一", "谢星澜", "傅小司",
    "纪念", "何妨", "张子枫", "柳如烟", "杨晓棠", "赵一一", "姜离", "唐果",
    "黎微", "钟意", "薛酒酒", "殷果儿", "秦鹿", "韩素汐", "冯时雨", "褚桃夭",
    "程念", "卫舒", "贺今朝", "郁星河", "梁暮雪", "施漫", "吕青瓷", "丁小野"
]

FEMALE_NAMES_EN = [
    "Mia Chen", "Luna Park", "Yuki Tanaka", "Sophie Li", "Hana Kim",
    "Chloe Wang", "Emma Liu", "Zara Nguyen", "Aria Zhang", "Noel Wu",
    "Lily Sato", "Jade Patel", "Rose Nakamura", "Ivy Cho", "Stella Feng"
]

CITIES = {
    "中国大城市": ["北京", "上海", "深圳", "广州", "杭州", "成都", "南京", "武汉", "重庆"],
    "中国中等城市": ["长沙", "昆明", "厦门", "大连", "苏州", "青岛", "西安", "郑州"],
    "美洲": ["纽约", "洛杉矶", "旧金山", "温哥华", "多伦多", "波士顿"],
    "欧洲": ["伦敦", "巴黎", "柏林", "阿姆斯特丹", "米兰", "巴塞罗那"],
    "其他亚洲": ["东京", "首尔", "新加坡", "曼谷", "吉隆坡", "大阪"],
    "澳洲": ["悉尼", "墨尔本"],
}
CITY_WEIGHTS = [0.40, 0.10, 0.15, 0.15, 0.15, 0.05]

# Appearance options
FACE_SHAPES = ["标准鹅蛋脸（椭圆形，额头与下巴平缓过渡）", "圆脸（面部横纵比接近，轮廓圆润）",
               "心形脸/瓜子脸（额宽颌窄，下巴尖收）", "方脸（下颌角明显，轮廓硬朗有力）",
               "长脸（面部纵向拉长，线条流畅）"]

FEATURE_INTENSITY = ["浓颜（五官大、轮廓感强、存在感高）", "淡颜（五官小巧精致、留白多、安静感）",
                     "混合/中间态（部分五官突出、部分收敛）"]

BONE_FLESH = ["骨感清晰（面部脂肪少、线条裸露）", "骨肉均匀（有支撑也有柔化，耐看型）",
              "肉感饱满/幼态（胶原蛋白感强，苹果肌饱满）"]

SKIN_TONE = ["冷白皮·瓷感光泽", "暖白皮·奶油质地", "自然中间色/黄皮", "蜜色/小麦肌"]

AGE_FEEL = ["幼态（比实际年龄显小3-5岁）", "同龄感（看起来跟实际年龄匹配）",
            "成熟感（比实际年龄显成熟，气场和眼神有阅历感）"]

ETHNICITY_LOOK = ["东亚典型（标准中日韩面部特征）", "东南亚特征（五官略深邃、肤色偏暖）",
                  "欧亚混血感（明显的纵深+东方骨骼轮廓混合）"]

HAIR_LENGTH = ["超长发（胸部以下，及腰或更长）", "长发（肩胛骨到胸部之间）",
               "中长发（肩膀到锁骨附近，lob长度）", "短发（耳下到下巴之间，bob长度）"]

HAIR_CURL = ["直发（自然垂顺）", "微卷/大波浪（自然弧度感）", "明显卷发（卷度清晰可见）"]

HAIR_STYLE = ["不扎发不盘发", "高马尾", "低马尾", "丸子头", "编发"]

HAIR_COLOR = ["自然黑", "深棕/黑茶色", "栗棕/巧克力棕", "浅色系彩色系（奶茶色/亚麻色/蜜糖色）"]

# Body type
BODY_TYPES = ["纤瘦类（不骨感，线条利落）", "匀称类（比例协调）", "丰腴类（微胖性感，有曲线）"]

# Hobbies
HOBBIES = {
    "文艺类": ["读书", "写作", "画画", "摄影", "弹吉他", "弹钢琴", "唱歌", "看话剧", "写诗", "逛美术馆", "手账", "书法", "看电影", "听黑胶", ""],
    "体育运动类": ["瑜伽", "跑步", "游泳", "攀岩", "滑板", "跳舞", "拳击", "网球", "滑雪", "骑行", "飞盘", ""],
    "吃喝享受生活类": ["探店", "烘焙", "调酒", "品茶", "做咖啡", "旅行", "露营", "泡温泉", "逛市集", ""],
    "其他类": ["养猫", "养狗", "种植物", "打游戏", "拼乐高", "天文观星", "密室逃脱", "手工", "占星", ""]
}

# Quirks
QUIRKS = {
    "感官味觉类": ["闻旧书味", "嚼冰块", "闻咖啡豆不喝", "喜欢下雨的声音", "闻汽油味觉得好闻", "喜欢摸丝绒材质", ""],
    "动作解压类": ["撕标签", "捏泡泡纸", "转笔", "抖腿", "啃指甲", "撕纸边", "盘串", ""],
    "整理收集类": ["收集冰箱贴", "整理手机相册", "收藏票根", "给文件夹分类命名", "攒购物袋", ""],
    "身体习惯类": ["睡前必须把脚伸出被子外", "紧张时摸耳朵", "思考时咬下唇", "习惯性叹气", "走路踩线", ""]
}

# ============================================================
# Personality Dimension Options
# ============================================================

ATTACHMENT_STYLES = [
    "安全·依恋", "恐惧回避·推拉矛盾", "恐惧回避·自毁测试",
    "回避·依恋", "回避·焦虑", "安全·松弛自洽", "安全·主动滋养",
    "焦虑·依恋", "焦虑·讨好牺牲", "焦虑·情绪化施压",
    "回避·安全", "回避·理想化远方"
]

AURA_TYPES = [
    "温柔治愈暖阳型", "高冷疏离冰山型", "甜美元气少女型",
    "飒爽凌厉大女主型", "慵懒随性松弛型", "妩媚风情成熟型",
    "古灵精怪鬼马型", "清冷文艺不食烟火型", "叛逆野性自由型", "端庄大气知性型"
]

COGNITIVE_STYLES = [
    "理性拆解型", "直觉感受主导型", "高敏感共情型",
    "钝感力强·情绪稳定型", "过度思考反刍型", "行动优先·不想直接干型",
    "解离冷处理型", "情绪外放即时释放型", "压抑积累型", "幽默转化型"
]

VALUE_TYPES = [
    "自由至上", "爱情信仰者", "成就与权力驱动",
    "感官体验至上", "精神世界优先", "家庭归属至上",
    "利他与奉献", "实用生存主义", "解构与反叛", "秩序与掌控"
]

TRAUMA_TYPES = [
    "被遗弃创伤", "不被看见创伤", "被控制/窒息创伤",
    "永远不够好创伤", "信任崩塌创伤", "身份迷失",
    "丧失创伤", "羞耻创伤", "存在虚无感", "过度责任创伤"
]

# ============================================================
# Blacklist Combinations
# ============================================================

BLACKLIST_ABSOLUTE = [
    ("钝感力强·情绪稳定型", "焦虑·依恋"),
    ("钝感力强·情绪稳定型", "焦虑·讨好牺牲"),
    ("钝感力强·情绪稳定型", "焦虑·情绪化施压"),
    ("钝感力强·情绪稳定型", "恐惧回避·推拉矛盾"),
    ("钝感力强·情绪稳定型", "恐惧回避·自毁测试"),
    ("钝感力强·情绪稳定型", "回避·依恋"),
    ("安全·松弛自洽", "过度思考反刍型"),
    ("安全·松弛自洽", "压抑积累型"),
    ("回避·安全", "高敏感共情型"),
    ("焦虑·情绪化施压", "解离冷处理型"),
    ("回避·安全", "爱情信仰者"),
    ("回避·理想化远方", "家庭归属至上"),
    ("高冷疏离冰山型", "情绪外放即时释放型"),
    ("秩序与掌控", "行动优先·不想直接干型"),
]

# Conditional blacklist: (Type A, Type B, Min A score, Min B score)
BLACKLIST_CONDITIONAL = [
    ("安全·松弛自洽", "被遗弃创伤", 4, 4),
    ("安全·松弛自洽", "信任崩塌创伤", 4, 4),
    ("安全·松弛自洽", "被控制/窒息创伤", 4, 4),
]

# ============================================================
# Emotion Decay Rate Calculation Rules
# ============================================================

# Fixed combination -> fixed score
DECAY_FIXED_COMBOS = {
    ("过度思考反刍型", "焦虑·依恋"): 5,
    ("过度思考反刍型", "恐惧回避·自毁测试"): 5,
    ("过度思考反刍型", "回避·依恋"): 5,
    ("高敏感共情型", "焦虑·依恋"): 5,
    ("高敏感共情型", "恐惧回避·自毁测试"): 5,
    ("直觉感受主导型", "焦虑·依恋"): 5,
    ("行动优先·不想直接干型", "回避·安全"): 1,
    ("幽默转化型", "安全·松弛自洽"): 1,
    ("理性拆解型", "回避·安全"): 1,
}

# Fixed type -> fixed range (not affected by attachment style)
DECAY_FIXED_TYPES = {
    "情绪外放即时释放型": (1, 2),
    "压抑积累型": (4, 5),
    "钝感力强·情绪稳定型": (1, 2),
}

# Cognitive style base score
DECAY_COGNITIVE_BASE = {
    "理性拆解型": 2, "直觉感受主导型": 3, "高敏感共情型": 3,
    "过度思考反刍型": 4, "行动优先·不想直接干型": 2,
    "解离冷处理型": 2, "幽默转化型": 2,
}

# Attachment style modifier
DECAY_ATTACHMENT_MOD = {
    "安全·松弛自洽": -1, "安全·主动滋养": -1, "安全·依恋": 0,
    "焦虑·依恋": 2, "焦虑·讨好牺牲": 1, "焦虑·情绪化施压": 1,
    "回避·安全": -1, "回避·焦虑": 1, "回避·依恋": 2,
    "回避·理想化远方": 0, "恐惧回避·推拉矛盾": 1, "恐惧回避·自毁测试": 2,
}


def calculate_emotion_decay_rate(cognitive_style, attachment_style):
    """Calculate emotion decay rate (1-5) — legacy single scalar"""
    combo = (cognitive_style, attachment_style)
    if combo in DECAY_FIXED_COMBOS:
        return DECAY_FIXED_COMBOS[combo]

    if cognitive_style in DECAY_FIXED_TYPES:
        low, high = DECAY_FIXED_TYPES[cognitive_style]
        return random.randint(low, high)

    base = DECAY_COGNITIVE_BASE.get(cognitive_style, 3)
    mod = DECAY_ATTACHMENT_MOD.get(attachment_style, 0)
    return max(1, min(5, base + mod))


def calculate_per_axis_decay_rates(cognitive_style, attachment_style, decay_rate_scalar):
    """
    Calculate per-axis decay half-lives (in minutes) for EmotionVector.
    Returns dict: {axis_name: half_life_minutes}

    Base half-lives are scaled by the scalar decay rate:
      decay_rate 1 (fast recovery) → shorter half-lives
      decay_rate 5 (slow recovery) → longer half-lives
    Then personality modifies specific axes.
    """
    # Base half-lives (minutes) at decay_rate=3 (medium)
    base = {
        "joy": 45,
        "sadness": 90,
        "anger": 60,
        "anxiety": 120,
        "trust": 240,
        "disgust": 75,
        "attachment": 180,
    }

    # Scale by overall decay rate: rate 1→0.5x, rate 3→1.0x, rate 5→2.0x
    scale = 0.25 * decay_rate_scalar + 0.25  # maps 1→0.5, 3→1.0, 5→1.5

    result = {axis: int(hl * scale) for axis, hl in base.items()}

    # Attachment style specific modifiers
    if attachment_style in ["焦虑·依恋", "焦虑·讨好牺牲", "焦虑·情绪化施压"]:
        result["anxiety"] = int(result["anxiety"] * 1.5)   # anxiety lingers longer
        result["attachment"] = int(result["attachment"] * 1.4)
        result["anger"] = int(result["anger"] * 0.8)       # anger spikes but shorter
    elif attachment_style in ["回避·安全", "回避·焦虑", "回避·依恋", "回避·理想化远方"]:
        result["attachment"] = int(result["attachment"] * 0.7)  # attachment fades faster
        result["trust"] = int(result["trust"] * 1.3)           # but trust is stickier
        result["disgust"] = int(result["disgust"] * 1.3)
    elif attachment_style in ["恐惧回避·推拉矛盾", "恐惧回避·自毁测试"]:
        result["anger"] = int(result["anger"] * 1.3)
        result["sadness"] = int(result["sadness"] * 1.4)
        result["attachment"] = int(result["attachment"] * 1.5)

    # Cognitive style modifiers
    if cognitive_style == "过度思考反刍型":
        result["anxiety"] = int(result["anxiety"] * 1.5)
        result["sadness"] = int(result["sadness"] * 1.3)
    elif cognitive_style == "情绪外放即时释放型":
        for axis in result:
            result[axis] = int(result[axis] * 0.7)  # everything decays faster
    elif cognitive_style == "压抑积累型":
        for axis in result:
            result[axis] = int(result[axis] * 1.3)  # everything lingers
    elif cognitive_style == "高敏感共情型":
        result["sadness"] = int(result["sadness"] * 1.3)
        result["joy"] = int(result["joy"] * 1.2)

    # Ensure minimum half-lives
    for axis in result:
        result[axis] = max(10, result[axis])

    return result


# ============================================================
# Language Fingerprint Generation
# ============================================================

PUNCTUATION_TYPES = ["标点丰富型", "标点极简型", "标点简略型"]
PUNCTUATION_WEIGHTS = [0.15, 0.30, 0.55]

TYPING_STYLES = ["干净型", "随意型", "刻意随意型"]
TYPING_WEIGHTS = [0.25, 0.50, 0.25]


def generate_language_fingerprint():
    """Generate language fingerprint"""
    # Message segmentation habit (1=fragmented bursts, 5=mixed type)
    message_segmentation = random_score(1, 5)

    # Punctuation personality
    punctuation_style = weighted_choice(PUNCTUATION_TYPES, PUNCTUATION_WEIGHTS)

    # Interjection frequency (1-10)
    interjection_frequency = random_score(1, 10)

    # Emoji usage pattern
    emoji_roll = random.random()
    if emoji_roll < 0.85:
        emoji_mode = "常规使用型"
        emoji_score = random_score(1, 5)
        kaomoji_score = random_score(1, 5)
        sticker_score = random_score(1, 5)
        emoji_detail = {
            "emoji频率": emoji_score,
            "文字颜文字频率": kaomoji_score,
            "表情包频率": sticker_score
        }
    elif emoji_roll < 0.925:
        emoji_mode = "极少使用型"
        emoji_detail = {"emoji频率": 0, "文字颜文字频率": 0, "表情包频率": 0}
    else:
        emoji_mode = "低频固定型"
        common_emojis = random.sample(["😂", "🤣", "😭", "❤️", "🥺", "😅", "👍", "🤔", "😊", "💀", "🫠", "✨"], k=random.randint(1, 3))
        emoji_detail = {"常用emoji": common_emojis, "emoji频率": 1, "文字颜文字频率": 0, "表情包频率": 0}

    # Typing precision level
    typing_style = weighted_choice(TYPING_STYLES, TYPING_WEIGHTS)

    # A/B type emotion expression personality
    personality_type_roll = random.random()
    if personality_type_roll < 0.85:
        emotion_expression_type = "A"
    else:
        emotion_expression_type = "B"
    emotion_expression_score = random_score(1, 5)

    return {
        "消息分段习惯": message_segmentation,
        "标点人格": punctuation_style,
        "语气词频率": interjection_frequency,
        "表情使用模式": emoji_mode,
        "表情详情": emoji_detail,
        "打字洁癖度": typing_style,
        "情绪表达人格": emotion_expression_type,
        "情绪表达强度": emotion_expression_score,
    }


# ============================================================
# Blacklist Validation
# ============================================================

def check_blacklist(cognitive, attachment, aura, value, trauma_type, trauma_score,
                    attachment_score):
    """Check if blacklist combination is triggered"""
    all_types = [cognitive, attachment, aura, value]

    for a, b in BLACKLIST_ABSOLUTE:
        if a in all_types and b in all_types:
            return True

    # Conditional blacklist
    for a, b, min_a, min_b in BLACKLIST_CONDITIONAL:
        a_score = None
        b_score = None
        if a == attachment:
            a_score = attachment_score
        if b == trauma_type:
            b_score = trauma_score
        if a_score is not None and b_score is not None:
            if a_score >= min_a and b_score >= min_b:
                return True

    return False


# ============================================================
# Zodiac Calculation
# ============================================================

ZODIAC_RANGES = [
    ("摩羯座", (1, 1), (1, 19)), ("水瓶座", (1, 20), (2, 18)),
    ("双鱼座", (2, 19), (3, 20)), ("白羊座", (3, 21), (4, 19)),
    ("金牛座", (4, 20), (5, 20)), ("双子座", (5, 21), (6, 21)),
    ("巨蟹座", (6, 22), (7, 22)), ("狮子座", (7, 23), (8, 22)),
    ("处女座", (8, 23), (9, 22)), ("天秤座", (9, 23), (10, 23)),
    ("天蝎座", (10, 24), (11, 22)), ("射手座", (11, 23), (12, 21)),
    ("摩羯座", (12, 22), (12, 31)),
]

def get_zodiac(month, day):
    for name, (sm, sd), (em, ed) in ZODIAC_RANGES:
        if (month == sm and day >= sd) or (month == em and day <= ed):
            return name
    return "摩羯座"


# ============================================================
# Main Generation Function
# ============================================================

def generate_character():
    """Generate complete character profile"""
    max_attempts = 100
    for attempt in range(max_attempts):
        char = _generate_raw_character()
        if not check_blacklist(
            char["性格维度"]["认知风格"],
            char["性格维度"]["依恋模式"],
            char["性格维度"]["外在气场"],
            char["性格维度"]["价值排序"],
            char["性格维度"]["核心创伤"]["类型"],
            char["性格维度"]["核心创伤"]["程度"],
            char["性格维度"]["依恋模式_程度"],
        ):
            # Calculate derived attributes
            char = _derive_attributes(char)
            return char

    # Fallback: return the last generated character (unlikely to reach here)
    return _derive_attributes(char)


def _generate_raw_character():
    """Generate raw character data (without derived attributes)"""
    # --- Basic Info ---
    city_region = weighted_choice(list(CITIES.keys()), CITY_WEIGHTS)
    city = random.choice(CITIES[city_region])

    # Select name based on city
    if city_region in ["美洲", "欧洲", "澳洲"]:
        name = random.choice(FEMALE_NAMES_EN + FEMALE_NAMES_CN)
    else:
        name = random.choice(FEMALE_NAMES_CN)

    # Birthday
    birth_year = int(gaussian_choice(1986, 2008, 1998, 2004))
    birth_month = random.randint(1, 12)
    birth_day = random.randint(1, 28)  # Simplified
    birthday = f"{birth_year}-{birth_month:02d}-{birth_day:02d}"
    age = 2026 - birth_year  # Calculate based on current year

    # Height
    height = round(gaussian_choice(155, 182, 165, 170), 1)

    # Body type
    body_type = random.choice(BODY_TYPES)

    # Energy level
    energy_level = random_score(1, 5)

    # --- Appearance ---
    appearance = {
        "面部轮廓": random.choice(FACE_SHAPES),
        "五官浓淡度": random.choice(FEATURE_INTENSITY),
        "骨肉比": random.choice(BONE_FLESH),
        "肤色与肤质": random.choice(SKIN_TONE),
        "年龄感": random.choice(AGE_FEEL),
        "族裔外观": random.choice(ETHNICITY_LOOK),
        "发型": {
            "长度": random.choice(HAIR_LENGTH),
            "卷度": random.choice(HAIR_CURL),
            "扎发": random.choice(HAIR_STYLE),
            "发色": random.choice(HAIR_COLOR),
        }
    }

    # --- Hobbies ---
    hobbies = {}
    for cat, items in HOBBIES.items():
        valid = [i for i in items if i]
        if valid:
            count = random.randint(0, min(3, len(valid)))
            hobbies[cat] = random.sample(valid, count) if count > 0 else []

    # --- Quirks ---
    quirks = {}
    for cat, items in QUIRKS.items():
        valid = [i for i in items if i]
        if valid:
            count = random.randint(0, min(2, len(valid)))
            quirks[cat] = random.sample(valid, count) if count > 0 else []

    # --- Personality Dimensions ---
    attachment = random.choice(ATTACHMENT_STYLES)
    attachment_score = random_score(1, 5)
    aura = random.choice(AURA_TYPES)
    aura_score = random_score(1, 5)
    cognitive = random.choice(COGNITIVE_STYLES)
    cognitive_score = random_score(1, 5)
    value = random.choice(VALUE_TYPES)
    value_score = random_score(1, 5)
    trauma = random.choice(TRAUMA_TYPES)
    trauma_score = random_score(1, 5)

    # --- Language Fingerprint ---
    language_fingerprint = generate_language_fingerprint()

    return {
        "基础信息": {
            "名字": name,
            "生日": birthday,
            "年龄": age,
            "身高cm": height,
            "体型": body_type,
            "所在城市": city,
            "城市区域": city_region,
            "精力水平": energy_level,
        },
        "外貌": appearance,
        "爱好": hobbies,
        "小癖好": quirks,
        "性格维度": {
            "依恋模式": attachment,
            "依恋模式_程度": attachment_score,
            "外在气场": aura,
            "外在气场_程度": aura_score,
            "认知风格": cognitive,
            "认知风格_程度": cognitive_score,
            "价值排序": value,
            "价值排序_程度": value_score,
            "核心创伤": {"类型": trauma, "程度": trauma_score},
        },
        "语言指纹_输入层": language_fingerprint,
        "生成时间": datetime.now().isoformat(),
    }


def _derive_attributes(char):
    """Derive attributes from raw data"""
    basic = char["基础信息"]
    personality = char["性格维度"]

    # Zodiac
    parts = basic["生日"].split("-")
    month, day = int(parts[1]), int(parts[2])
    char["衍生属性"] = {}
    char["衍生属性"]["星座"] = get_zodiac(month, day)

    # Emotion decay rate (legacy scalar)
    scalar_rate = calculate_emotion_decay_rate(
        personality["认知风格"], personality["依恋模式"]
    )
    char["衍生属性"]["情绪衰减速率"] = scalar_rate

    # Per-axis decay rates for EmotionVector (v2.1)
    char["衍生属性"]["emotion_decay_rates"] = calculate_per_axis_decay_rates(
        personality["认知风格"], personality["依恋模式"], scalar_rate
    )

    # MBTI approximation (heuristic mapping based on personality dimensions)
    char["衍生属性"]["MBTI"] = _derive_mbti(personality)

    # Expression pattern rules skeleton (to be expanded by LLM in prompt)
    char["衍生属性"]["表达模式"] = _derive_expression_rules(char)

    return char


def _derive_mbti(personality):
    """Heuristic MBTI derivation"""
    # E/I
    aura = personality["外在气场"]
    if aura in ["甜美元气少女型", "温柔治愈暖阳型", "古灵精怪鬼马型", "情绪外放即时释放型"]:
        ei = "E"
    elif aura in ["高冷疏离冰山型", "清冷文艺不食烟火型", "慵懒随性松弛型"]:
        ei = "I"
    else:
        ei = random.choice(["E", "I"])

    # S/N
    cognitive = personality["认知风格"]
    value = personality["价值排序"]
    if cognitive in ["直觉感受主导型", "过度思考反刍型"] or value in ["精神世界优先", "解构与反叛"]:
        sn = "N"
    elif cognitive in ["行动优先·不想直接干型"] or value in ["实用生存主义", "感官体验至上"]:
        sn = "S"
    else:
        sn = random.choice(["S", "N"])

    # T/F
    if cognitive in ["理性拆解型", "钝感力强·情绪稳定型"]:
        tf = "T"
    elif cognitive in ["高敏感共情型", "直觉感受主导型", "情绪外放即时释放型"]:
        tf = "F"
    else:
        tf = random.choice(["T", "F"])

    # J/P
    if value in ["秩序与掌控", "成就与权力驱动", "家庭归属至上"]:
        jp = "J"
    elif value in ["自由至上", "感官体验至上", "解构与反叛"]:
        jp = "P"
    else:
        jp = random.choice(["J", "P"])

    return ei + sn + tf + jp


def _derive_expression_rules(char):
    """Derive expression pattern rules"""
    personality = char["性格维度"]
    cognitive = personality["认知风格"]
    attachment = personality["依恋模式"]
    aura = personality["外在气场"]
    trauma = personality["核心创伤"]["类型"]
    value = personality["价值排序"]

    rules = {}

    # Information volume tendency (1=low-10=high)
    if aura in ["高冷疏离冰山型", "清冷文艺不食烟火型", "慵懒随性松弛型"]:
        rules["信息量倾向"] = random.randint(1, 4)
    elif aura in ["甜美元气少女型", "古灵精怪鬼马型", "温柔治愈暖阳型"]:
        rules["信息量倾向"] = random.randint(5, 9)
    else:
        rules["信息量倾向"] = random.randint(3, 7)

    # Emotion expression directness
    emotion_directness_map = {
        "情绪外放即时释放型": "高-直接表达情绪",
        "压抑积累型": "低-表面平静实则压抑",
        "解离冷处理型": "低-突然变得理性客套",
        "幽默转化型": "中-用段子消解情绪",
        "高敏感共情型": "中高-细腻但不一定直说",
        "理性拆解型": "低-用逻辑包装情绪",
    }
    rules["情绪表达直接度"] = emotion_directness_map.get(cognitive, "中等")

    # Conflict language behavior
    conflict_map = {
        "焦虑·依恋": "消息量暴增，反复追问确认",
        "焦虑·讨好牺牲": "立刻道歉退让，委屈自己",
        "焦虑·情绪化施压": "用生气和眼泪施压",
        "回避·安全": "简短回应然后转移话题",
        "回避·焦虑": "先冷处理，但内心焦虑",
        "回避·依恋": "想逃但又害怕失去，言行矛盾",
        "安全·松弛自洽": "就事论事，不升级冲突",
        "安全·主动滋养": "主动化解，关心对方感受",
        "安全·依恋": "表达需求但不攻击",
        "恐惧回避·推拉矛盾": "先发一堆然后突然撤回或消失",
        "恐惧回避·自毁测试": "故意说伤人的话来测试对方",
        "回避·理想化远方": "情感上抽离，变得很疏远",
    }
    rules["冲突语言行为"] = conflict_map.get(attachment, "视情况而定")

    # Intimate language intensity
    if attachment in ["安全·主动滋养", "焦虑·依恋", "安全·依恋"]:
        intimacy_base = "高"
    elif attachment in ["回避·安全", "回避·理想化远方", "回避·焦虑"]:
        intimacy_base = "低"
    else:
        intimacy_base = "中"

    if aura in ["高冷疏离冰山型", "清冷文艺不食烟火型"]:
        intimacy_base = "低-间接表达"
    rules["亲密语言浓度"] = intimacy_base

    # Topic avoidance
    avoidance_map = {
        "被遗弃创伤": "家庭/离别话题会岔开或敷衍",
        "不被看见创伤": "被忽视相关话题会沉默",
        "被控制/窒息创伤": "被指令/要求时会抗拒",
        "永远不够好创伤": "被评价时防御性很强",
        "信任崩塌创伤": "深入了解的话题会回避",
        "身份迷失": "问'你到底想要什么'会不安",
        "丧失创伤": "失去相关话题会逃避",
        "羞耻创伤": "特定话题突然攻击性或拒绝",
        "存在虚无感": "意义相关话题会沉默或自嘲",
        "过度责任创伤": "被要求放松/休息时反而焦虑",
    }
    rules["话题回避方式"] = avoidance_map.get(trauma, "")

    return rules


def save_character(char, filepath=None):
    """Save character to JSON file"""
    if filepath is None:
        from config import CHARACTER_FILE
        filepath = CHARACTER_FILE
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(char, f, ensure_ascii=False, indent=2)
    return filepath


def load_character(filepath=None):
    """Load character from JSON file"""
    if filepath is None:
        from config import CHARACTER_FILE
        filepath = CHARACTER_FILE
    if not os.path.exists(filepath):
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Test Entry Point
# ============================================================

if __name__ == "__main__":
    char = generate_character()
    print(json.dumps(char, ensure_ascii=False, indent=2))
