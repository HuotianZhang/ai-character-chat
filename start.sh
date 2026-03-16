#!/bin/bash
echo "=========================================="
echo "  AI 角色聊天系统 - 启动中..."
echo "=========================================="
echo

# 安装依赖
echo "[1/2] 安装依赖..."
pip install -r requirements.txt -q 2>/dev/null || pip3 install -r requirements.txt -q

# 检查 API Key
if [ -z "$GEMINI_API_KEY" ]; then
    echo
    echo "[提示] 未检测到 GEMINI_API_KEY 环境变量"
    echo "请在 config.py 中填入你的 Gemini API Key"
    echo "或运行: export GEMINI_API_KEY=你的key"
    echo
fi

# 启动
echo "[2/2] 启动服务器..."
echo
echo "请打开浏览器访问: http://localhost:5000"
echo "按 Ctrl+C 停止服务器"
echo
python main.py 2>/dev/null || python3 main.py
