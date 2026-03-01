"""
Agent核心模块
负责对话管理、LLM调用、工具调用循环。
"""

import json
import os
from datetime import datetime

import requests

import config
from tools import TOOLS, execute_tool


# ============================================================
# 对话存储
# ============================================================


def _conversation_path(session_id: str) -> str:
    """获取指定 session_id 对应的对话文件路径。"""
    return os.path.join(config.CONVERSATIONS_DIR, f"{session_id}.json")


def load_conversation(session_id: str) -> list:
    """
    加载对话历史。

    Returns:
        对话消息列表，格式:
        [{"role": "user"/"agent", "timestamp": "...", "content": "..."}]
    """
    path = _conversation_path(session_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_conversation(session_id: str, messages: list) -> None:
    """将对话历史保存到文件。"""
    os.makedirs(config.CONVERSATIONS_DIR, exist_ok=True)
    path = _conversation_path(session_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)


# ============================================================
# 消息格式转换
# ============================================================


def _stored_to_llm_messages(stored_messages: list) -> list:
    """
    将存储格式的消息转换为 OpenAI API 格式。
    存储格式 role: user/agent  ->  OpenAI格式 role: user/assistant
    """
    llm_messages = []
    for msg in stored_messages:
        role = "assistant" if msg["role"] == "agent" else msg["role"]
        llm_messages.append({"role": role, "content": msg["content"]})
    return llm_messages


def _now_timestamp() -> str:
    """获取当前时间戳字符串。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============================================================
# LLM 调用
# ============================================================


def _call_llm(messages: list, tools: list = None) -> dict:
    """
    调用 OpenAI 兼容的 Chat Completion API。

    Args:
        messages: OpenAI 格式的消息列表
        tools: 工具定义列表（可选）

    Returns:
        API 响应的 JSON 字典
    """
    url = config.LLM_API_BASE.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.LLM_API_KEY}",
    }

    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools

    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    if resp.status_code != 200:
        # 将详细错误信息抛出，便于调试
        raise RuntimeError(
            f"LLM API 返回 {resp.status_code}: {resp.text}"
        )
    return resp.json()


# ============================================================
# Agent 主循环
# ============================================================


def chat(session_id: str, user_message: str) -> str:
    """
    处理一次用户消息，返回 Agent 的回复。

    流程:
    1. 加载历史对话
    2. 追加用户消息
    3. 构造 LLM 消息（含系统提示词）
    4. 循环: 调用 LLM -> 若有 tool_calls 则执行并回传 -> 否则返回文本
    5. 保存对话

    Args:
        session_id: 会话 ID
        user_message: 用户消息文本

    Returns:
        Agent 的回复文本
    """
    # 1. 加载历史
    conversation = load_conversation(session_id)

    # 2. 追加用户消息到存储
    conversation.append(
        {
            "role": "user",
            "timestamp": _now_timestamp(),
            "content": user_message,
        }
    )

    # 3. 构造发送给 LLM 的消息
    llm_messages = [{"role": "system", "content": config.SYSTEM_PROMPT}]
    llm_messages.extend(_stored_to_llm_messages(conversation))

    # 4. Agent 循环（工具调用）
    for _ in range(config.MAX_TOOL_ROUNDS):
        response = _call_llm(llm_messages, tools=TOOLS)

        choice = response["choices"][0]
        message = choice["message"]

        # 4a. 如果没有 tool_calls，说明是最终文本回复
        if not message.get("tool_calls"):
            reply_content = message.get("content", "")

            # 5. 保存 Agent 回复到对话历史
            conversation.append(
                {
                    "role": "agent",
                    "timestamp": _now_timestamp(),
                    "content": reply_content,
                }
            )
            save_conversation(session_id, conversation)

            return reply_content

        # 4b. 有 tool_calls，执行工具并回传结果
        # 先将 assistant 的 tool_calls 消息追加到 LLM 消息列表
        llm_messages.append(message)

        for tool_call in message["tool_calls"]:
            func_name = tool_call["function"]["name"]

            # 解析参数
            try:
                func_args = json.loads(tool_call["function"]["arguments"])
            except (json.JSONDecodeError, TypeError):
                func_args = {}

            # 执行工具
            tool_result = execute_tool(func_name, func_args)

            # 将工具结果追加到 LLM 消息
            llm_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": tool_result,
                }
            )

    # 达到最大轮次，返回提示
    fallback = "抱歉，处理您的请求时工具调用轮次过多，请尝试简化您的问题。"
    conversation.append(
        {
            "role": "agent",
            "timestamp": _now_timestamp(),
            "content": fallback,
        }
    )
    save_conversation(session_id, conversation)
    return fallback
