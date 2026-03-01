"""
租房Agent HTTP 服务入口
提供 POST /chat 接口，通过 curl 调试。
"""

import uuid

from flask import Flask, request, jsonify

from agent import chat

app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """
    问答接口

    请求体 (JSON):
        {
            "session_id": "abc123",   // 可选，不传则自动生成
            "message": "帮我找西二旗附近的两居室"  // 必填
        }

    响应体 (JSON):
        {
            "session_id": "abc123",
            "reply": "好的，我来帮您查找..."
        }

    curl 调试示例:
        curl -X POST http://localhost:5000/chat \
            -H "Content-Type: application/json" \
            -d "{\"session_id\":\"test1\",\"message\":\"帮我找西二旗附近3000以下的一居室\"}"
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "请求体必须为 JSON 格式"}), 400

    message = data.get("message", "").strip()
    if not message:
        return jsonify({"error": "message 字段不能为空"}), 400

    session_id = data.get("session_id", "").strip()
    if not session_id:
        session_id = uuid.uuid4().hex

    try:
        reply = chat(session_id, message)
        return jsonify({"session_id": session_id, "reply": reply})
    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
