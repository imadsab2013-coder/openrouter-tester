import streamlit as st
import requests
import json
import time

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="OpenRouter Key Tester",
    page_icon="🔬",
    layout="centered"
)

# ─── Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@300;400;600&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    background-color: #010307 !important;
    color: #e0e6f0 !important;
    font-family: 'IBM Plex Sans Arabic', sans-serif !important;
}

.stApp { background-color: #010307 !important; }

/* Status banner */
.status-ok {
    background: linear-gradient(90deg, #001a0d 0%, #003320 100%);
    border: 1px solid #00ff88;
    border-left: 4px solid #00ff88;
    border-radius: 8px;
    padding: 14px 20px;
    font-family: 'JetBrains Mono', monospace;
    color: #00ff88;
    font-size: 14px;
    margin: 12px 0;
}

.status-fail {
    background: linear-gradient(90deg, #1a0005 0%, #330010 100%);
    border: 1px solid #ff0044;
    border-left: 4px solid #ff0044;
    border-radius: 8px;
    padding: 14px 20px;
    font-family: 'JetBrains Mono', monospace;
    color: #ff4466;
    font-size: 14px;
    margin: 12px 0;
}

.status-checking {
    background: linear-gradient(90deg, #0a0a00 0%, #1a1a00 100%);
    border: 1px solid #ffcc00;
    border-left: 4px solid #ffcc00;
    border-radius: 8px;
    padding: 14px 20px;
    font-family: 'JetBrains Mono', monospace;
    color: #ffcc00;
    font-size: 14px;
    margin: 12px 0;
}

/* Chat messages */
.msg-user {
    background: #0d1520;
    border: 1px solid #1e3050;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    color: #b0c8e8;
    font-size: 14px;
    text-align: right;
    direction: rtl;
}

.msg-ai {
    background: #0a0f1a;
    border: 1px solid #00ccff22;
    border-left: 3px solid #00ccff;
    border-radius: 4px 12px 12px 12px;
    padding: 12px 16px;
    margin: 8px 40px 8px 0;
    color: #c8dff0;
    font-size: 14px;
    direction: rtl;
}

.msg-model-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px;
    color: #00ccff88;
    margin-bottom: 6px;
    direction: ltr;
}

/* Title */
.main-title {
    font-size: 26px;
    font-weight: 600;
    color: #00ccff;
    letter-spacing: 1px;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 4px;
}

.sub-title {
    font-size: 13px;
    color: #556677;
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 24px;
}

/* Input overrides */
.stTextInput input, .stSelectbox select {
    background-color: #0a0f1a !important;
    border: 1px solid #1e3050 !important;
    color: #e0e6f0 !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stTextInput input:focus {
    border-color: #00ccff !important;
    box-shadow: 0 0 0 2px #00ccff22 !important;
}

/* Buttons */
.stButton > button {
    background: #00ccff11 !important;
    border: 1px solid #00ccff44 !important;
    color: #00ccff !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 13px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #00ccff22 !important;
    border-color: #00ccff !important;
    box-shadow: 0 0 12px #00ccff33 !important;
}

/* Divider */
hr { border-color: #1e3050 !important; }

/* Selectbox */
[data-baseweb="select"] > div {
    background-color: #0a0f1a !important;
    border-color: #1e3050 !important;
    color: #e0e6f0 !important;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── DeepSeek Models ─────────────────────────────────────────
DEEPSEEK_MODELS = {
    "deepseek/deepseek-chat-v3-0324": "DeepSeek Chat V3 (0324) — Latest",
    "deepseek/deepseek-r1": "DeepSeek R1 — Reasoning",
    "deepseek/deepseek-r1-zero": "DeepSeek R1 Zero",
    "deepseek/deepseek-r1-distill-llama-70b": "R1 Distill — Llama 70B",
    "deepseek/deepseek-r1-distill-llama-8b": "R1 Distill — Llama 8B",
    "deepseek/deepseek-r1-distill-qwen-32b": "R1 Distill — Qwen 32B",
    "deepseek/deepseek-r1-distill-qwen-14b": "R1 Distill — Qwen 14B",
    "deepseek/deepseek-r1-distill-qwen-7b": "R1 Distill — Qwen 7B",
    "deepseek/deepseek-r1-distill-qwen-1.5b": "R1 Distill — Qwen 1.5B",
    "deepseek/deepseek-prover-v2": "DeepSeek Prover V2",
}

# ─── Session State ───────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_status" not in st.session_state:
    st.session_state.api_status = None  # None / "ok" / "fail"
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "balance_info" not in st.session_state:
    st.session_state.balance_info = None

# ─── Functions ───────────────────────────────────────────────
def check_api_key(api_key: str) -> dict:
    """Test the API key by fetching account info"""
    try:
        r = requests.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10
        )
        if r.status_code == 200:
            return {"ok": True, "data": r.json()}
        else:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:200]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

def chat_with_model(api_key: str, model: str, messages: list) -> dict:
    """Send chat request to OpenRouter"""
    try:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 1000,
        }
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/openrouter-tester",
                "X-Title": "OpenRouter Key Tester"
            },
            json=payload,
            timeout=60
        )
        if r.status_code == 200:
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage", {})
            return {"ok": True, "content": content, "usage": usage}
        else:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ─── UI Layout ───────────────────────────────────────────────
st.markdown('<div class="main-title">⬡ OPENROUTER TESTER</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">DeepSeek Model Explorer · Key Validator</div>', unsafe_allow_html=True)

# ─── API Key Section ─────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    api_key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-or-v1-...",
        value=st.session_state.api_key,
        label_visibility="collapsed"
    )
with col2:
    check_btn = st.button("تحقق", use_container_width=True)

if check_btn and api_key_input:
    st.session_state.api_key = api_key_input
    st.markdown('<div class="status-checking">⏳ جاري التحقق من المفتاح...</div>', unsafe_allow_html=True)
    result = check_api_key(api_key_input)
    if result["ok"]:
        st.session_state.api_status = "ok"
        data = result["data"].get("data", {})
        st.session_state.balance_info = data
    else:
        st.session_state.api_status = "fail"
        st.session_state.balance_info = {"error": result["error"]}
    st.rerun()

# ─── Status Display ──────────────────────────────────────────
if st.session_state.api_status == "ok":
    info = st.session_state.balance_info or {}
    limit = info.get("limit", "—")
    usage = info.get("usage", "—")
    label = info.get("label", "—")
    
    if limit is not None and usage is not None:
        try:
            remaining = float(limit) - float(usage)
            remaining_str = f"${remaining:.4f}"
        except:
            remaining_str = "—"
    else:
        remaining_str = "—"

    st.markdown(f"""
    <div class="status-ok">
        🟢 &nbsp; المفتاح شغال — Key is ACTIVE<br>
        <span style="font-size:12px; color:#00cc66">
        Label: {label} &nbsp;|&nbsp; Remaining: {remaining_str}
        </span>
    </div>
    """, unsafe_allow_html=True)

elif st.session_state.api_status == "fail":
    err = ""
    if st.session_state.balance_info:
        err = st.session_state.balance_info.get("error", "")
    st.markdown(f"""
    <div class="status-fail">
        🔴 &nbsp; المفتاح ما خدامش — Key FAILED<br>
        <span style="font-size:11px; color:#ff6688">{err}</span>
    </div>
    """, unsafe_allow_html=True)

# ─── Model Selector ──────────────────────────────────────────
st.markdown("---")
selected_model_key = st.selectbox(
    "اختار الموديل",
    options=list(DEEPSEEK_MODELS.keys()),
    format_func=lambda x: DEEPSEEK_MODELS[x],
    index=0
)

# ─── Chat History ─────────────────────────────────────────────
if st.session_state.messages:
    st.markdown("---")
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f'<div class="msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            model_used = msg.get("model", "")
            st.markdown(f"""
            <div class="msg-ai">
                <div class="msg-model-tag">{model_used}</div>
                {msg["content"]}
            </div>
            """, unsafe_allow_html=True)

# ─── Chat Input ───────────────────────────────────────────────
st.markdown("---")
col_input, col_send = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        "رسالتك",
        placeholder="اكتب رسالتك هنا...",
        label_visibility="collapsed",
        key="chat_input"
    )
with col_send:
    send_btn = st.button("إرسال", use_container_width=True)

if st.session_state.api_status != "ok" and (send_btn or user_input):
    st.warning("⚠️ خاصك تحقق من المفتاح أولاً")

if send_btn and user_input and st.session_state.api_status == "ok":
    api_key = st.session_state.api_key
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Build messages list for API (only role + content)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
    
    with st.spinner("جاري الاتصال..."):
        result = chat_with_model(api_key, selected_model_key, api_messages)
    
    if result["ok"]:
        usage = result.get("usage", {})
        tokens_info = f"[tokens: {usage.get('total_tokens', '?')}]" if usage else ""
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["content"],
            "model": f"{selected_model_key} {tokens_info}"
        })
    else:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"❌ خطأ: {result['error']}",
            "model": selected_model_key
        })
    
    st.rerun()

# ─── Clear Chat ───────────────────────────────────────────────
if st.session_state.messages:
    if st.button("🗑️ مسح المحادثة", use_container_width=False):
        st.session_state.messages = []
        st.rerun()
