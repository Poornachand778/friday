"""
Friday AI Streamlit Testing App
===============================

Interactive chat interface for the Friday SageMaker endpoint with:
- Real-time endpoint status monitoring
- Telugu/English persona controls
- MCP screenplay tool bridging
- Token/inference metrics

Launch via:
    streamlit run src/testing/friday_streamlit_tester.py
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List

import sys
from pathlib import Path

import boto3
import streamlit as st
from botocore.config import Config
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# MCP import is optional - chat works without it
try:
    from mcp.scene_manager.server import SceneManagerMCPServer

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    SceneManagerMCPServer = None

# ---------------------------------------------------------------------------
# Streamlit + environment setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Friday AI - SageMaker Tester",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

SYSTEM_PROMPT = (
    "You are Friday, Poorna’s personal assistant.\n"
    "Voice: curious, witty, Telugu+English blend when natural, direct but warm.\n"
    "If the user switches to Telugu, match the language.\n"
    "Prefer concrete steps, examples, and film/cooking analogies.\n"
    "\n"
    'You may call tools using <tool_call name="...">{json}</tool_call>.\n'
    "Tools available: scene_search(query, top_k, project_slug), scene_get(scene_code, project_slug).\n"
    "Default project slug is 'aa-janta-naduma'. Always confirm tool results before answering.\n"
)

TELUGU_SYSTEM_PROMPT = (
    "నువ్వు Friday, Poorna యొక్క వ్యక్తిగత సహాయకురాలు.\n"
    "శబ్దం: కుతూహలంగా, చమత్కారంగా, అవసరమైతే తెలుగు+ఆంగ్లం మిక్స్.\n"
    "వాడుకరి తెలుగు లో మాట్లాడితే, నువ్వు కూడా bilingual గా సమాధానం ఇవ్వాలి.\n"
    "కాన్క్రీట్ స్టెప్స్, ఉదాహరణలు, సినిమా/వంట అనలజీలతో స్పష్టంగా సమాధానం ఇవ్వు.\n"
    "\n"
    'టూల్స్ వాడడానికి <tool_call name="...">{json}</tool_call> ఫార్మాట్ ఉపయోగించు.\n'
    "లభ్యమైన టూల్స్: scene_search(query, top_k, project_slug), scene_get(scene_code, project_slug).\n"
    "డీఫాల్ట్ ప్రాజెక్ట్ స్లగ్ 'aa-janta-naduma'. టూల్ ఫలితాలు చూసి తర్వాత సమాధానం ఇవ్వు.\n"
)

DEFAULT_PROJECT = os.getenv("DEFAULT_PROJECT", "aa-janta-naduma")
MCP_BRIDGE = (
    SceneManagerMCPServer(default_project=DEFAULT_PROJECT) if MCP_AVAILABLE else None
)
TOOL_CALL_RE = re.compile(
    r"<tool_call\\s+name=\\\"([^\\\"]+)\\\">(.*?)</tool_call>", re.DOTALL
)

LOGGER = logging.getLogger("friday_streamlit")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Custom styling
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
.main > div {
    padding-top: 2rem;
}
.stTextArea > div > div > textarea {
    font-family: 'Source Code Pro', monospace;
}
.metric-card {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.5rem 0;
}
.success-metric {
    background: linear-gradient(90deg, #56ab2f 0%, #a8e6cf 100%);
}
.error-metric {
    background: linear-gradient(90deg, #ff416c 0%, #ff4b2b 100%);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar: AWS configuration
# ---------------------------------------------------------------------------
st.sidebar.header("🔧 AWS Configuration")

default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
default_endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME", "friday-rt")

region = st.sidebar.text_input("AWS Region", value=default_region)
endpoint_name = st.sidebar.text_input("Endpoint Name", value=default_endpoint)


@st.cache_resource(show_spinner=False)
def get_runtime_client(region_name: str):
    return boto3.client(
        "sagemaker-runtime",
        region_name=region_name,
        config=Config(
            read_timeout=300,
            connect_timeout=20,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


@st.cache_resource(show_spinner=False)
def get_sagemaker_client(region_name: str):
    return boto3.client("sagemaker", region_name=region_name)


try:
    runtime_client = get_runtime_client(region)
    sagemaker_client = get_sagemaker_client(region)
    st.sidebar.success("✅ AWS clients initialized")
except Exception as exc:  # noqa: broad-except
    st.sidebar.error(f"❌ AWS client error: {exc}")
    st.stop()

# ---------------------------------------------------------------------------
# Endpoint status cards
# ---------------------------------------------------------------------------
st.header("📊 Endpoint Status")

try:
    endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_config = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_desc["EndpointConfigName"]
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = endpoint_desc["EndpointStatus"]
        status_color = "success-metric" if status == "InService" else "error-metric"
        st.markdown(
            f"""
            <div class=\"metric-card {status_color}\">\n            <h4>Status</h4><h2>{status}</h2>\n            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        instance_type = endpoint_config["ProductionVariants"][0]["InstanceType"]
        st.markdown(
            f"""
            <div class=\"metric-card\">\n            <h4>Instance Type</h4><h2>{instance_type}</h2>\n            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        current_instances = endpoint_desc["ProductionVariants"][0][
            "CurrentInstanceCount"
        ]
        desired_instances = endpoint_desc["ProductionVariants"][0][
            "DesiredInstanceCount"
        ]
        st.markdown(
            f"""
            <div class=\"metric-card\">\n            <h4>Instances</h4><h2>{current_instances}/{desired_instances}</h2>\n            </div>
            """,
            unsafe_allow_html=True,
        )

    with col4:
        created_time = endpoint_desc["CreationTime"].strftime("%Y-%m-%d")
        st.markdown(
            f"""
            <div class=\"metric-card\">\n            <h4>Created</h4><h2>{created_time}</h2>\n            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔍 Detailed Endpoint Info"):
        st.json(
            {
                "EndpointArn": endpoint_desc["EndpointArn"],
                "EndpointConfigName": endpoint_desc["EndpointConfigName"],
                "ProductionVariants": endpoint_desc["ProductionVariants"],
                "LastModifiedTime": str(endpoint_desc["LastModifiedTime"]),
            }
        )
except Exception as exc:  # noqa: broad-except
    st.error(f"❌ Could not describe endpoint '{endpoint_name}': {exc}")
    st.info("💡 Verify the endpoint name, region, and AWS credentials.")

# ---------------------------------------------------------------------------
# Generation parameters + controls
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Generation Parameters")

max_new_tokens = st.sidebar.slider("Max New Tokens", 8, 1024, 256, step=8)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.7, step=0.05)
top_p = st.sidebar.slider("Top-p", 0.1, 1.0, 0.9, step=0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty", 1.0, 2.0, 1.05, step=0.01)

telugu_mode = st.sidebar.checkbox("Reply in Telugu (తెలుగు)", value=False)
stop_sequences = st.sidebar.text_input("Stop Sequences (comma separated)", value="")
stops = (
    [s.strip() for s in stop_sequences.split(",") if s.strip()]
    if stop_sequences
    else []
)

st.sidebar.subheader("🧭 Conversation Controls")
if st.sidebar.button("🔄 Reset Conversation", use_container_width=True):
    st.session_state["conversation"] = [{"role": "system", "content": SYSTEM_PROMPT}]
    st.session_state["last_usage"] = None
    st.session_state["last_time_ms"] = None
    st.rerun()

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def prepare_messages_for_endpoint(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    prepared: List[Dict[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content") or ""
        if role == "tool":
            name = msg.get("name", "tool")
            content = f"[tool:{name}] {content}".strip()
            role = "user"
        prepared.append({"role": role, "content": content})
    return prepared


def call_endpoint(
    messages: List[Dict[str, Any]], params: Dict[str, Any]
) -> Dict[str, Any]:
    payload = {
        "messages": prepare_messages_for_endpoint(messages),
        "parameters": params,
    }
    start = time.perf_counter()
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)
    body = response["Body"].read().decode("utf-8", errors="ignore")
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        data = {"generated_text": body}
    data.setdefault("usage", {})
    data["_inference_time"] = elapsed_ms
    return data


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for match in TOOL_CALL_RE.finditer(text):
        name = match.group(1)
        raw_args = match.group(2).strip()
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            LOGGER.warning("Failed to parse tool arguments for %s: %s", name, raw_args)
            continue
        calls.append({"name": name, "arguments": arguments})
    return calls


def clean_tool_markup(text: str) -> str:
    return TOOL_CALL_RE.sub("", text).strip()


def execute_tool_call(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if MCP_BRIDGE is None:
        return {"error": "MCP tools not available - scene_manager module not loaded"}
    try:
        wrapper = MCP_BRIDGE._dispatch_tool({"name": name, "arguments": arguments})
        return wrapper.get("content")
    except Exception as exc:  # noqa: broad-except
        LOGGER.exception("Tool call failed: %s", name, exc_info=exc)
        return {"error": str(exc)}


def run_friday_turn(params: Dict[str, Any]):
    conversation = st.session_state["conversation"]
    final_text = ""
    usage = None
    inference_time = None

    for _ in range(4):  # Prevent infinite loops
        response = call_endpoint(conversation, params)
        inference_time = response.get("_inference_time")
        usage = response.get("usage")
        assistant_content = response.get("generated_text", "")
        conversation.append({"role": "assistant", "content": assistant_content})

        tool_calls = parse_tool_calls(assistant_content)
        if not tool_calls:
            final_text = clean_tool_markup(assistant_content)
            break

        for call in tool_calls:
            tool_result = execute_tool_call(call["name"], call["arguments"])
            conversation.append(
                {
                    "role": "tool",
                    "name": call["name"],
                    "content": json.dumps(tool_result, ensure_ascii=False, indent=2),
                }
            )
    else:
        final_text = clean_tool_markup(assistant_content)

    return final_text, usage, inference_time


def render_conversation() -> None:
    for message in st.session_state["conversation"][1:]:
        role = message.get("role")
        if role == "user":
            with st.chat_message("user"):
                st.markdown(message.get("content", ""))
        elif role == "assistant":
            with st.chat_message("assistant"):
                content = clean_tool_markup(message.get("content", "")) or "_No reply_"
                st.markdown(content)
        elif role == "tool":
            with st.chat_message("assistant"):
                st.markdown(f"🛠️ **{message.get('name', 'tool')}** result:")
                st.code(message.get("content", ""), language="json")


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "conversation" not in st.session_state:
    st.session_state["conversation"] = [{"role": "system", "content": SYSTEM_PROMPT}]
if "last_usage" not in st.session_state:
    st.session_state["last_usage"] = None
if "last_time_ms" not in st.session_state:
    st.session_state["last_time_ms"] = None

st.session_state["conversation"][0]["content"] = (
    TELUGU_SYSTEM_PROMPT if telugu_mode else SYSTEM_PROMPT
)

generation_params: Dict[str, Any] = {
    "max_new_tokens": int(max_new_tokens),
    "temperature": float(temperature),
    "top_p": float(top_p),
}
if repetition_penalty != 1.0:
    generation_params["repetition_penalty"] = float(repetition_penalty)
if stops:
    generation_params["stop"] = stops

# ---------------------------------------------------------------------------
# Chat interface
# ---------------------------------------------------------------------------
st.header("💬 Chat with Friday AI")

render_conversation()

if st.session_state.get("last_usage"):
    st.sidebar.subheader("📊 Last Response Usage")
    usage = st.session_state["last_usage"] or {}
    st.sidebar.write(f"Prompt tokens: {usage.get('prompt_tokens', '-')}")
    st.sidebar.write(f"Completion tokens: {usage.get('completion_tokens', '-')}")
    st.sidebar.write(f"Total tokens: {usage.get('total_tokens', '-')}")
    if st.session_state.get("last_time_ms") is not None:
        st.sidebar.write(f"Inference time: {st.session_state['last_time_ms']:.0f} ms")

user_input = st.chat_input("Ask Friday anything…")

if user_input:
    st.session_state["conversation"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        with st.spinner("🤖 Friday is thinking..."):
            reply_text, usage, inference_time = run_friday_turn(generation_params)

        st.session_state["last_usage"] = usage
        st.session_state["last_time_ms"] = inference_time

        with st.chat_message("assistant"):
            st.markdown(reply_text or "_No reply_")

        st.rerun()
    except Exception as exc:  # noqa: broad-except
        LOGGER.exception("Friday request failed", exc_info=exc)
        error_msg = f"⚠️ Request failed: {exc}"
        st.session_state["conversation"].append(
            {"role": "assistant", "content": error_msg}
        )
        with st.chat_message("assistant"):
            st.error(error_msg)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    🎭 Friday AI SageMaker Tester · Streaming from SageMaker endpoint
</div>
""",
    unsafe_allow_html=True,
)
