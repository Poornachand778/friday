"""
Friday AI Streamlit Testing App
===============================

A comprehensive Streamlit app to test the Friday AI SageMaker endpoint with:
- Real-time endpoint status monitoring
- Telugu and English support
- Context-aware prompting
- Performance metrics
- Chat history
- File upload for context

Usage:
    streamlit run src/testing/friday_streamlit_tester.py

Environment Variables:
    - AWS_DEFAULT_REGION: AWS region (default: us-east-1)
    - HF_TOKEN: HuggingFace token
    - SAGEMAKER_ENDPOINT_NAME: Endpoint name (default: friday-rt)
"""

import os
import json
import time
import streamlit as st
import boto3
from botocore.config import Config
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Friday AI - SageMaker Tester",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better UI
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

# --- Sidebar: AWS Configuration ---
st.sidebar.header("🔧 AWS Configuration")

# Load defaults from environment
default_region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
default_endpoint = os.getenv("SAGEMAKER_ENDPOINT_NAME", "friday-rt")
hf_token = os.getenv("HF_TOKEN", "")

region = st.sidebar.text_input("AWS Region", value=default_region)
endpoint_name = st.sidebar.text_input("Endpoint Name", value=default_endpoint)


# AWS Client Configuration
@st.cache_resource(show_spinner=False)
def get_runtime_client(region_name: str):
    """Create SageMaker Runtime client with optimized timeouts"""
    return boto3.client(
        "sagemaker-runtime",
        region_name=region_name,
        config=Config(
            read_timeout=300,  # 5 minutes for first inference
            connect_timeout=20,
            retries={"max_attempts": 3, "mode": "standard"},
        ),
    )


@st.cache_resource(show_spinner=False)
def get_sagemaker_client(region_name: str):
    """Create SageMaker client for endpoint management"""
    return boto3.client("sagemaker", region_name=region_name)


# Initialize clients
try:
    runtime_client = get_runtime_client(region)
    sagemaker_client = get_sagemaker_client(region)
    st.sidebar.success("✅ AWS clients initialized")
except Exception as e:
    st.sidebar.error(f"❌ AWS client error: {e}")
    st.stop()

# --- Header ---
st.title("🎭 Friday AI — Real-time Endpoint Tester")
st.markdown(
    """
Test your Friday AI SageMaker endpoint with support for:
- **Telugu & English** conversations
- **Context-aware** prompting  
- **File uploads** for additional context
- **Real-time** performance monitoring
"""
)

# --- Endpoint Status Dashboard ---
st.header("📊 Endpoint Status")

try:
    endpoint_desc = sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_config = sagemaker_client.describe_endpoint_config(
        EndpointConfigName=endpoint_desc["EndpointConfigName"]
    )

    # Status metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status = endpoint_desc["EndpointStatus"]
        status_color = "success-metric" if status == "InService" else "error-metric"
        st.markdown(
            f"""
        <div class="metric-card {status_color}">
            <h4>Status</h4>
            <h2>{status}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        instance_type = endpoint_config["ProductionVariants"][0]["InstanceType"]
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>Instance Type</h4>
            <h2>{instance_type}</h2>
        </div>
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
        <div class="metric-card">
            <h4>Instances</h4>
            <h2>{current_instances}/{desired_instances}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        created_time = endpoint_desc["CreationTime"].strftime("%Y-%m-%d")
        st.markdown(
            f"""
        <div class="metric-card">
            <h4>Created</h4>
            <h2>{created_time}</h2>
        </div>
        """,
            unsafe_allow_html=True,
        )

    # Detailed info in expander
    with st.expander("🔍 Detailed Endpoint Info"):
        st.json(
            {
                "EndpointArn": endpoint_desc["EndpointArn"],
                "EndpointConfigName": endpoint_desc["EndpointConfigName"],
                "ProductionVariants": endpoint_desc["ProductionVariants"],
                "LastModifiedTime": str(endpoint_desc["LastModifiedTime"]),
            }
        )

except Exception as e:
    st.error(f"❌ Could not describe endpoint '{endpoint_name}': {e}")
    st.info(
        "💡 Make sure the endpoint exists and you have proper AWS credentials configured"
    )

# --- Generation Parameters ---
st.sidebar.header("⚙️ Generation Parameters")

max_new_tokens = st.sidebar.slider(
    "Max New Tokens", 8, 1024, 256, step=8, help="Maximum number of tokens to generate"
)
temperature = st.sidebar.slider(
    "Temperature",
    0.0,
    2.0,
    0.7,
    step=0.05,
    help="Controls randomness. Higher = more creative",
)
top_p = st.sidebar.slider(
    "Top-p (Nucleus Sampling)",
    0.1,
    1.0,
    0.9,
    step=0.05,
    help="Cumulative probability cutoff",
)
repetition_penalty = st.sidebar.slider(
    "Repetition Penalty", 1.0, 2.0, 1.05, step=0.01, help="Penalty for repeating tokens"
)

st.sidebar.subheader("🌐 Language Options")
telugu_mode = st.sidebar.checkbox(
    "Reply in Telugu (తెలుగు)", value=False, help="Force Telugu responses"
)

st.sidebar.subheader("🛑 Stop Sequences")
stop_sequences = st.sidebar.text_input(
    "Stop Sequences (comma-separated)",
    value="Human:,Assistant:,###",
    help="Sequences that will stop generation",
)

# --- Input Section ---
st.header("💬 Chat with Friday AI")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Your Question")
    user_prompt = st.text_area(
        "Ask anything...",
        height=150,
        placeholder="Ask Friday anything in English or Telugu...\n\nExample:\n- What is artificial intelligence?\n- కృత్రిమ మేధస్సు అంటే ఏమిటి?",
        help="Type your question or prompt here",
    )

with col2:
    st.subheader("Additional Context")
    context_text = st.text_area(
        "Optional context",
        height=150,
        placeholder="Paste relevant text, documents, or context here...",
        help="Provide additional context to improve responses",
    )

# File upload for context
uploaded_file = st.file_uploader(
    "📎 Upload a file for additional context",
    type=["txt", "md", "py", "json", "yaml", "yml"],
    help="Upload text files to provide additional context",
)

if uploaded_file is not None:
    try:
        file_content = uploaded_file.read().decode("utf-8", errors="ignore")
        context_text = (context_text + "\n\n" + file_content).strip()
        st.success(
            f"✅ Loaded {len(file_content)} characters from '{uploaded_file.name}'"
        )
    except Exception as e:
        st.warning(f"⚠️ Could not read uploaded file: {e}")


# --- Prompt Building ---
def build_friday_prompt(
    user_input: str, context: str = "", telugu: bool = False
) -> str:
    """
    Build Friday AI prompt following the established format
    """
    if telugu:
        system_msg = "నువ్వు Friday AI. తెలుగులో మాత్రమే సమాధానం ఇవ్వాలి. సంక్షిప్తంగా, స్పష్టంగా రాయండి."
    else:
        system_msg = "You are Friday, an AI assistant. Be helpful, concise, and accurate. Use the provided context when relevant."

    prompt_parts = [system_msg]

    if context and context.strip():
        prompt_parts.extend(["\n# Context:", context.strip(), ""])

    prompt_parts.extend(["\n# Question:", user_input.strip(), "\n# Answer:"])

    return "\n".join(prompt_parts)


# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "total_requests" not in st.session_state:
    st.session_state.total_requests = 0

if "total_time" not in st.session_state:
    st.session_state.total_time = 0

# --- Generate Response ---
generate_button = st.button(
    "🚀 Generate Response",
    type="primary",
    use_container_width=True,
    disabled=not user_prompt.strip(),
)

if generate_button:
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a question or prompt")
    else:
        # Build the prompt
        final_prompt = build_friday_prompt(user_prompt, context_text, telugu_mode)

        # Parse stop sequences
        stops = (
            [s.strip() for s in stop_sequences.split(",") if s.strip()]
            if stop_sequences
            else []
        )

        # Build payload
        payload = {
            "inputs": final_prompt,
            "parameters": {
                "max_new_tokens": int(max_new_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
            },
        }

        if stops:
            payload["parameters"]["stop"] = stops

        # Show payload in expander
        with st.expander("📋 Request Payload"):
            st.json(payload)

        # Make the request
        try:
            start_time = time.perf_counter()

            with st.spinner("🤖 Friday is thinking..."):
                response = runtime_client.invoke_endpoint(
                    EndpointName=endpoint_name,
                    ContentType="application/json",
                    Body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                )

            end_time = time.perf_counter()
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds

            # Parse response
            raw_response = response["Body"].read().decode("utf-8", errors="ignore")

            try:
                response_data = json.loads(raw_response)

                # Handle different response formats
                if isinstance(response_data, dict):
                    if "generated_text" in response_data:
                        generated_text = response_data["generated_text"]
                    else:
                        generated_text = str(response_data)
                elif isinstance(response_data, list) and response_data:
                    if (
                        isinstance(response_data[0], dict)
                        and "generated_text" in response_data[0]
                    ):
                        generated_text = response_data[0]["generated_text"]
                    else:
                        generated_text = str(response_data[0])
                else:
                    generated_text = raw_response
            except json.JSONDecodeError:
                generated_text = raw_response

            # Display response
            st.success(f"✅ Response generated in {response_time:.0f} ms")

            # Response container
            with st.container():
                st.subheader("🤖 Friday's Response:")
                st.markdown(
                    f"""
                <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 10px; margin: 1rem 0;">
                    {generated_text}
                </div>
                """,
                    unsafe_allow_html=True,
                )

            # Update session state
            st.session_state.chat_history.append(
                {
                    "timestamp": datetime.now(),
                    "question": user_prompt,
                    "context": context_text,
                    "response": generated_text,
                    "response_time": response_time,
                    "telugu_mode": telugu_mode,
                    "parameters": payload["parameters"],
                }
            )

            st.session_state.total_requests += 1
            st.session_state.total_time += response_time

            # Performance metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Response Time", f"{response_time:.0f} ms")
            with col2:
                st.metric("Total Requests", st.session_state.total_requests)
            with col3:
                avg_time = st.session_state.total_time / st.session_state.total_requests
                st.metric("Avg Response Time", f"{avg_time:.0f} ms")

        except Exception as e:
            st.error(f"❌ Request failed: {e}")
            st.info("💡 Check your endpoint status and AWS credentials")

            # Show error details in expander
            with st.expander("🔍 Error Details"):
                st.code(str(e))

# --- Chat History ---
if st.session_state.chat_history:
    st.header("📚 Chat History")

    # Clear history button
    if st.button("🗑️ Clear History", help="Clear all chat history"):
        st.session_state.chat_history = []
        st.session_state.total_requests = 0
        st.session_state.total_time = 0
        st.rerun()

    # Display history
    for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(
            f"💬 Conversation {i} - {chat['timestamp'].strftime('%H:%M:%S')}",
            expanded=False,
        ):
            col1, col2 = st.columns([3, 1])

            with col1:
                st.markdown("**👤 You asked:**")
                st.info(chat["question"])

                if chat["context"]:
                    st.markdown("**📄 Context provided:**")
                    st.text(
                        chat["context"][:200] + "..."
                        if len(chat["context"]) > 200
                        else chat["context"]
                    )

                st.markdown("**🤖 Friday replied:**")
                st.success(chat["response"])

            with col2:
                st.metric("Response Time", f"{chat['response_time']:.0f} ms")
                st.write(
                    f"**Language:** {'Telugu' if chat['telugu_mode'] else 'English'}"
                )
                st.write(f"**Temperature:** {chat['parameters']['temperature']}")
                st.write(f"**Max Tokens:** {chat['parameters']['max_new_tokens']}")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1rem;">
    🎭 Friday AI SageMaker Tester | Built with ❤️ and Streamlit
</div>
""",
    unsafe_allow_html=True,
)
