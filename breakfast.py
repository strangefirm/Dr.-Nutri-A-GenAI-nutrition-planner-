import os
import re
import time
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
# ------------------------------------------------------------------
# Page Setup & Styling
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NutriBot | Production GenAI Engine",
    page_icon="🥗",
    layout="centered",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f7faf7; }
    .header-card {
        background: linear-gradient(135deg, #1b5e20 0%, #4caf50 100%);
        padding: 1.8rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(27, 94, 32, 0.15);
    }
    .header-card h1 { color: white !important; margin-bottom: 0.3rem; }
    .chat-bubble-ai {
        background-color: #ffffff;
        border-left: 5px solid #2e7d32;
        padding: 1.2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    .eval-box {
        background-color: #f1f8e9;
        border: 1px solid #c8e6c9;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="header-card">
        <h1>🥗 Dr. Nutri GenAI Engine</h1>
        <p style="font-size: 1rem; margin: 0;">Production-grade architecture with Input Guardrails, Real-Time Streaming, and Automated Evaluation.</p>
    </div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------------
st.sidebar.subheader("Dietary Goals")
high_protein = st.sidebar.checkbox("High protein", value=False)
low_sugar = st.sidebar.checkbox("Low sugar", value=False)
high_fibre = st.sidebar.checkbox("High fibre", value=False)

enable_input_guard = True
enable_output_guard = True

# ------------------------------------------------------------------
# Guardrails Functions
# ------------------------------------------------------------------
def apply_input_guardrails(text: str) -> tuple[bool, str]:
    """Validates input for prompt injection, toxic keywords, or off-topic queries."""
    if not text.strip():
        return False, "Please enter a valid request."

    # Check for basic prompt injection or inappropriate commands
    jailbreak_patterns = [
        r"ignore previous instructions",
        r"system prompt",
        r"bypass",
        r"hack",
    ]
    for pattern in jailbreak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return (
                False,
                "⚠️ Input Guardrail Violation: Adversarial or jailbreak attempt detected.",
            )

    # Check off-topic queries (e.g., non-dietary technical/coding/political topics)
    off_topic_keywords = ["python script", "crypto", "election", "passwords"]
    for word in off_topic_keywords:
        if word in text.lower():
            return (
                False,
                f"⚠️ Input Guardrail Violation: Dr. Nutri only answers nutrition/breakfast queries. Query contained disallowed topic '{word}'.",
            )

    return True, "Input passed guardrails."


def apply_output_guardrails(
    response_text: str, expected_count: int
) -> tuple[bool, list[str]]:
    """Validates the output string against formatting constraints."""
    violations = []

    # Verify numbered items count matches expected count
    numbered_items = re.findall(
        r"^\d+\.\s", response_text, flags=re.MULTILINE
    )
    if len(numbered_items) != expected_count:
        violations.append(
            f"Expected {expected_count} numbered items, but found {len(numbered_items)}."
        )

    # Verify safety / medical disclaimer absence or unsafe medical claims
    unsafe_claims = ["cures disease", "guaranteed weight loss in 1 day"]
    for claim in unsafe_claims:
        if claim in response_text.lower():
            violations.append(
                f"Output contained unsafe medical claim: '{claim}'"
            )

    passed = len(violations) == 0
    return passed, violations


# ------------------------------------------------------------------
# Evaluation Function
# ------------------------------------------------------------------
def evaluate_response(
    response_text: str, expected_count: int, latency_sec: float
) -> dict:
    """Evaluates output quality, structural adherence, and execution metrics."""
    numbered_items = re.findall(
        r"^\d+\.\s", response_text, flags=re.MULTILINE
    )

    # Metric 1: Structural Score (0 - 100%)
    structure_score = 100 if len(numbered_items) == expected_count else 50

    # Metric 2: Bold title check
    bold_titles = re.findall(r"\*\*(.*?)\*\*", response_text)
    formatting_score = 100 if len(bold_titles) >= expected_count else 50

    # Metric 3: Total Word Count
    word_count = len(response_text.split())

    return {
        "Structure Adherence": f"{structure_score}%",
        "Formatting Score": f"{formatting_score}%",
        "Word Count": word_count,
        "Latency": f"{latency_sec:.2f}s",
    }


# ------------------------------------------------------------------
# Form Interface
# ------------------------------------------------------------------
with st.form("nutrition_consultation"):
    st.subheader("💬 Ask Dr. Nutri")

    col1, col2 = st.columns([2, 1])
    with col1:
        dietary_focus = st.text_input(
            "Dietary preferences/goals:",
            value="",
        )
    with col2:
        idea_count = st.select_slider(
            "Number of ideas:", options=[3, 5, 7], value=5
        )

    submit_button = st.form_submit_button("Generate Recommendations 🍳")

# ------------------------------------------------------------------
# Streaming Engine Execution
# ------------------------------------------------------------------
if submit_button:
    selected_goals = [
        goal
        for goal, selected in (
            ("High protein", high_protein),
            ("Low sugar", low_sugar),
            ("High fibre", high_fibre),
        )
        if selected
    ]
    dietary_preferences = ", ".join([*selected_goals, dietary_focus]).strip(", ")

    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if not dietary_preferences.strip():
        st.error("Please enter a dietary preference or select at least one dietary goal.")
    elif not api_key:
        st.error(
            "⚠️ Please supply an OpenAI API Key in the sidebar or environment."
        )
    else:
        # STEP 1: Input Guardrails Check
        input_passed = True
        if enable_input_guard:
            input_passed, guard_msg = apply_input_guardrails(dietary_preferences)

        if not input_passed:
            st.error(guard_msg)
        else:
            # Initialize LangChain Streaming Model
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.8,
                openai_api_key=api_key,
                streaming=True,  # Enables streaming chunks
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are Dr. Nutri, a warm nutritionist. Give whole-food breakfast ideas.",
                    ),
                    (
                        "user",
                        "Provide exactly {count} healthy breakfast options for: {preferences}.\n\n"
                        "Formatting rules:\n"
                        "- Greet the user in 1 short sentence.\n"
                        "- Numbered list (1 to {count}).\n"
                        "- Bold item title followed by a 1-sentence nutritional benefit.\n"
                        "- End with 1 short encouraging sentence.",
                    ),
                ]
            )

            chain = prompt | llm

            # STEP 2: Real-time Streaming Output
            st.subheader("⚡ Real-Time Stream Response")

            # Streamlit placeholder for live token updates
            stream_container = st.empty()
            full_response = ""
            raw_response = None

            start_time = time.time()

            # Stream chunks token-by-token
            for chunk in chain.stream(
                {"count": idea_count, "preferences": dietary_preferences}
            ):
                full_response += chunk.content
                raw_response = (
                    chunk if raw_response is None else raw_response + chunk
                )
                stream_container.markdown(
                    f'<div class="chat-bubble-ai">{full_response}▌</div>',
                    unsafe_allow_html=True,
                )

            latency = time.time() - start_time

            # Display final static text without the streaming cursor ▌
            stream_container.markdown(
                f'<div class="chat-bubble-ai">{full_response}</div>',
                unsafe_allow_html=True,
            )

            # STEP 3: Output Guardrail Check
            st.divider()
            if enable_output_guard:
                output_passed, violations = apply_output_guardrails(
                    full_response, idea_count
                )
                if output_passed:
                    st.success("✅ Output Guardrails: Passed all checks.")
                else:
                    st.warning(
                        f"⚠️ Output Guardrail Warning: {', '.join(violations)}"
                    )

            # STEP 4: Automated Response Evaluation
            st.subheader("📊 Automated Response Evaluation")
            eval_metrics = evaluate_response(full_response, idea_count, latency)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Structure", eval_metrics["Structure Adherence"])
            m2.metric("Formatting", eval_metrics["Formatting Score"])
            m3.metric("Word Count", eval_metrics["Word Count"])
            m4.metric("Latency", eval_metrics["Latency"])

            # Developer View
            with st.expander("🛠️ Developer Trace"):
                st.write("**Raw LLM Response:**")
                if raw_response is not None:
                    st.json(raw_response.model_dump(mode="json"))