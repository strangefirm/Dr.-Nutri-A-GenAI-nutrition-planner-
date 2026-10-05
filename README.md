Dr. Nutri
A GenAI nutrition planner with guardrails, live streaming and automated evaluation

Overview
Dr. Nutri turns a short user request into a tailored, evidence-informed nutrition plan. It is more than a thin wrapper around an LLM: the model call is surrounded by input checks, output checks, real-time streaming and automatic quality scoring, in the way a production LLM service would be.

The project is a practical example of the design patterns used to make LLM applications safer, faster and measurable.

Key Features
Feature	What it does
Input guardrails	Rejects empty input and blocks known jailbreak and prompt-injection patterns before the model is called
Output guardrails	Checks the generated plan and shows a warning badge if a rule is violated
Live streaming	Streams tokens as they are generated, so the user sees text immediately (low time to first token)
Automated evaluation	Scores each response on structure, formatting, word count and latency, shown as dashboard cards
Adjustable preferences	User-controlled priorities such as protein, fiber and sugar limits
Custom interface	Streamlit styling designed to feel like a dietitian consultation
System Architecture
 User input
     │
     ▼
 ┌─────────────────────────┐
 │  Input guardrail check  │ ── fails ──► Error message
 └─────────────────────────┘
     │ passes
     ▼
 ┌─────────────────────────┐
 │   LangChain execution   │
 │ (ChatPromptTemplate+LLM)│
 └─────────────────────────┘
     │ token stream
     ▼
 ┌─────────────────────────┐
 │  Streamlit live output  │ ◄── live cursor rendering
 └─────────────────────────┘
     │
     ▼
 ┌─────────────────────────┐
 │  Output guardrail check │ ── violation ──► Warning badge
 └─────────────────────────┘
     │
     ▼
 ┌─────────────────────────┐
 │  Automated evaluation   │ ──► Score cards
 └─────────────────────────┘
Tech Stack
Orchestration: LangChain (langchain-core, langchain-openai)
Model: OpenAI gpt-4o-mini
Interface: Streamlit
Language: Python 3.10+
Getting Started
Prerequisites
Python 3.10 or higher
An OpenAI API key
Installation
bash
# 1. Clone the repository
git clone https://github.com/<your-username>/dr-nutri-genai.git
cd dr-nutri-genai

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate            # Windows PowerShell: venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

requirements.txt:

streamlit
langchain
langchain-core
langchain-openai
Configuration

Set your OpenAI API key as an environment variable:

bash
export OPENAI_API_KEY="your-openai-api-key"          # macOS / Linux
$env:OPENAI_API_KEY="your-openai-api-key"            # Windows PowerShell

Never commit your key to the repository.

Run
bash
streamlit run app.py

The app opens at http://localhost:8501.

How It Works
1. Input and output guardrails

Every request is validated before it reaches the model.

python
def apply_input_guardrails(text: str) -> tuple[bool, str]:
    # 1. Reject empty inputs
    if not text.strip():
        return False, "Input cannot be empty."

    # 2. Block adversarial / jailbreak patterns
    jailbreak_patterns = [r"ignore previous instructions", r"system prompt", r"bypass"]
    for pattern in jailbreak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "Input Guardrail: Adversarial prompt detected."

    return True, "Input passed guardrails."
2. Token-by-token streaming

LangChain's chain.stream() renders tokens as they arrive:

python
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | llm

stream_container = st.empty()
full_response = ""

for chunk in chain.stream({"count": 5, "preferences": "high protein"}):
    full_response += chunk.content
    stream_container.markdown(f"{full_response}▌")
3. Automated response evaluation

Each response is scored in code, so quality can be checked without a human reviewer.

Metric	Target	Description
Structure adherence	100%	The requested number of items was generated
Formatting score	100%	Titles follow the required Markdown format (for example, bold)
Latency	under 2.0 s	Total response time in seconds
Word count	Variable	Indicates verbosity and content density
Extending the Project

The chain can be made more robust with LangChain Expression Language (LCEL) features:

with_retry: recover from temporary network errors.
with_fallbacks: switch to a backup model if the primary model is unavailable.
with_structured_output: enforce a Pydantic schema for downstream services.
Project Structure
text
dr-nutri-genai/
├── app.py              # Streamlit UI and LangChain pipeline
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
├── .env.example        # Environment variable template
└── docs/
    └── screenshot.png  # Screenshot used in this README
Disclaimer

Dr. Nutri provides general nutrition information for learning and demonstration purposes. It is not medical advice and does not replace a registered dietitian or doctor. People with medical conditions, allergies or special dietary needs should consult a qualified professional.

Limitations
Guardrails are rule-based, so they can miss new or reworded attacks. They reduce risk but do not remove it.
Evaluation metrics check format and speed, not whether the nutrition content is correct.
Latency depends on the model, the network and the length of the request.
License

Distributed under the MIT License. See LICENSE for details.
