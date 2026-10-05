# 🥗 Dr. Nutri — Production GenAI Engine

A production-grade, full-stack Generative AI application built with **LangChain**, **Streamlit**, and **OpenAI**. This project demonstrates enterprise design patterns for LLM systems, including strict **input/output guardrails**, **token-by-token real-time streaming**, **structured evaluation metrics**, and **resilient execution chains**.

---

## 📋 Table of Contents

* [Overview](#-overview)
* [Key Features](#-key-features)
* [System Architecture](#-system-architecture)
* [Tech Stack](#-tech-stack)
* [Getting Started](#-getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
  * [Environment Setup](#environment-setup)
* [Usage](#-usage)
* [Enterprise Patterns & Concepts](#-enterprise-patterns--concepts)
  * [1. Input & Output Guardrails](#1-input--output-guardrails)
  * [2. Token-by-Token Streaming](#2-token-by-token-streaming)
  * [3. Automated Response Evaluation](#3-automated-response-evaluation)
  * [4. Resilient LangChain Architecture](#4-resilient-langchain-architecture)
* [Project Structure](#-project-structure)
* [License](#-license)

---

## 🩺 Overview

**Dr. Nutri** transforms simple user prompts into tailored, evidence-based nutrition plans. Rather than operating as a raw LLM wrapper, this system demonstrates how to wrap foundation models in safety protocols, performance analytics, and dynamic user interfaces suited for enterprise deployment.

---

## ✨ Key Features

- **🛡️ Multi-Tier Guardrails:** Prevents prompt injections, off-topic requests, and unsafe medical claims using regex and semantic filtering.
- **⚡ Low-Latency Streaming:** Real-time token streaming yields instant visual feedback, minimizing Time to First Token (TTFT).
- **📊 Real-Time Evaluation Engine:** Measures output quality across structure adherence, formatting precision, word count, and latency ($s$).
- **🎨 Interactive UX:** Built with custom Streamlit styling mimicking a professional dietitian consultation interface.
- **⚙️ Configurable Parameters:** User-adjustable macro priorities (protein, fiber, sugar limits) and output controls.

---

## 🏗️ System Architecture

```
[ User Input ]
      │
      ▼
┌───────────────────────────────┐
│     Input Guardrail Check     │ ──(Fails)──► [ Return Error Message ]
└───────────────────────────────┘
      │ (Passes)
      ▼
┌───────────────────────────────┐
│     LangChain Execution       │
│  (ChatPromptTemplate + LLM)   │
└───────────────────────────────┘
      │
      ▼ (Chunk-by-Chunk Token Stream)
┌───────────────────────────────┐
│     Streamlit Real-Time UI    │ ◄── [ Live Cursor Rendering ]
└───────────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│    Output Guardrail Check     │ ──(Violations)──► [ Warning Badge ]
└───────────────────────────────┘
      │
      ▼
┌───────────────────────────────┐
│   Automated Eval Metrics      │ ──► [ Dashboard Score Cards ]
└───────────────────────────────┘
```

---

## 🛠️ Tech Stack

- **Core Framework:** [LangChain Core](https://python.langchain.com/) / `langchain-openai`
- **Foundation Model:** OpenAI `gpt-4o-mini`
- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Language:** Python 3.10+
- **Security & Quality:** Custom Guardrail Expressions & Automated Eval Engines

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher installed.
- An active **OpenAI API Key**.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/dr-nutri-genai.git
   cd dr-nutri-genai
   ```

2. **Create and Activate Virtual Environment:**
   ```bash
   # macOS / Linux
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies:**
   ```bash
   pip install streamlit langchain langchain-core langchain-openai
   ```

### Environment Setup

Set your OpenAI API Key as an environment variable:

```bash
# macOS / Linux
export OPENAI_API_KEY="your-actual-openai-api-key"

# Windows (PowerShell)
$env:OPENAI_API_KEY="your-actual-openai-api-key"
```

---

## 💻 Usage

Launch the Streamlit application locally:

```bash
streamlit run app.py
```

The application will automatically open in your browser at `http://localhost:8501`.

---

## 🧠 Enterprise Patterns & Concepts

### 1. Input & Output Guardrails

The application enforces strict validation rules on both inputs and outputs:

```python
def apply_input_guardrails(text: str) -> tuple[bool, str]:
    # 1. Reject empty inputs
    if not text.strip():
        return False, "Input cannot be empty."

    # 2. Block adversarial/jailbreak patterns
    jailbreak_patterns = [r"ignore previous instructions", r"system prompt", r"bypass"]
    for pattern in jailbreak_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False, "Input Guardrail: Adversarial prompt detected."

    return True, "Input passed guardrails."
```

### 2. Token-by-Token Streaming

By leveraging LangChain's `chain.stream()`, tokens are rendered immediately as they are generated:

```python
llm = ChatOpenAI(model="gpt-4o-mini", streaming=True)
chain = prompt | llm

stream_container = st.empty()
full_response = ""

for chunk in chain.stream({"count": 5, "preferences": "high protein"}):
    full_response += chunk.content
    stream_container.markdown(f"{full_response}▌")
```

### 3. Automated Response Evaluation

Evaluates outputs programmatically to measure compliance without requiring human-in-the-loop validation:

| Metric | Target | Description |
| :--- | :--- | :--- |
| **Structure Adherence** | 100% | Validates that the requested number of items was generated. |
| **Formatting Score** | 100% | Ensures titles match required markdown formatting (e.g., **Bold**). |
| **Latency** | $< 2.0\text{s}$ | Measures total response duration in seconds. |
| **Word Count** | Variable | Verifies verbosity and content density. |

### 4. Resilient LangChain Architecture

For enterprise scaling, execution chains can be upgraded using LangChain Expression Language (LCEL) features:

- **Retries (`with_retry`):** Recovers from transient network glitches.
- **Fallbacks (`with_fallbacks`):** Automatically routes requests to backup models during primary model outages.
- **Structured Outputs (`with_structured_output`):** Forces Pydantic schema adherence for downstream microservice integration.

---

## 📂 Project Structure

```text
dr-nutri-genai/
├── app.py              # Main Streamlit UI & LangChain pipeline
├── requirements.txt    # Python dependencies
├── README.md           # Documentation
└── .env.example        # Environment variable template
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.