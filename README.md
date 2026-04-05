# AI Misuse Triage - OpenEnv RL Environment

An AI Misuse Triage environment designed for the OpenEnv Hackathon. This project implements a reinforcement-learning-style triage system that classifies AI assistant interactions for potential misuse (phishing, scams, prompt injection, etc.).

## 🚀 Overview

The system consists of:
-   **Triage Environment**: An OpenEnv-compliant environment for simulating misuse scenarios.
-   **Learning Agent**: A weight-based RL agent that improves its triage policy via user feedback.
-   **Web UI**: A Flask-based interface for interactive testing, evaluation, and training.
-   **Automated Evaluation**: A CLI interface (`inference.py --minimal`) for zero-manual-step grading.

## 🛠 Features

-   **Operating Modes**:
    -   `Evaluation`: Deterministic rule-based triage for stable baseline testing.
    -   `Training`: Interactive RL mode where users can provide reward signals (+1/-1) to update the agent's policy.
-   **Rich Aesthetics**: State-of-the-art Web UI with real-time status indicators and JSON inspection.
-   **Persistence**: Agent state is saved in `agent_memory.json` and episodic logs in `training_log.jsonl`.

## 📦 Local Setup

1.  **Clone the Repository**:
    ```bash
    git clone <repository-url>
    cd ai-misuse-triage
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Web UI**:
    ```bash
    python app.py
    ```
    Visit `http://127.0.0.1:5000` in your browser.

4.  **Run the Baseline Inference (Interactive)**:
    ```bash
    python inference.py
    ```

## 🐳 Docker Usage

To run the project in a containerized environment:

1.  **Build the Image**:
    ```bash
    docker build -t ai-misuse-triage .
    ```

2.  **Run the Web UI (Port 7860)**:
    ```bash
    docker run -p 7860:7860 ai-misuse-triage
    ```

3.  **Run Automated Evaluation**:
    ```bash
    docker run ai-misuse-triage python inference.py --minimal
    ```

## 🚀 Hugging Face Spaces Deployment

This project is ready for deployment on [Hugging Face Spaces](https://huggingface.co/spaces):

1.  Create a **New Space** using the **Docker** SDK.
2.  Choose the **Blank** template or link your GitHub repository.
3.  Ensure the Space is set to use the provided `Dockerfile`.
4.  The default entrypoint (`gunicorn`) will automatically start the UI on port `7860`.

## ✅ Evaluation Compliance

-   **Output Schema**: Every decision returns `risk_label`, `category`, `action`, and `rationale`.
-   **CLI Interface**: `python inference.py --minimal` outputs a single JSON object per episode for automated grading.
-   **Metadata**: Configuration is defined in `openenv.yaml`.

## 📂 Project Structure

```text
├── openenv_misuse_triage/ # Core Environment SDK
├── templates/             # UI Templates
├── static/                # UI Assets & JS
├── inference.py           # Baseline CLI Entrypoint
├── learning.py            # RL Agent Logic
├── app.py                 # Flask App
├── Dockerfile             # Container configuration
├── openenv.yaml           # OpenEnv metadata
├── requirements.txt       # Dependencies
└── README.md              # Documentation
```

---
*Created for the OpenEnv Hackathon — Round 1 Submission.*
