# 🧪 AI Misuse Triage Testing Guide

This project includes two types of tests: **Core Unit Tests** and **End-to-End (E2E) Automation Tests**.

---

## 1. Core Unit Tests (`pytest`)
These tests verify the core logic, schema validation, and environment initialization without requiring the Flask server to be running.

### Installation
First, ensure `pytest` is installed:
```bash
pip install -r requirements.txt
# or manually
pip install pytest
```

### Running the Tests
To run all unit tests:
```bash
pytest tests/test_core.py
```

---

## 2. E2E Automation Tests (`test_automation.py`)
These tests simulate a full interaction cycle with the Flask web server, validating the API endpoints, reward registration, and training log updates.

### Prerequisites
The Flask server **must** be running in the background:
```bash
python app.py
```

### Running the Tests
In a separate terminal, run:
```bash
python test_automation.py
```
This script will:
1.  Check if the server is reachable.
2.  Run evaluation and training suites.
3.  Validate the JSON schema of the API responses.
4.  Submit binary reward signals and verify the training logs.
5.  Provide a detailed summary report.

---

## 3. Manual Verification (Web UI)
You can also manually verify the app by visiting `http://127.0.0.1:7860` (or `5000` depending on your environment).
- Enter sample inputs.
- Click **"Get Decision"**.
- Provide feedback in **Training Mode**.
- Inspect the **Agent Memory** and **Training Log** tabs to see the reflection of your interactions.
