# AI Essay Detector

A sentence-level AI essay detection system that identifies measurable linguistic patterns associated with AI-generated writing and explains the evidence behind each prediction.



The system analyzes college-admission-style essays using **17 linguistic features** across perplexity, burstiness, vocabulary, writing style, and syntax. Instead of asking an LLM to make a subjective judgment, the extracted signals are passed to a calibrated machine-learning classifier that produces an AI probability for each sentence.

---

## Overview

AI-generated text can often differ from human writing in measurable ways such as:

* Language predictability
* Sentence-length consistency
* Vocabulary diversity
* Use of formulaic expressions
* Perplexity variation across a passage
* First-person usage
* Punctuation patterns
* Syntactic structure

This project combines these signals into an explainable detection pipeline.

The application works at the **sentence level**, allowing users to see which parts of an essay are more likely to contain AI-generated characteristics.

### Core idea

```text
Essay
  │
  ▼
Sentence Segmentation
  │
  ▼
17 Linguistic Features
  │
  ├── GPT-2 Perplexity
  ├── Burstiness
  ├── Vocabulary
  ├── Style
  └── Syntax
  │
  ▼
Feature Scaling
  │
  ▼
Calibrated Gradient-Boosted Classifier
  │
  ▼
AI Probability per Sentence
  │
  ▼
SHAP Explainability
  │
  ▼
Interactive Web Interface
```

---

## Features

### 1. Sentence-Level Detection

The essay is split into individual sentences and each sentence receives:

* AI probability
* Classification label
* Character position
* Supporting evidence
* SHAP attribution values

The three classification categories are:

| Probability | Label               |
| ----------- | ------------------- |
| `< 35%`     | Likely Human        |
| `35% – 64%` | Uncertain           |
| `≥ 65%`     | Likely AI-Generated |

These thresholds are intended for interpretation rather than proof of authorship.

---

### 2. 17 Linguistic Signals

The detector uses five groups of measurable features.

#### Perplexity

Generated using a locally executed GPT-2 model.

| Feature          | Description                           |
| ---------------- | ------------------------------------- |
| `mean_log_prob`  | Average token log probability         |
| `perplexity`     | Overall text predictability           |
| `min_token_prob` | Probability of the least likely token |

GPT-2 is used as a **measurement instrument**, not as the final decision-maker.

---

#### Burstiness

Measures variation across sentences.

| Feature                  | Description                               |
| ------------------------ | ----------------------------------------- |
| `perplexity_zscore`      | Sentence perplexity relative to the essay |
| `passage_perplexity_std` | Perplexity variation across the passage   |

Human writing can contain larger variations in complexity, while generated text may exhibit more uniform patterns.

---

#### Vocabulary

| Feature            | Description                            |
| ------------------ | -------------------------------------- |
| `type_token_ratio` | Vocabulary diversity                   |
| `rare_word_rate`   | Frequency of uncommon words            |
| `hedging_rate`     | Presence of common hedging expressions |
| `formulaic_count`  | Presence of formulaic phrases          |

---

#### Style

| Feature                  | Description                           |
| ------------------------ | ------------------------------------- |
| `sentence_length_tokens` | Number of tokens                      |
| `sentence_length_chars`  | Number of characters                  |
| `length_zscore`          | Sentence length relative to the essay |
| `first_person_density`   | Usage of first-person pronouns        |
| `punctuation_variety`    | Variety of punctuation                |

---

#### Syntax

Syntax features are extracted using spaCy.

| Feature              | Description                     |
| -------------------- | ------------------------------- |
| `pos_entropy`        | Distribution of parts of speech |
| `dep_tree_depth`     | Maximum dependency tree depth   |
| `avg_dep_tree_depth` | Average dependency tree depth   |

---

## Explainability

A major design goal of the project is **explainability**.

The detector does not simply return:

> "This sentence is AI."

Instead, it provides evidence explaining why the classifier reached its prediction.

The system uses **SHAP-based feature attribution** to identify the signals that contributed most strongly to each prediction.

Example:

```text
Sentence
   │
   ▼
AI Probability: 78%
   │
   ├── Perplexity → toward AI
   ├── Sentence Length → toward AI
   ├── Vocabulary Diversity → toward Human
   └── Formulaic Language → toward AI
```

The frontend visualizes these contributions using evidence cards and directional SHAP bars.

---

## Machine Learning Model

The final classifier is a:

**HistGradientBoostingClassifier**

wrapped using:

**CalibratedClassifierCV**

with sigmoid/Platt calibration.

### Why gradient boosting?

Gradient-boosted decision trees work well for this problem because the feature set contains:

* Different numerical scales
* Correlated linguistic signals
* Non-linear relationships
* Feature interactions

Calibration is used so that probability outputs are more meaningful than raw classifier scores.

---

## Dataset

The current prototype uses a small proof-of-concept dataset containing approximately **320 labeled sentences**.

| Category             | Approx. Count |
| -------------------- | ------------: |
| Human                |           160 |
| AI-generated         |           150 |
| AI-polished / Hybrid |            10 |
| **Total**            |       **320** |

The examples represent college-admission-style writing patterns.

The dataset is intentionally small and should **not** be considered representative of real-world writing populations.

No real student's essay is reproduced verbatim in the project dataset.

---

## Evaluation

The current held-out test set contains **48 sentences**.

| Metric          |    Result |
| --------------- | --------: |
| Accuracy        | **87.5%** |
| ROC-AUC         | **0.951** |
| Brier Score     | **0.098** |
| Human Precision | **0.875** |
| Human Recall    | **0.875** |
| AI Precision    | **0.875** |
| AI Recall       | **0.875** |

The Brier score indicates that the calibrated probabilities are substantially better than a random baseline.

However, these results are **in-distribution prototype results** and should not be interpreted as real-world production accuracy.

---

## Project Architecture

```text
                         ┌──────────────────────┐
                         │      User Essay      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Sentence Splitter    │
                         │ spaCy / NLTK /        │
                         │ fallback segmentation│
                         └──────────┬───────────┘
                                    │
                                    ▼
                    ┌──────────────────────────────┐
                    │      Feature Extraction     │
                    ├──────────────────────────────┤
                    │ GPT-2 Perplexity             │
                    │ Burstiness                   │
                    │ Vocabulary                   │
                    │ Style                        │
                    │ Syntax                       │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Feature Scaling               │
                    │ StandardScaler               │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Calibrated Gradient Boosting │
                    │ Classifier                   │
                    └──────────────┬───────────────┘
                                   │
                         AI probability
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ SHAP Explainability          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ FastAPI REST API              │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Interactive Web Frontend      │
                    └──────────────────────────────┘
```

---

## Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn
* Pydantic

### Machine Learning

* Scikit-learn
* NumPy
* Pandas
* Transformers
* PyTorch
* SHAP

### NLP

* spaCy
* NLTK
* GPT-2

### Frontend

* HTML5
* CSS3
* JavaScript

### Testing

* pytest
* HTTPX

---

## Project Structure

```text
ai-essay-detector/
│
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py
│   │
│   ├── models/
│   │   └── classifier.pkl
│   │
│   ├── pipeline/
│   │   ├── classifier.py
│   │   ├── explainer.py
│   │   ├── extractor.py
│   │   ├── splitter.py
│   │   │
│   │   └── features/
│   │       ├── burstiness.py
│   │       ├── perplexity.py
│   │       ├── style.py
│   │       ├── syntax.py
│   │       └── vocabulary.py
│   │
│   ├── tests/
│   │   ├── test_features.py
│   │   └── test_pipeline.py
│   │
│   ├── training/
│   │   ├── build_dataset.py
│   │   ├── evaluate.py
│   │   ├── seed_data.py
│   │   └── train.py
│   │
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── docs/
│   ├── methodology.md
│   ├── evaluation_report.md
│   └── limitations.md
│
├── frontend/
│   ├── index.html
│   ├── app.js
│   └── style.css
│
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/srimsarkar/AI-essay-detector.git
cd AI-essay-detector
```

---

## 2. Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

Install PyTorch separately if required:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

---

## 4. Install the spaCy model

```bash
python -m spacy download en_core_web_sm
```

The first time the application uses GPT-2, the required model files may also be downloaded by Hugging Face Transformers.

---

# Running the Application

From the project root:

```bash
cd backend
uvicorn main:app --reload
```

Then open:

```text
http://localhost:8000
```

The FastAPI backend serves the frontend directly.

---

# API

## Analyze an Essay

### Endpoint

```text
POST /api/analyze
```

### Request

```json
{
  "essay": "Paste your essay text here. The essay should contain at least 50 characters."
}
```

### Response

```json
{
  "essay": "...",
  "sentences": [
    {
      "sentence": "...",
      "start_char": 0,
      "end_char": 80,
      "ai_probability": 0.72,
      "label": "ai",
      "evidence": [
        {
          "signal": "perplexity",
          "value": 12.4,
          "explanation": "The sentence is relatively predictable.",
          "shap": 0.31
        }
      ]
    }
  ],
  "overall_ai_probability": 0.72,
  "overall_label": "ai",
  "model_version": "1.0"
}
```

---

## Health Check

```text
GET /health
```

Response:

```json
{
  "status": "ok"
}
```

---

# Training the Model

The repository contains scripts for rebuilding the feature dataset and classifier.

From the `backend` directory:

```bash
python -m training.build_dataset
```

Then:

```bash
python -m training.train
```

The trained model is saved to:

```text
backend/models/classifier.pkl
```

---

# Evaluation

Run:

```bash
cd backend
python -m training.evaluate
```

The evaluation script generates:

```text
docs/evaluation_report.md
```

The report contains:

* Classification metrics
* ROC-AUC
* Calibration results
* Feature importance
* Confidently incorrect predictions
* Failure analysis

---

# Testing

Run the complete test suite:

```bash
pytest
```

For a faster test run excluding GPT-2 perplexity tests:

```bash
pytest -k "not TestPerplexity"
```

---

# Design Principles

## No LLM Judge

The project intentionally does not ask ChatGPT, Claude, Gemini, or another LLM:

> "Is this essay AI-generated?"

Instead, GPT-2 is used only to calculate numerical language-model statistics.

The final decision is made by a trained machine-learning classifier.

---

## Explainability First

Every prediction can be connected to measurable linguistic signals.

This makes the system more transparent than a black-box binary detector.

---

## Calibrated Probabilities

The system reports probabilities instead of pretending that AI authorship can be determined with absolute certainty.

For example:

```text
AI Probability: 52%
```

should be interpreted as an uncertain result, not as proof that 52% of the essay was written by AI.

---

## Human-in-the-Loop

The detector is designed as a **decision-support system**, not an automated disciplinary or admissions decision-maker.

Human review remains essential.

---

# Limitations

The current prototype has several important limitations.

### Small Dataset

The training dataset contains only approximately 320 sentences.

A production system would require a much larger and more diverse corpus.

### Domain Dependence

The model was developed primarily around college-admission-style writing.

Performance may change substantially for:

* Academic papers
* Business documents
* Technical writing
* Creative writing
* Social media
* Non-native English writing

### ESL / Multilingual Bias Risk

The system has not been empirically validated on a dedicated English-as-a-Second-Language evaluation dataset.

Certain formal writing patterns may therefore be incorrectly associated with AI-generated text.

### Short Sentences

Very short sentences contain fewer measurable signals and may produce unstable predictions.

### Modern LLMs

GPT-2 is used as the perplexity measurement model. Its probability distribution may differ substantially from modern LLMs such as GPT-class or Claude-class models.

### AI-Polished Human Writing

Text that was originally written by a person and subsequently edited or polished using AI may fall into an ambiguous region.

### No Definitive Authorship Detection

The system cannot establish who wrote a piece of text.

It should not be treated as forensic proof of AI authorship.

---

# Ethical Use

This project should be used as an **assistive analytical tool**.

It should not be used as the sole basis for:

* Rejecting a student
* Penalizing academic work
* Making disciplinary decisions
* Determining scholarship eligibility
* Establishing academic misconduct

A probabilistic detector cannot reliably establish authorship.

Human judgment and contextual evidence should always be considered.

---

# Future Improvements

Potential future development includes:

* Larger and more diverse training datasets
* Human-written and AI-written paired essays
* Dedicated ESL evaluation datasets
* Cross-domain evaluation
* Modern language-model probability features
* Transformer-based embeddings
* Adversarial robustness testing
* Better hybrid human/AI classification
* Sentence and paragraph-level aggregation
* Model monitoring and drift detection
* Bias and fairness evaluation
* Cloud deployment
* Authentication and usage analytics

---

# Documentation

Additional technical documentation is available in:

```text
docs/methodology.md
docs/evaluation_report.md
docs/limitations.md
```

### Methodology

Explains the complete detection pipeline and 17 linguistic features.

### Evaluation Report

Contains detailed performance metrics and documented incorrect predictions.

### Limitations

Documents known failure modes, dataset limitations, ESL concerns, and appropriate use.

---

# Project Status

**Status:** Prototype / Hackathon Project

The current implementation is a research-oriented proof of concept rather than a production-grade authorship verification system.

---

# Author

**Sreyanka Sarkar**

Computer Science & Engineering

GitHub: `@srimsarkar`

---

## License

This project can be licensed according to the requirements of the repository owner or hackathon. Add an explicit license file before presenting the repository as an open-source project.
