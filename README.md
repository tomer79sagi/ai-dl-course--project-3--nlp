# Cyberbullying Detection (Project 3 — NLP)

## Project summary
This repository contains a Colab-first NLP pipeline for cyberbullying detection and *context-level* risk estimation.

## Background / problem statement
Social platforms often need to detect **group-level / crowded cyberbullying**: situations where many messages pile onto a target, often sharing similar semantics (e.g., repeated insults, coordinated harassment, raids).

Why this is hard in practice:
- **Per-message inference is expensive at scale**: running a transformer classifier on every single message in real time can be too slow/costly for large streams.
- **LLM-based moderation is typically even slower and more expensive**, and can be hard to operationalize under strict latency budgets.
- **Basic RAG is limited to similarity retrieval**: it can fetch “similar past messages”, but it does not directly output a reliable *risk score* for a *set* (crowd) of messages, nor does it explicitly capture *density* and *concentration* of harm.

What this project aims to do instead:
- Build a **fast inference system** that detects **dense and concentrated bullying**, where messages are **semantically similar** and collectively indicate escalation.
- Convert message-level signals into **set-level signals** (mean harm + concentration/burstiness), enabling real-time “situation awareness” over a rolling context.

The project builds:
- A **message-level binary classifier** (DistilBERT) for `bully` vs `normal`.
- A **confidence-augmented dataset** where each message is assigned a bullying probability.
- **Semantic embeddings** (MiniLM) for downstream clustering / retrieval.
- **set-based models** (Deep Sets, Set Transformer) that score *clusters / windows* of messages (i.e., “how risky is this conversation slice?”), emphasizing **density and concentration** rather than only single-message classification.

## Team members
- **[TODO]** Tomer Sagi.

## Repository contents (what to run)
This repo is notebook-based:
- `Cyberbullying_DistilBERT.ipynb`
  - Fine-tunes DistilBERT on the dataset and saves a model.
  - Produces a processed parquet with `bullying_confidence`.
- `Cyberbullying_MiniLM_Embeddings.ipynb`
  - Encodes all messages with MiniLM, saves embeddings, and creates “strong bully / strong safe / ambiguous” masks.
- `Cyberbullying_Cluster_Level_Bullying_Detection_(Set_Based_Model).ipynb`
  - Deep Sets baseline for **cluster-level bullying intensity**.
- `Cyberbullying_Set_Transformer.ipynb`
  - Set Transformer regressor for **cluster-level bullying intensity**.
- `Cyberbullying_kNN.ipynb`
  - kNN-based cluster sampling and a dual-head Set Transformer variant.
- `Cyberbullying_Inference_NO_TIME_AWARE.ipynb`
  - Streaming-style inference over a rolling context buffer (context set), producing alerts and qualitative explanations.

> Important: The notebooks assume Google Drive paths such as `/content/drive/MyDrive/grunitech-project3-cyberbullying` for artifacts.

---

## Dataset
### Source
- Hugging Face Datasets: `cike-dev/cyberbullying_dataset`

### Fields
- `cleaned_text`: normalized message text.
- `label`: `bully` / `normal`.

### Splits and balance (as used in notebooks)
From the notebook exploration:
- **Train**: 30,240 rows
- **Validation**: 3,780 rows
- **Test**: 3,780 rows
- **Class balance (train)**: 50% `bully`, 50% `normal`

### Notes / risks
- The dataset is perfectly balanced, which is convenient for training and reporting metrics but may not reflect real-world prevalence.
- Labels are binary; they do not separate harassment types (threats, identity attacks, profanity-only, sarcasm, etc.).

### Inference simulation dataset (2nd dataset)
The inference notebook also uses a second dataset to simulate *live* social-media streams:
- Hugging Face Datasets: `civil_comments`

How it is used in `Cyberbullying_Inference_NO_TIME_AWARE.ipynb`:
- A small slice is loaded (e.g., `train[:5000]`).
- The notebook uses:
  - `text` as message content.
  - `toxicity` as a **proxy ground-truth** signal to evaluate alert behavior.
- Synthetic `conv_id` values and arrival timestamps (`ts`) are generated to emulate multiple concurrent threads.

This dataset is **not** used to train the cyberbullying classifier; it is used to stress-test the *group-level detection / alerting loop* under realistic stream conditions.

---

## Models & methods

### 1) Message-level classifier: DistilBERT
Notebook: `Cyberbullying_DistilBERT.ipynb`

- Model: `distilbert-base-uncased` with a 2-class classification head.
- Training:
  - `num_train_epochs=3`
  - `learning_rate=2e-5`
  - `per_device_train_batch_size=16`
  - `weight_decay=0.01`
- Metrics: accuracy + F1.

**Confidence augmentation**
After training, the notebook runs batched inference and computes:
- `bullying_confidence = P(label=bully)`

It then exports a parquet:
- `messages_with_confidence.parquet` (saved to Google Drive under `data/processed/`)

### 2) Embeddings: MiniLM
Notebook: `Cyberbullying_MiniLM_Embeddings.ipynb`

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Output:
  - `minilm_embeddings.npy` with shape `(34020, 384)`
  - `labels.npy`
  - `confidences.npy`

The notebook also defines confidence-based masks:
- **Strong bully**: `bullying_confidence > 0.9` (10,619 samples)
- **Strong safe**: `bullying_confidence < 0.1` (5,669 samples)
- **Ambiguous**: the rest (17,732 samples)

### 3) Cluster-level / set-based modeling
Goal: Given a *set* of message embeddings (a cluster or a context window), output a continuous risk score.

#### 3.1 Deep Sets baseline
Notebook: `Cyberbullying_Cluster_Level_Bullying_Detection_(Set_Based_Model).ipynb`

- Architecture: `phi(·)` per element + mean pooling + `rho(·)`.
- Target label: `mean(bullying_confidence)` within the sampled set.

#### 3.2 Set Transformer
Notebook: `Cyberbullying_Set_Transformer.ipynb`

- Architecture: ISAB → ISAB → PMA pooling → MLP head.
- Target label: `mean(bullying_confidence)` within the sampled set.

#### 3.3 kNN cluster sampler + dual-head set model
Notebook: `Cyberbullying_kNN.ipynb`

- Uses a kNN index over embedding space to construct “mostly-local + some-random” clusters.
- Trains a dual-head Set Transformer:
  - **Mean head**: overall intensity (`mean(confidence)`).
  - **Top-k / concentration head**: concentration-style target (implemented as Gini in later iterations).

### 4) Inference (streaming-style, no time-awareness)
Notebook: `Cyberbullying_Inference_NO_TIME_AWARE.ipynb`

- Maintains a rolling buffer (a *context set*) of the last `CTX_MAXLEN` messages.
- Runs the set model on the buffer.
- Computes explainability stats (context size, density, toxic fraction, top contributors).
- Produces alerts using percentile thresholds over rolling signals.

**Simulating raids (`inject_raid`)**
To evaluate detection of crowded/bursty harassment, the notebook defines an `inject_raid(...)` function that:
- Selects a high-toxicity message template from `civil_comments` (by `toxicity` threshold).
- Inserts a burst of near-duplicate toxic messages into the stream (controlled by `raid_start_idx`, `raid_size`, and inter-message spacing `dt`).
- Assigns the injected burst to a dedicated `conv_id` (e.g., `999`) to emulate a coordinated “raid”.

This helps validate whether the system detects **dense and semantically consistent bullying** without relying on expensive per-message LLM reasoning.

---

## LoRA, RAG pipeline, and “agentic” structure
Your project brief mentions **LoRA**, **RAG**, and an **agentic structure**.

**Status in this repository (important for correctness):**
- **LoRA**: not implemented in the notebooks here (training is full fine-tuning via Hugging Face `Trainer`).
- **RAG pipeline**: not implemented (no document store / retrieval index used for generation). More importantly, **this project is not trying to solve the problem with similarity retrieval alone**; it focuses on **fast discriminative scoring** of *sets* of semantically related messages.
- **Agentic structure**: not implemented as an LLM agent loop; however, the inference notebook does implement a *rule-based monitoring loop* (rolling context, anomaly signals, alerts, and explanations), which is “agent-like” in the sense of continuous perception → scoring → action (alert).

**How you would add them (recommended future work):**
- **LoRA (PEFT)**
  - Replace full fine-tuning with LoRA adapters using `peft`.
  - Benefits: faster training, lower GPU memory, easier domain adaptation.
- **RAG for moderation support**
  - Retrieve similar historical messages / policy snippets.
  - Use retrieved evidence to justify decisions and reduce unsupported claims.
- **Agentic moderation loop**
  - Add a policy layer that decides actions: log, warn user, rate-limit, escalate to human.
  - Add tool calls: fetch conversation history, retrieve similar incidents, generate a structured report.

---

## Results

### DistilBERT (message-level)
From `Cyberbullying_DistilBERT.ipynb`:
- `eval_accuracy`: **0.7886**
- `eval_f1`: **0.7972**

Classification report shown in the notebook (validation):
- **Accuracy**: **0.8061**
- `normal` F1: **0.7955**
- `bully` F1: **0.8156**

Artifacts/plots generated in-notebook:
- Label distribution
- Train vs validation loss curve
- Confusion matrix

### MiniLM embedding space sanity checks
From `Cyberbullying_MiniLM_Embeddings.ipynb`:
- A simple 2-means clustering produced a **low silhouette score (~0.032)**, indicating that bully vs normal is not cleanly separable in embedding space with this simplistic clustering setup.
- UMAP visualization is generated and colored by `bullying_confidence`.

### Cluster-level models
From the cluster/set notebooks:
- Deep Sets and Set Transformer both produce monotonic responses to synthetic cluster composition.
- The kNN + dual-head experiment reports relatively good rank correlation (Spearman) but also shows **prediction collapse** (near-constant outputs), suggesting a mismatch between target definition, loss, or model capacity/regularization.

---

## Qualitative examples

### Message-level confidence example
From `Cyberbullying_DistilBERT.ipynb`:
- Input: `you are such an idiot`
  - Prediction: `bully` with bullying confidence ≈ **0.945**
- Input: `nice photo!`
  - Prediction: `normal` with bullying confidence ≈ **0.046**

### Context-level “alert” explanation
From `Cyberbullying_Inference_NO_TIME_AWARE.ipynb`, the monitoring loop emits explanations like:
- `ctx=5 msgs / 7.2s | density=0.69 msg/s | 20% toxic | meanT=0.18 peakT=0.89`

It also prints the top context contributors (message id, toxicity, snippet) to help debug *why* an alert fired.

---

## Setup instructions

### Recommended: run in Google Colab (as authored)
These notebooks are written for Colab and Google Drive.

- **Hugging Face token**: the DistilBERT notebook uses `huggingface_hub.login(token=...)`.
  - Set a Colab secret named `HF_TOKEN`.
- **Weights & Biases** (optional): training logs use `wandb`.
  - Set a Colab secret such as `w&b_api_key` (as referenced in the notebook).

Typical installs used across notebooks:
- `transformers==4.57.3`
- `datasets`
- `evaluate`
- `sentence-transformers`
- `scikit-learn`
- `umap-learn`
- `plotly`
- `pandas`
- `matplotlib`
- `torch`

### Local run (optional)
This repo does not currently include a `requirements.txt`.
If you want to run locally, create a virtualenv and install dependencies similar to the Colab cells:
- `pip install transformers==4.57.3 datasets evaluate sentence-transformers scikit-learn umap-learn plotly pandas matplotlib torch tqdm`

---

## Analysis

### Empirical findings (what we learned)
- DistilBERT achieves **~0.79–0.81 accuracy** and **~0.80 F1**, showing the dataset supports a reasonably strong binary classifier.
- Embedding-space clustering with a simple KMeans(2) baseline is weak (silhouette ≈ 0.03), suggesting:
  - bully vs normal is not a single linear/separable semantic axis, and/or
  - the “bully” concept is multi-modal.
- Set-based modeling is promising for conversation monitoring, but target design matters (mean vs concentration vs top-k).

### Overfitting / underfitting / collapse
- **DistilBERT**
  - Train loss (~0.36) vs eval loss (~0.48) indicates a modest generalization gap (mild overfitting).
- **kNN + dual-head Set Transformer**
  - The notebook shows **very low prediction variance** (near constant min/max, std ≈ 1e-6), which is a strong sign of underfitting or a degenerate solution.
  - The notebook text itself notes saturation in the “high-risk region”.

### Hallucination patterns
- This repo is primarily **discriminative modeling** (classification/regression), not LLM text generation, so classic LLM hallucinations do not apply.
- If/when you add an LLM-based RAG/agent layer, hallucinations can appear as:
  - unsupported rationales,
  - invented “policy citations”,
  - or fabricated conversation history.

### Limitations and failure cases
- **Ambiguity / pragmatics**: sarcasm, irony, playful insults between friends.
- **Domain shift**: platform slang changes rapidly; training data may become stale.
- **Binary labeling**: does not capture severity and type of harm.
- **Cluster construction**: synthetic clusters may not match real conversational dynamics.
- **Context missing**: the base dataset is message-level; conversation-level meaning often needs thread context.

### Suggested improvements
- **Longer / structured context**
  - Include true conversation threads and temporal windows.
  - Add explicit time features and per-user identity features (when permitted).
- **Better retrieval / clustering**
  - Replace KMeans with density-based methods, or use supervised contrastive training.
  - Build a vector DB index and evaluate retrieval quality.
- **Training improvements**
  - Add calibration (temperature scaling), class-conditional thresholds, and uncertainty.
  - Consider LoRA adapters for fast iteration.
- **Evaluation improvements**
  - Report per-slice metrics: short vs long messages, profanity-only, identity mentions.
  - Stress-test with adversarial paraphrases.

---

## Future work
- **Per-thread aggregations**
  - Replace the “rolling last-N messages” context with thread-aware features (per-thread mean risk, per-thread concentration, unique attackers, reply graph statistics).
  - This should reduce false positives caused by mixing unrelated conversations into the same context buffer.
- **Conversation-level RL**
  - Train an RL policy that chooses actions (alert / rate-limit / require verification / escalate to human) using conversation state features.
  - Reward shaping can penalize missed escalation (false negatives) and noisy moderation (false positives).
- **Time window modeling**
  - Incorporate real timestamps and learn risk over explicit windows (e.g., 30s / 5m / 1h), enabling true *density* estimates (messages per time).
  - Evaluate raid-like patterns: sudden bursts vs sustained harassment.
- **Target classification**
  - Extend from “is this bullying?” to “who/what is being targeted?” (user, group, protected class, creator/channel), and predict severity/type.
  - This enables downstream actions that depend on target type and policy.

---

## Reproducibility checklist
- Run notebooks in this order:
  1. `Cyberbullying_DistilBERT.ipynb`
  2. `Cyberbullying_MiniLM_Embeddings.ipynb`
  3. `Cyberbullying_Cluster_Level_Bullying_Detection_(Set_Based_Model).ipynb`
  4. `Cyberbullying_Set_Transformer.ipynb`
  5. `Cyberbullying_kNN.ipynb`
  6. `Cyberbullying_Inference_NO_TIME_AWARE.ipynb`

- Verify your `BASE_DIR` points to a writable Drive folder.

