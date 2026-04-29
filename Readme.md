# i232599-NLP-Assignment03

## Assignment 3: Transformers + RAG

This project implements a complete **Retrieval-Augmented Generation (RAG) pipeline** from scratch using Transformer architectures. The system performs review understanding, retrieves similar examples, and generates grounded explanations.

---

## 📌 Project Overview

The pipeline consists of three main components:

### 🔹 Part A — Encoder (Understanding)

* Custom **encoder-only Transformer** (implemented from scratch)
* Multi-task learning:

  * Sentiment Classification (Negative / Neutral / Positive)
  * Derived Feature Prediction (user-defined)
* Outputs:

  * Sentiment label
  * Derived feature
  * Dense vector embedding (used for retrieval)

---

### 🔹 Part B — Retrieval Module

* Stores embeddings of all training reviews
* Retrieves **top-k similar reviews** using cosine similarity
* Provides contextual grounding for generation

---

### 🔹 Part C — Decoder (Generation)

* Custom **decoder-only Transformer** (from scratch)
* Generates **1–2 sentence explanations**
* Input includes:

  * Original review
  * Predicted sentiment
  * Derived feature
  * Retrieved similar reviews (RAG)

---

## 📊 Dataset

* Source: Amazon Reviews Dataset
* Subset:

  * 3+ product categories
  * ~30,000–45,000 total reviews
* Each sample includes:

  * Review text
  * Star rating (1–5)

### Data Split

* Train: 70%
* Validation: 15%
* Test: 15%

---

## ⚙️ Preprocessing Pipeline

* Text cleaning
* Tokenization
* Vocabulary construction (training data only)
* Token → index conversion
* Padding & truncation (fixed sequence length)

---

## 🧠 Model Implementation

### Encoder

* Multi-head self-attention (custom)
* Feed-forward layers
* Residual connections + Layer normalization
* Joint loss for multi-task learning

### Retrieval

* Embedding storage (NumPy)
* Cosine similarity search
* Configurable top-k retrieval

### Decoder

* Masked self-attention (no future tokens)
* Autoregressive text generation
* Language modeling objective

---

## 📈 Evaluation

### Part A

* Accuracy
* Macro F1-score

### Part B

* Qualitative retrieval analysis

### Part C

* Perplexity (with RAG vs without RAG)
* Generated explanation samples

---


---

## 📁 Repository Structure

```
.
├── i23XXXX-NLP-Assignment2.ipynb
├── models/
│   ├── encoder_best.pt
│   ├── decoder_rag_best.pt
│   └── decoder_norag_best.pt
├── results/
│   ├── train_embeddings.npy
│   ├── test_embeddings.npy
│   ├── vocab.json
│   ├── train_meta.json
│   ├── perplexity.json
├── figure/
│   ├── encoder_training_curves.png
│   ├── decoder_training_curves.png
│   ├── rag_ablation.png
│   ├── retrieval_analysis.png
│   └── sentiment_distribution.png
└── README.md
```

---

## ▶️ How to Run

1. Open the notebook:

   ```
   i232599-NLP-Assignment3.ipynb
   ```

2. Run all cells sequentially:

   * Preprocessing
   * Training (Encoder → Retrieval → Decoder)
   * Evaluation

3. Outputs will be saved in:

   * `models/`
   * `results/`

---

## 🔧 Requirements

* Python 3.x
* PyTorch
* NumPy
* Matplotlib
* Scikit-learn

(Recommended: Run on GPU using Google Colab)

---

## 🚫 Restrictions

* No pretrained models (BERT, GPT, T5, etc.)
* No built-in Transformer modules:

  * `nn.Transformer`
  * `nn.MultiheadAttention`
  * `nn.TransformerEncoder`

All components are implemented from scratch.



