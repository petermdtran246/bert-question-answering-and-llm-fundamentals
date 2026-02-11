# BERT Question Answering & LLM Fundamentals

This project is a hands-on exploration of **how BERT-style models perform extractive Question Answering**, along with a broader walkthrough of **LLM fundamentals using Hugging Face Transformers**.

Rather than treating language models as black boxes, the focus here is on understanding:
- how questions and context are encoded
- how tokens and segment embeddings are constructed
- how BERT predicts answer spans
- and how these mechanics differ from generative models like GPT

---

## What This Project Covers

### 1. BERT Question Answering (SQuAD-style)
- Loading a **fine-tuned BERT QA model**
- Encoding questions + context using special tokens (`[CLS]`, `[SEP]`)
- Understanding:
  - `input_ids`
  - `token_type_ids` (question vs context)
  - `attention_mask`
- Running inference directly with PyTorch
- Extracting answers via **start and end logits**
- Visualizing token-level probabilities

### 2. Building a Simple FAQ Bot
- Using BERT QA to answer questions from a fixed knowledge context
- Manual construction of segment embeddings
- Cleaning subword outputs (e.g. `##` tokens)
- Handling unanswerable or ambiguous questions

### 3. Model Variants Overview
- **BERT** – baseline encoder model
- **RoBERTa** – optimized BERT training strategy
- **DistilBERT** – lightweight, faster inference model

### 4. Model Comparison & Trade-offs
- Accuracy vs latency
- Training strategy vs architecture
- When to prefer encoder-only QA models over generative models

---

## Why BERT for Question Answering?

BERT excels at **extractive QA** tasks because it:
- encodes full bidirectional context
- predicts answer spans instead of generating text
- is well-suited for FAQ systems, search, and document understanding

This makes it fundamentally different from GPT-style autoregressive models, which are optimized for generation rather than precise span selection.

---

## Key Technical Concepts Demonstrated

- Transformer encoder-only architecture
- Tokenization differences and special tokens
- Segment embeddings (question vs context)
- Start / end span prediction using logits
- Torch-based inference (no pipelines)
- Token-level probability visualization

---
