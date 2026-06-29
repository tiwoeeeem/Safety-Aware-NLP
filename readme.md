# Safety-Aware NLP Pipeline Using Gemma 2B

![Safety Architecture](https://img.shields.io/badge/Architecture-Dual--LLM%20Safety-blue)
![Model](https://img.shields.io/badge/Model-Gemma--2B-orange)
![Fine-Tuning](https://img.shields.io/badge/Fine--Tuning-LoRA%20%7C%20QLoRA-success)

## 📌 Overview

As Large Language Models (LLMs) become increasingly integrated into public-facing applications, ensuring they generate safe, non-toxic, and helpful content is a critical priority. This project implements a proactive, **dual-layer safety alignment system** utilizing the **Gemma 2B** architecture to act as a robust firewall against toxic inputs, jailbreak attempts, and model hallucinations.

### Why Safety-Aware NLP?

- **Harm Reduction:** Prevents the generation of dangerous instructions or hate speech.
- **User Trust:** Ensures reliable and safe interactions, essential for public adoption.
- **Adversarial Robustness:** Builds inherent defenses against prompt injections and malicious misuse.

---

## System Architecture

This pipeline leverages the Gemma 2B model (featuring GeGLU activations and Multi-Query Attention for high efficiency) in a two-stage filtering and generation pipeline:

1. **Pre-Check (Input Guard):** Scans the user's prompt before it reaches the Main LLM. Blatant toxicity is flagged immediately, saving compute.
2. **Main LLM (Generative Safety):** A fine-tuned generator that naturally and politely refuses harmful prompts.
3. **Post-Check (Output Guard):** Scans the final generated output, acting as a final safety net against accidental hallucinations or jailbreaks.

---

## Model 1: Main LLM (Generative Safety)

Responsible for conversational generation, this model is fine-tuned via Supervised Fine-Tuning (SFT) to handle nuanced instructions and consistently refuse malicious prompts.

- **Dataset:** Anthropic `hh-rlhf` (10,000-sample subset focusing on safe/helpful responses).
- **Methodology:** Parameter-Efficient Fine-Tuning (PEFT) using **LoRA** (Rank=8, Alpha=16) loaded in Float16.
- **Target Modules:** `q_proj`, `v_proj`, `k_proj`, `o_proj`.
- **Performance:** - Test Loss: `1.8205`
  - Test Perplexity: `6.1751` (Demonstrating strong syntax retention and generative coherence).

---

## Model 2: Guard LLM (Toxicity Scorer)

Operates not as a text generator, but as a **sequence regression scorer**. It evaluates input/output text and predicts a continuous toxicity score. If the score exceeds a designated threshold, the system triggers a hardcoded refusal.

- **Dataset:** `ToxiGen` (raw toxicity scores normalized to a `0.0 - 1.0` scale).
- **Methodology:** 4-bit Quantized LoRA (**QLoRA**) with a custom Hugging Face `RegressionTrainer` computing Mean Squared Error (MSE).
- **Performance:**
  - Test MSE: `0.1097`
  - Test MAE: `0.2493` (Highly accurate alignment with ground-truth toxicity).

---

## Key Features & Heuristics

- **Programmatic Cut-Offs:** Implemented string-splitting heuristics (`generated_text.split("User:")[0]`) to actively intercept and remove multi-turn dialogue hallucinations caused by autocomplete behaviors.
- **Memory-Safe Tokenization:** Dynamic truncation (256 tokens for Main LLM, 512 for Guard LLM) and proper right-side EOS padding to prevent OOM spikes during training and inference.

---

## Future Scope

- **Native Chat Templates:** Transitioning from raw text formatting to Gemma's native `<start_of_turn>` and `<end_of_turn>` special tokens for stricter generation stopping.
- **Dynamic Thresholding:** Adjusting the Guard LLM's toxicity threshold dynamically based on the application context (e.g., stricter limits for child-facing apps).
- **Knowledge Distillation:** Using larger teacher models to distill advanced safety behaviors into the 2B framework to push perplexity down further.
