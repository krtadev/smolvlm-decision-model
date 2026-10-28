---
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
library_name: peft
tags:
- lora
- decision-model
- scienceqa
---

# SmolVLM-256M ScienceQA decision adapter

LoRA adapter (rank 16, 4.9M parameters) that turns SmolVLM-256M-Instruct into a multiple-choice decision model: one forward
pass, then a softmax over the `" A"` … `" E"` token scores at the last position. Divide the logits by T = 1.329 for calibrated
probabilities.

ScienceQA test split, 2,017 questions with images: **76.5% ± 1.8** accuracy (zero-shot 39.0%, random guess 35.5%), ECE 0.024
after calibration.

The usage code, the training script and all the metrics are in the repository README.
