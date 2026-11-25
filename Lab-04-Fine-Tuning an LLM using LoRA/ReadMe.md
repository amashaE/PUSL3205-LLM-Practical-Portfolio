
# Lab Sheet 04 – Fine-Tuning an LLM using LoRA/QLoRA

## Overview

This practical demonstrates how to fine-tune a pretrained instruction-following language model for a specific domain using **LoRA (Low-Rank Adaptation)** and **QLoRA (Quantized LoRA)**.

The experiment uses a fictional computer shop called **TechNova Computers** to adapt a small language model for customer-support questions.

## Objectives

- Load and configure a pretrained instruction-tuned language model.
- Understand GPU availability and memory requirements.
- Create a domain-specific question-and-answer dataset.
- Apply 4-bit quantization using BitsAndBytes.
- Configure LoRA adapters for parameter-efficient fine-tuning.
- Fine-tune the model using TRL's SFTTrainer.
- Compare the original and fine-tuned model responses.

## Tools and Technologies

- Python
- Google Colab
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- TRL
- BitsAndBytes

## Model and Environment

| Component | Configuration |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Platform | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU memory | Approximately 14.56 GiB |
| Quantization | 4-bit NF4 |
| Compute precision | FP16 |
| Fine-tuning method | QLoRA |
| Training framework | TRL SFTTrainer |

## Dataset

A custom dataset containing **32 question-and-answer pairs** was created for TechNova Computers.

The dataset covers topics such as:
- Shop opening hours and contact information
- Products and services
- Discounts and warranty policies
- Returns and repairs
- Delivery and payment methods

The dataset was divided into:
- Training set: 25 examples
- Evaluation set: 7 examples

## Implementation

The practical was completed through the following stages:

1. Installed the required Python libraries.
2. Identified the available GPU and its memory capacity.
3. Loaded the Qwen2.5-0.5B-Instruct model and tokenizer.
4. Created and prepared the TechNova customer-support dataset.
5. Configured 4-bit NF4 quantization with double quantization.
6. Configured LoRA adapters for selected model layers.
7. Fine-tuned the model using TRL's SFTTrainer.
8. Evaluated training and evaluation loss.
9. Compared the original and fine-tuned model using the same customer-support questions.
10. Saved the fine-tuned LoRA adapter.

## Training Results

The experiment trained 8,798,208 parameters, representing approximately **2.7162%** of the 323,917,696 parameters reported by the notebook.

| Epoch | Training Loss | Evaluation Loss |
|---|---:|---:|
| 1 | 1.8039 | 1.4901 |
| 2 | 0.5634 | 0.5562 |
| 3 | 0.2336 | 0.5336 |
| 4 | 0.3295 | 0.5780 |
| 5 | 0.0522 | 0.6142 |
| 6 | 0.0912 | 0.6414 |

Training loss generally decreased, while evaluation loss reached its lowest value in epoch 2 and increased afterwards. This suggests possible overfitting.

## Observations and Learning Outcomes

- 4-bit quantization reduced the memory requirements of the base model.
- LoRA enabled fine-tuning by updating only a small proportion of the model parameters.
- The fine-tuned model learned some TechNova-specific information.
- Some answers were correct, but others contained factual errors or inconsistencies.
- The small dataset limited the model's ability to generalise reliably.
- Evaluation loss helped identify possible overfitting during training.

## Limitations

- The dataset contained only 32 examples.
- The evaluation set contained only 7 examples.
- Some fine-tuned responses contradicted the information in the training dataset.
- The experiment used fictional business information rather than real customer-support data.

## Conclusion

This practical demonstrated how a pretrained language model can be adapted to a specific domain using QLoRA and LoRA. The experiment successfully completed the fine-tuning process using a Tesla T4 GPU while training only a small proportion of the model parameters.

However, the results showed that fine-tuning does not guarantee factual accuracy or consistent generalisation. Dataset quality, training examples and proper evaluation are important for developing reliable domain-specific language models.

## Repository Contents

- `Lab_Sheet_04.ipynb` – Completed Google Colab notebook.
- `README.md` – Practical description, implementation and observations.
- `screenshots/` – Screenshots of important implementation steps and outputs.

