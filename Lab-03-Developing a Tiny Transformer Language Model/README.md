# Lab Sheet 3 – Tiny Transformer Language Model

## Overview

This practical focuses on implementing and training a small character-level Transformer language model using the **TinyStories** dataset from Hugging Face.

The model learns to predict the next character in a sequence and generates simple story continuations based on given prompts. The practical also explores how changing the Transformer architecture affects model performance.

## Objectives

- Load and explore the TinyStories dataset.
- Preprocess text and perform character-level tokenization.
- Prepare input and target sequences for next-character prediction.
- Implement a Transformer architecture using PyTorch.
- Train the model using cross-entropy loss and backpropagation.
- Monitor training and validation loss.
- Evaluate the model using validation loss and perplexity.
- Generate stories using different starting prompts.
- Compare baseline and smaller Transformer configurations.

## Technologies and Libraries Used

- Python
- Google Colab
- PyTorch
- Hugging Face Datasets
- Matplotlib
- NumPy

## Dataset

**Dataset:** [TinyStories – Hugging Face](https://huggingface.co/datasets/roneneldan/TinyStories)

A subset of 5,000 training stories was used to keep the experiment manageable. Character-level tokenization was applied to convert text into numerical sequences suitable for training the Transformer model.

## Model Architecture

The baseline Transformer model used the following configuration:

| Parameter | Value |
|---|---:|
| Vocabulary size | 90 |
| Block size | 128 |
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 4 |
| Dropout | 0.1 |
| Batch size | 64 |
| Training epochs | 5 |
| Trainable parameters | 821,248 |

The implementation includes token embeddings, positional embeddings, masked multi-head self-attention, feed-forward networks, residual connections, and layer normalization.

## Training and Evaluation Results

The baseline model was trained using the CUDA-enabled GPU in Google Colab.

| Epoch | Training Loss | Validation Loss | Perplexity |
|---|---:|---:|---:|
| 1 | 2.1849 | 1.6626 | 5.27 |
| 2 | 1.5662 | 1.3176 | 3.73 |
| 3 | 1.3402 | 1.1910 | 3.29 |
| 4 | 1.2468 | 1.1397 | 3.13 |
| 5 | 1.2117 | 1.1284 | 3.09 |

Both training and validation losses decreased consistently throughout the five epochs, indicating that the model learned patterns from the training data and improved its performance on unseen validation data.

**Final baseline results:**
- Validation loss: **1.1284**
- Validation perplexity: **3.09**

## Model Comparison

A smaller Transformer configuration was also tested to observe the effect of model size on performance.

| Model | Parameters | Validation Loss | Perplexity |
|---|---:|---:|---:|
| Baseline (4 layers, 4 heads) | 821,248 | 1.1284 | 3.09 |
| Smaller (2 layers, 2 heads) | 424,704 | 1.1994 | 3.32 |

The baseline model achieved better validation loss and perplexity. However, the smaller model used approximately half the number of parameters, making it a more lightweight alternative.

## Story Generation

The trained baseline model was tested using three starting prompts:

- `One day, a little girl`
- `The little dog`
- `Once upon a time`

The generated outputs contained character names and basic story-like structures, showing that the model learned some language patterns. However, grammatical errors, repetition, and incoherent phrases were also observed.

This demonstrates the limitations of a small character-level Transformer trained for only five epochs on a limited dataset.

## Mechanisms Used to Improve Performance

The implementation included several mechanisms to support learning and training stability:

- Character and positional embeddings.
- Masked multi-head self-attention.
- Feed-forward networks.
- Residual connections and layer normalization.
- Dropout to help reduce overfitting.
- AdamW optimizer with weight decay.
- Gradient clipping.
- Learning-rate scheduling.
- Validation loss and perplexity monitoring.
- GPU acceleration using CUDA.

## Learning Outcomes

Through this practical, I gained experience in:

1. Understanding the basic architecture and working principles of Transformer language models.
2. Loading and preprocessing text datasets using Hugging Face.
3. Performing character-level tokenization.
4. Preparing input and target sequences for next-character prediction.
5. Implementing masked self-attention and multi-head attention using PyTorch.
6. Training a Transformer model using cross-entropy loss and backpropagation.
7. Monitoring training progress using loss curves.
8. Evaluating language models using validation loss and perplexity.
9. Generating text using trained language models.
10. Comparing different model configurations and understanding the relationship between model size and performance.

## Conclusion

This practical provided hands-on experience in building a small Transformer language model from scratch using PyTorch. The baseline model achieved a final validation loss of 1.1284 and perplexity of 3.09. The comparison experiment showed that the larger baseline model performed slightly better than the smaller configuration, while the generated stories demonstrated both the language-learning capabilities and limitations of a small Transformer.

## Repository Contents

- `Lab_Sheet_3_Tiny_Transformer.ipynb` – Completed Google Colab notebook containing the implementation, training, evaluation, experiments, and observations.
- `README.md` – Practical overview, results, and learning outcomes.
- `screenshots/` – Screenshots of important implementation steps and outputs.

---

**Environment:** Google Colab  
**Framework:** PyTorch  
**Dataset:** TinyStories  
**Practical:** Lab Sheet 3 – Tiny Transformer
