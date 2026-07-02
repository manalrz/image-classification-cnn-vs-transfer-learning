# Image Classification with CNNs: From Scratch vs. Transfer Learning

A comparative study of deep learning approaches for medical and agricultural image classification, built with PyTorch. This project evaluates a custom CNN architecture against a pretrained ResNet18 (frozen and fine-tuned) across two distinct domains: skin lesion diagnosis and plant disease detection.

## Overview

The goal was to understand how architectural choices and training strategy affect performance on small vs. large datasets, and to what extent transfer learning helps when labeled data is limited.

**Objectives:**
- Build and train a CNN from scratch using PyTorch
- Compare from-scratch training against transfer learning (frozen and fine-tuned ResNet18)
- Analyze the impact of dataset size, image quality, and model complexity on results

## Datasets

| Dataset | Images | Classes | Notes |
|---|---|---|---|
| Melanoma Cancer | 2,000 | Benign / Malignant | Balanced, resized to 224×224 |
| New Plant Diseases | 60,000 | Healthy / Disease | Imbalanced (more disease samples), augmented, resized to 224×224 |

Both datasets were split 80/20 (train/validation).

## Models

**1. CNN from scratch** — 2 convolutional layers (16 and 32 filters) with ReLU activations, followed by fully connected layers.

**2. ResNet18 (frozen)** — pretrained backbone with a custom classification head (512 → 128 → 1).

**3. ResNet18 (fine-tuned)** — same architecture, with backbone layers partially unfrozen for fine-tuning.

## Results

| Model | Dataset | Accuracy | F1-score |
|---|---|---|---|
| CNN from scratch | Melanoma | 83% | 0.88 |
| CNN from scratch | Plant Disease | 98% | 0.98 |
| ResNet18 (frozen) | Melanoma | 91% | 0.92 |
| ResNet18 (frozen) | Plant Disease | 71% | 0.83 |
| ResNet18 (fine-tuned) | Melanoma | 96% | 0.96 |
| ResNet18 (fine-tuned) | Plant Disease | 80% | 0.89 |

## Key Takeaways

- On the small, balanced Melanoma dataset, transfer learning (especially fine-tuned ResNet18) clearly outperformed training from scratch.
- On the large Plant Disease dataset, the from-scratch CNN performed best — with enough data, a simpler model can match or beat a pretrained one.
- Fine-tuning consistently improved on frozen transfer learning, confirming the value of adapting pretrained features to the target domain.

## Tech Stack

Python, PyTorch, scikit-learn (metrics)

## Team

Manâl Rhazza, Marianne Cheron, Oumou N Dongo — ENSEIRB-MATMECA, supervised by Pedro Coutinho
