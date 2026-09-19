# EEG-LLM: Foundation Language Model for EEG

## Overview
This repository contains the implementation for our journal paper:
"Foundation Language Model for EEG: Perplexity as a Universal 
Biomarker for Zero-Shot Neurological Disorder Detection"

## Core Idea
Current AI models for EEG-based disorder detection require large 
amounts of labeled training data — a major bottleneck in clinical 
neuroscience. We propose a fundamentally different approach:

Train a language model exclusively on healthy EEG signals.
Use the model's perplexity score as an anomaly detector.
High perplexity = the brain pattern looks unfamiliar = potential disorder.
Zero labeled anomalies required at any point.

## Method
1. EEG signals are converted into discrete token sequences using 
   three strategies: raw patch tokens, spectral tokens, and wavelet tokens.
2. GPT-2 Small is fine-tuned on healthy EEG token sequences only.
3. Perplexity is computed on test segments — healthy or disordered.
4. A statistical threshold (mean + 2σ of healthy perplexity) flags anomalies.

## Proof of Concept Result
Tested on CHB-MIT Scalp EEG Dataset (patient chb01):
- Healthy EEG perplexity: 1.67 ± 0.08
- Seizure EEG perplexity: 2.07
- Threshold: 1.84
- Result: Seizure correctly flagged with zero labeled training data

## Dataset
CHB-MIT Scalp EEG Database — freely available on PhysioNet
https://physionet.org/content/chbmit/1.0.0/

## Tech Stack
- Python 3.11
- MNE — EEG processing
- HuggingFace Transformers — GPT-2 fine-tuning
- PyTorch — model training
- Scikit-learn — baseline models and evaluation metrics
- SciPy — signal processing and statistical tests
- Matplotlib / Seaborn — visualization

## Authors
Rohini B and T Sai Snehitha
Amrita Vishwa Vidyapeetham, Bengaluru
