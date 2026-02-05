# Text-Classification-Experiments-on-Multiple-Datasets

# Text Classification Experiments on Multiple Datasets

This repository contains a series of experiments on text classification using both traditional machine learning and modern deep learning models across three benchmark datasets:

- **IMDB movie reviews** (sentiment analysis)
- **Fake and real news** (fake news detection)
- **Toxicity classification** (toxic vs. non‑toxic content)

The main goal is to systematically compare different architectures and training strategies on common NLP tasks.

---

## Datasets

1. **IMDB Movie Reviews**  
   Binary sentiment classification task with reviews labeled as positive or negative.

2. **Fake and Real News**  
   Binary classification of news articles as fake or real.

3. **Toxicity Classification**  
   Binary classification of user‑generated content as toxic or non‑toxic.

> Note: Please make sure you comply with the original dataset licenses and terms of use.  
> (we use this datasets from exists library)

---

## Implemented Models

The following models are implemented and evaluated:

- **Logistic Regression**
- **Naïve Bayes** (Multinomial)
- **k‑Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **XGBoost**
- **Support Vector Machine (SVM / LinearSVC)**
- **LSTM** (sequence model with embeddings)
- **BERT** (pre‑trained transformer fine‑tuned for classification)

Hyperparameters for several models are tuned using modern optimization techniques (e.g., Bayesian optimization / Optuna) and cross‑validation.

---

## Metrics

Each model is evaluated using standard classification metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1‑Score**

Results are reported separately for each dataset, and comparison tables highlight trade‑offs between different model families (linear, tree‑based, distance‑based, and transformer‑based).

---

## Citation
If you use this code or results in academic work, please consider citing this repository or referencing it in the experimental setup section of your paper/thesis.
