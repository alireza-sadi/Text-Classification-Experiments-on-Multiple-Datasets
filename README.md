# Text-Classification-Experiments-on-Multiple-Datasets

This repository contains a series of experiments on text classification using multiple traditional machine learning and deep learning models across three benchmark datasets:

IMDB movie reviews (sentiment analysis)

Fake and real news detection

Toxicity classification

The goal is to systematically compare different architectures and training strategies on common NLP tasks.

Datasets
IMDB
Binary sentiment classification (positive vs. negative) of movie reviews.

Fake and Real News
Binary classification of news articles as fake or real.

Toxicity Classification
Binary detection of toxic vs. non-toxic user-generated content.

(You can add dataset links or citations here if you are allowed to share them.)

Models
The following models are implemented and evaluated:

Logistic Regression (linear classifier with regularization)

Naïve Bayes (Multinomial)

k-Nearest Neighbors (KNN)

Decision Tree

Random Forest

XGBoost

Support Vector Machine (SVM / LinearSVC)

LSTM (sequence model with word embeddings)

BERT (pre-trained transformer model fine-tuned for classification)

Hyperparameters for several models are tuned using Optuna or cross-validation to obtain competitive baselines.

Evaluation
Each model is evaluated using the following metrics:

Accuracy

Precision

Recall

F1-Score

Results are reported separately for each dataset, and summary tables highlight the trade-offs between models (e.g., linear vs. tree-based vs. transformer-based).
Requirements
Typical main dependencies (adapt to your environment):

Python 3.10+

NumPy, pandas, scikit-learn

XGBoost

Optuna

TensorFlow / Keras or PyTorch (for LSTM and BERT)

Hugging Face Transformers (for BERT)

Citation
If you use this code or results in academic work, please consider citing this repository or referencing it in the experimental setup section of your paper/thesis.
