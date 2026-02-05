!pip install -U datasets fsspec optuna
!pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
import optuna

dataset = load_dataset("imdb")

train_texts = dataset['train']['text']
train_labels = dataset['train']['label']
test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
y_train = train_labels
y_test = test_labels

def objective(trial):
    C = trial.suggest_float("C", 1e-4, 10.0, log=True)
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)
    dual = trial.suggest_categorical("dual", [True, False])

    model = LinearSVC(
        C=C,
        tol=tol,
        dual=dual,
        max_iter=5000,
        loss='squared_hinge',
        penalty='l2',
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(" Best hyperparameters:")
print(study.best_params)
print("The best F1:", study.best_value)

best_params = study.best_params
final_model = LinearSVC(
    C=best_params["C"],
    tol=best_params["tol"],
    dual=best_params["dual"],
    max_iter=5000,
    loss='squared_hinge',
    penalty='l2',
    random_state=42
)
final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)
print("\n🎯 Final Evaluation")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

feature_names = vectorizer.get_feature_names_out()
coefs = final_model.coef_[0]
top_pos = np.argsort(coefs)[-10:]
top_neg = np.argsort(coefs)[:10]

print("\n🔝 Top Positive Words:")
for i in top_pos:
    print(f"{feature_names[i]}: {coefs[i]:.4f}")

print("\n🔻 Top Negative Words:")
for i in top_neg:
    print(f"{feature_names[i]}: {coefs[i]:.4f}")

cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
print("\n📊 Cross-Validation Accuracy in each fold:", cv_scores)
print("Mean:", cv_scores.mean())
print("Standard deviation:", cv_scores.std())

