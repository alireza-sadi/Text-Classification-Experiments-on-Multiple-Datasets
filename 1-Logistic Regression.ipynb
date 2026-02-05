!pip install -q optuna
!pip install -U datasets fsspec
!pip install torch==2.6.0+cpu --index-url https://download.pytorch.org/whl/cpu

from datasets import load_dataset

dataset = load_dataset("imdb")

train_texts = dataset['train']['text']
train_labels = dataset['train']['label']

test_texts = dataset['test']['text']
test_labels = dataset['test']['label']

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)
)

X_train = vectorizer.fit_transform(train_texts)
X_test = vectorizer.transform(test_texts)
y_train = train_labels
y_test = test_labels

import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def objective(trial):
    C = trial.suggest_float('C', 1e-4, 1e2, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'saga', 'liblinear'])

    model = LogisticRegression(
        C=C,
        solver=solver,
        penalty='l2',
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("best hyperparameters:", study.best_params)
print("the best F1:", study.best_value)

best_params = study.best_params

final_model = LogisticRegression(
    C=best_params['C'],
    solver=best_params['solver'],
    penalty='l2',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("Finall results:")
print(f"Accuracy :  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall   :  {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score :  {f1_score(y_test, y_pred):.4f}")
print("\n full report:\n")
print(classification_report(y_test, y_pred, digits=4))

from sklearn.model_selection import cross_val_score


scores = cross_val_score(final_model, X_train, train_labels, cv=5, scoring='accuracy')

print("Accuracy in each fold:", scores)
print("Mean Accuracy:", scores.mean())
print("Standard deviation:", scores.std())
