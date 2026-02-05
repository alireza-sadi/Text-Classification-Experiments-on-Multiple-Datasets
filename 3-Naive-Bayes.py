!pip install -q optuna
!pip install -U datasets fsspec
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score

def objective(trial):
    alpha = trial.suggest_float('alpha', 1e-3, 10.0, log=True)

    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return f1_score(y_test, preds)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print("✅ the best hyperparameters:", study.best_params)
print("✅ The best F1:", study.best_value)

best_alpha = study.best_params['alpha']
final_model = MultinomialNB(alpha=best_alpha)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("\n🎯 Final result:")
print(f"Accuracy :  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall   :  {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score :  {f1_score(y_test, y_pred):.4f}")
print("\n📄 full report:\n")
print(classification_report(y_test, y_pred, digits=4))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
print("\n📊 Cross-Validation Accuracy in each fold:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())
