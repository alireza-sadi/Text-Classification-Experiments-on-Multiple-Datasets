!pip install -q optuna scikit-learn
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

def objective(trial):
    n_neighbors = trial.suggest_int("n_neighbors", 3, 15)
    weights = trial.suggest_categorical("weights", ["uniform", "distance"])
    metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "cosine"])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric=metric
    )

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
    return scores.mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print(" best hyperparameters:", study.best_params)
print(" best F1 in Cross-Validation:", study.best_value)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# مدل نهایی با بهترین پارامترها
best_params = study.best_params
final_model = KNeighborsClassifier(**best_params)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# نتایج نهایی
print("🎯 result of KNN:")
print(f"Accuracy :  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall   :  {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score :  {f1_score(y_test, y_pred):.4f}")
print("\n📋 full report:\n", classification_report(y_test, y_pred, digits=4))

from sklearn.model_selection import cross_val_score

scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
print("\n📊 Cross-Validation Accuracy in each fold:", scores)
print("Mean:", scores.mean())
print("Standard deviation:", scores.std())
