!pip install -q xgboost optuna scikit-learn
!pip install -U datasets fsspec

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
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
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.3, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': 42,
        'verbosity': 0,
    }

    model = XGBClassifier(**params)
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1') 
    return scores.mean()

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=10)

print("the bset hyperparaetes:", study.best_params)
print("best F1 in cross-validation:", study.best_value)

final_model = XGBClassifier(
    **study.best_params,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    verbosity=0,
)

final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

print("final result in test data:")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

print("\n full report:\n")
print(classification_report(y_test, y_pred, digits=4))

final_cv_scores = cross_val_score(final_model, X_train, y_train, cv=5, scoring='accuracy')
print("Cross-Validation F1 in each fold:", final_cv_scores)
print("Mean:", final_cv_scores.mean())
print("اStandard deviation:", final_cv_scores.std())
