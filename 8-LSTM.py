!pip install optuna

import tensorflow as tf
import tensorflow_datasets as tfds
import optuna
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import numpy as np

(ds_train, ds_test), ds_info = tfds.load(
    "imdb_reviews",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

train_texts = []
train_labels = []
for text, label in tfds.as_numpy(ds_train):
    train_texts.append(text.decode("utf-8"))
    train_labels.append(label)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

def preprocess(texts, labels):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=200, padding='post')
    return padded, np.array(labels)

split_idx = int(len(train_texts)*0.8)
X_train, y_train = preprocess(train_texts[:split_idx], train_labels[:split_idx])
X_val, y_val = preprocess(train_texts[split_idx:], train_labels[split_idx:])

test_texts = []
test_labels = []
for text, label in tfds.as_numpy(ds_test):
    test_texts.append(text.decode("utf-8"))
    test_labels.append(label)
X_test, y_test = preprocess(test_texts, test_labels)

def objective(trial):
    model = create_model(trial)

    history = model.fit(
        X_train, y_train,
        epochs=3,
        batch_size=trial.suggest_categorical("batch_size", [32, 64]),
        validation_data=(X_val, y_val),
        verbose=0 
    )

    val_acc = history.history['val_accuracy'][-1]
    return val_acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)  

print("Best parameters:", study.best_params)


best_model = create_model(study.best_trial)
best_model.fit(X_train, y_train, epochs=5, batch_size=study.best_params["batch_size"], verbose=1)

y_pred_prob = best_model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

print("Final results:")
print(f"Accuracy :  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision:  {precision_score(y_test, y_pred):.4f}")
print(f"Recall   :  {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score :  {f1_score(y_test, y_pred):.4f}")
print("\n📋 Classification Report:\n", classification_report(y_test, y_pred, digits=4))
