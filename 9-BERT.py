!pip install -q transformers
!pip install -U datasets fsspec
!pip install -U tensorflow
from datasets import load_dataset

dataset = load_dataset("imdb")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


from transformers import DefaultDataCollator

data_collator = DefaultDataCollator()

tokenized_datasets.set_format("tensorflow", columns=["input_ids", "attention_mask", "label"])

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns="input_ids",
    label_cols="label",
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator
)

tf_test_dataset = tokenized_datasets["test"].to_tf_dataset(
    columns="input_ids",
    label_cols="label",
    shuffle=False,
    batch_size=8,
    collate_fn=data_collator
)


from transformers import TFAutoModelForSequenceClassification
import tensorflow as tf

model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

model.fit(tf_train_dataset, validation_data=tf_test_dataset, epochs=2)


import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

y_true = np.concatenate([y[1].numpy() for y in tf_test_dataset])

preds = model.predict(tf_test_dataset)
y_pred = np.argmax(preds.logits, axis=1)

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print("\n Classification Report:\n")
print(classification_report(y_true, y_pred, digits=4))
