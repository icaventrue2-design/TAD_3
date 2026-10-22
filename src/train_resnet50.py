import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from preprocess import load_image, make_dataset
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

SEED = 42
IMAGE_SIZE = 512
BATCH_SIZE = 32
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 10

IMAGE_DIR = "data/processed"

CLASS_NAMES = [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab"
]

NUM_CLASSES = len(CLASS_NAMES)

tf.keras.utils.set_random_seed(SEED)

train_df = pd.read_csv("data/processed/train_processed.csv")
test_df = pd.read_csv("data/processed/test_processed.csv")

train_df["label"] = (
    train_df[CLASS_NAMES]
    .values.argmax(axis=1)
)

train_part, val_part = train_test_split(
    train_df,
    test_size=0.20,
    random_state=SEED,
    stratify=train_df["label"]
)

train_part = train_part.reset_index(drop=True)
val_part = val_part.reset_index(drop=True)

print("Train samples:", len(train_part))
print("Validation samples:", len(val_part))
print(train_part["label"].value_counts().sort_index())


train_ds = make_dataset(train_part, training=True)
val_ds = make_dataset(val_part, training=True)
test_ds = make_dataset(test_df)

base_model = tf.keras.applications.ResNet50(
    weights="imagenet",
    include_top=False,
    input_shape=(512, 512, 3)
)

# Этап 1: замораживаем backbone
base_model.trainable = False

inputs = tf.keras.Input(
    shape=(512, 512, 3)
)

x = tf.keras.applications.resnet50.preprocess_input(inputs)

training=False сохраняет BatchNormalization
в режиме inference даже при fine-tuning
x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.30)(x)

outputs = tf.keras.layers.Dense(
    4,
    activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_head = [
    tf.keras.callbacks.ModelCheckpoint(
        "resnet50_head_best.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]

history_head = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=HEAD_EPOCHS,
    callbacks=callbacks_head
)

Загружаем лучшую модель после первого этапа
model = tf.keras.models.load_model(
    "resnet50_head_best.keras"
)

# Размораживаем backbone
base_model.trainable = True

# По умолчанию замораживаем все слои
for layer in base_model.layers:
    layer.trainable = False

# Размораживаем последние 30 слоёв,
# кроме BatchNormalization
for layer in base_model.layers[-30:]:
    if not isinstance(
        layer, tf.keras.layers.BatchNormalization
    ):
        layer.trainable = True

print(
    "Trainable layers:",
    sum(layer.trainable for layer in base_model.layers)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

callbacks_finetune = [
    tf.keras.callbacks.ModelCheckpoint(
        "models/resnet50_finetuned_best.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1
    )
]

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks_finetune
)
