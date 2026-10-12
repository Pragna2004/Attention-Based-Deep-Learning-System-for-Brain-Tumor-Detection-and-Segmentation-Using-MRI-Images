import tensorflow as tf
import numpy as np
import json
from sklearn.utils.class_weight import compute_class_weight

IMG_SIZE = 224
BATCH_SIZE = 16

train_dir = "dataset/detection/Train"
val_dir = "dataset/detection/Val"

# -----------------------------
# LOAD DATA
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode="categorical"
)

# Save class names
class_names = train_ds.class_names
with open("models/classes.json", "w") as f:
    json.dump(class_names, f)

# -----------------------------
# PREPROCESS
# -----------------------------
train_ds = train_ds.map(lambda x,y: (
    tf.keras.applications.efficientnet.preprocess_input(x), y
))
val_ds = val_ds.map(lambda x,y: (
    tf.keras.applications.efficientnet.preprocess_input(x), y
))

# -----------------------------
# DATA AUGMENTATION (CRITICAL)
# -----------------------------
data_aug = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
labels = []
for _, y in train_ds:
    labels.extend(np.argmax(y.numpy(), axis=1))

labels = np.array(labels)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -----------------------------
# MODEL
# -----------------------------
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base.trainable = False

x = data_aug(base.input)
x = base(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)

# 🔥 STRONG CLASSIFIER
x = tf.keras.layers.Dense(512, activation="relu")(x)
x = tf.keras.layers.Dropout(0.5)(x)

x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.4)(x)

x = tf.keras.layers.Dense(128, activation="relu")(x)

output = tf.keras.layers.Dense(4, activation="softmax")(x)

model = tf.keras.Model(base.input, output)

# 🔥 LABEL SMOOTHING (VERY IMPORTANT)
loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=loss,
    metrics=["accuracy"]
)

# -----------------------------
# CALLBACKS
# -----------------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=4,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    patience=2,
    factor=0.3
)

# -----------------------------
# TRAIN
# -----------------------------
print("Training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=35,
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

# -----------------------------
# FINE TUNING
# -----------------------------
for layer in base.layers[-25:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=loss,
    metrics=["accuracy"]
)

print("Fine tuning...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    class_weight=class_weights,
    callbacks=[early_stop]
)

model.save("models/detection_model.h5")

print("✅ FINAL ALL-CLASS MODEL READY")