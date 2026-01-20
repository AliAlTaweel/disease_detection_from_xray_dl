import tensorflow as tf

# Check for Apple Silicon GPU (MPS)
devices = tf.config.list_physical_devices()
print(f"Devices found: {devices}")


keras = tf.keras
layers = tf.keras.layers
# Constants
TRAIN_DIR = "data/chest_xray/train"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data Loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="binary"
)

# Model Definition
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
base_model.trainable = False

model = keras.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./127.5, offset=-1),
    layers.RandomFlip("horizontal"),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Training
print("Starting training...")
model.fit(train_ds, validation_data=val_ds, epochs=5) # Reduced epochs for testing

# Save the model
model.save("model/pneumonia_model.h5")
print("Model saved to model/pneumonia_model.h5")