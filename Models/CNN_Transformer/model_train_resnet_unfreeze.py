from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
import pickle
import tensorflow as tf
import cv2
import numpy as np


# Mannual padding so the image wont get distorted
def pad_image(img):
    target_size = (224, 224)

    h, w, _ = img.shape

    pad_h = max(0, target_size[0] - h)
    pad_w = max(0, target_size[1] - w)

    if pad_h > 0 or pad_w > 0:
        top_pad = pad_h // 2
        bottom_pad = pad_h - top_pad
        left_pad = pad_w // 2
        right_pad = pad_w - left_pad

        img = np.pad(
            img,
            ((top_pad, bottom_pad),
             (left_pad, right_pad),
             (0, 0)),
            mode='constant',
            constant_values=0
        )

    if img.shape[0] != target_size[0] or img.shape[1] != target_size[1]:
        img = cv2.resize(img, target_size)

    return img


# ImageDataGenerator for data augmentation and normalization
datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2,
    preprocessing_function=pad_image
)

input_shape = (224, 224, 3)

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=input_shape
)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

for layer in base_model.layers[-15:]:
    layer.trainable = True

# Add custom layers on top
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
predictions = Dense(36, activation='softmax')(x)

# Construct the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=5e-5),
              loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True)

train_generator = datagen.flow_from_directory(
    'output-train-by-char',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=64,
    class_mode='categorical',
    subset='training',
    shuffle=True)

val_generator = datagen.flow_from_directory(
    'output-train-by-char',
    target_size=(224, 224),
    color_mode='rgb',
    batch_size=64,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_resnet-50-unfreeze-15.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='min',
    verbose=1
)
# Train the model with data augmentation and early stopping
history = model.fit(
    train_generator,
    epochs=150,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint]
)

with open('history_resnet50-unfreeze-15.pkl', 'wb') as f:
    pickle.dump(history.history, f)

model.save('resnet50_model-unfreeze-15-backup.keras')
