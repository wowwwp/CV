import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Dropout, Multiply
from tensorflow.keras.models import Model
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os
from tensorflow.keras.optimizers import Adam


def pad_image(img):
    target_size = (24, 24)
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
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    shear_range=0.08,
    brightness_range=[0.85, 1.15],
    fill_mode='nearest',
    validation_split=0.2,
    preprocessing_function=pad_image
)


def residual_block(x, filters):
    shortcut = x
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters, (3, 3), padding='same')(x)

    if shortcut.shape[-1] != filters:
        shortcut = BatchNormalization()(shortcut)
        shortcut = ReLU()(shortcut)
        shortcut = Conv2D(filters, (1, 1), padding='same')(shortcut)

    return Add()([x, shortcut])


def build_model(input_shape=(24, 24, 3), num_classes=36, dropout_rate=0.35):
    # Input Layer
    input_tensor = Input(shape=input_shape)

    # Initial Convolutional Layer
    x = Conv2D(64, (3, 3), padding='same', strides=1)(input_tensor)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128, (3, 3), padding='same', strides=1,
               kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    for i in range(10):
        x = residual_block(x, 64)

    x = GlobalAveragePooling2D()(x)

    x = Dropout(dropout_rate)(x)

    output_tensor = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model


input_shape = (24, 24, 3)

train_generator = datagen.flow_from_directory(
    'output-train-by-char',
    target_size=(24, 24),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True)

val_generator = datagen.flow_from_directory(
    'output-train-by-char',
    target_size=(24, 24),
    color_mode='rgb',
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_model_resnet-50-5-rd10.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Build the model
model = build_model(input_shape, num_classes=36)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                                 patience=5, min_lr=0.001)

model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=150,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

with open('history_resnet50-5-rd10.pkl', 'wb') as f:
    pickle.dump(history.history, f)

model.save('resnet50_model-5-backup-rd10.keras')
