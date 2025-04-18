import tensorflow as tf
import pickle
import cv2
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, Checkpoint, ReduceLROnPlateau
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2


def pad_image(img):
    """Pad image to target size (24, 24)"""
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


def residual_conv_block(x, filters):
    """Residual convolution block for CNN backbone"""
    shortcut = x

    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters, 3, padding='same',
                      kernel_regularizer=l2(2e-4))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Add()([x, shortcut])
    x = layers.Activation('relu')(x)

    return x


def transformer_block(x, num_heads, ff_dim, dropout_rate=0.3):
    attn_input = layers.LayerNormalization(epsilon=1e-6)(x)

    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads, key_dim=x.shape[-1] // num_heads)(attn_input, attn_input)
    attn_output = layers.Dropout(dropout_rate)(attn_output)

    x = layers.Add()([x, attn_output])
    ff_input = layers.LayerNormalization(epsilon=1e-6)(x)

    ffn = layers.Dense(ff_dim, kernel_regularizer=l2(2e-4))(ff_input)
    ffn = layers.Activation('gelu')(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(x.shape[-1])(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = layers.Add()([x, ffn])

    return x


def build_improved_transformer_model(input_shape=(24, 24, 3), num_classes=36):
    inputs = tf.keras.Input(shape=input_shape)

    # CNN backbone with residual connections
    x = layers.Conv2D(32, 3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = residual_conv_block(x, 64)
    x = layers.MaxPooling2D()(x)

    x = residual_conv_block(x, 128)
    x = layers.MaxPooling2D()(x)

    x = residual_conv_block(x, 256)
    x = layers.MaxPooling2D()(x)

    # Reshape for transformer input
    h, w = x.shape[1], x.shape[2]
    seq_len = h * w
    embed_dim = x.shape[-1]
    x = layers.Reshape((seq_len, embed_dim))(x)
    # Add positional encoding
    pos_emb = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)(
        tf.range(start=0, limit=seq_len, delta=1))
    x = x + pos_emb
    # Transformer blocks
    for i in range(4):
        x = transformer_block(
            x,
            num_heads=8,
            ff_dim=embed_dim * 4,
            dropout_rate=0.1 + (i * 0.05)
        )

    # Final normalization and pooling
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


# Build the model
model = build_improved_transformer_model(
    input_shape=(24, 24, 3), num_classes=36)

# Load and augment the data
datagen = ImageDataGenerator(
    rescale=1./255,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.2,
    fill_mode='nearest',
    validation_split=0.2,
    preprocessing_function=pad_image
)

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
# Compile and set up the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(
    monitor='val_accuracy', patience=5, restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=3, min_lr=1e-6)

checkpoint = ModelCheckpoint(
    'best_model_cnn-transformer-rd5.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

history = model.fit(
    train_generator,
    epochs=150,
    validation_data=val_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr],
)

with open('history_cnn-transformer-rd5.pkl', 'wb') as f:
    pickle.dump(history.history, f)

model.save('cnn_transformer-rd5.keras')
