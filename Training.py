import os
import numpy as np
from Model_Architecture import unet_3d
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
# Set training parameters
batch_sizes = [2, 4, 8]
num_epochs = [50, 100, 150]
learning_rates = [1e-4, 1e-3, 1e-2]
validation_split = 0.2

# Load preprocessed data
X,y = np.load('preprocessed_data/X.npy')
y = np.load('preprocessed_data/y.npy')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, random_state=42)

best_val_loss = np.inf
best_model = None

# Hyperparameter tuning
for batch_size in batch_sizes:
    for epochs in num_epochs:
        for lr in learning_rates:
            # Create the 3D U-Net model
            model = unet_3d(input_shape=(128, 128, 65, 1), num_classes=1)
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

            # Set up callbacks
            checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_loss', save_best_only=True, save_weights_only=True)
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
            callbacks = [checkpoint, early_stopping, reduce_lr]

            # Train the model
            history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=callbacks, verbose=0)

            # Evaluate the model on the validation set
            y_val_pred = model.predict(X_val)
            y_val_pred = np.round(y_val_pred)

            # Calculate evaluation metrics
            val_loss = history.history['val_loss'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            val_precision = precision_score(y_val.flatten(), y_val_pred.flatten())
            val_recall = recall_score(y_val.flatten(), y_val_pred.flatten())
            val_f1 = f1_score(y_val.flatten(), y_val_pred.flatten())

            # Print evaluation results
            print(f"Batch size: {batch_size}, Epochs: {epochs}, Learning rate: {lr}")
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            print(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}, Validation F1-score: {val_f1:.4f}")

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_model.save('tuned_model.h5')

# Print the best hyperparameters
print(f"\nBest hyperparameters: Batch size: {batch_size}, Epochs: {epochs}, Learning rate: {lr}")
