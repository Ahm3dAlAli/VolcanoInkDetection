import os
import numpy as np
import pandas as pd
from EDA import load_data
from Pre_Proccessing import preprocess_data
from keras.models import load_model
from skimage import morphology
from Pre_Proccessing import rle_to_mask

def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

# Load the tuned model
model = load_model("tuned_model.h5")

# Load and preprocess test data
base_path = "test"
fragment_ids = ["a", "b"]
X_test = []

for fragment_id in fragment_ids:
    surface_volume, mask = load_data(base_path, fragment_id, is_train=False)
    surface_volume_preprocessed, mask_preprocessed = preprocess_data(surface_volume, mask)
    X_test.append(surface_volume_preprocessed)

# Generate predictions for the test set
y_test_pred = model.predict(X_test)

# Apply post-processing techniques
y_test_pred = np.round(y_test_pred)
y_test_pred = [morphology.remove_small_objects(pred, min_size=100) for pred in y_test_pred]

# Generate the submission file
submission = []

for i, pred in enumerate(y_test_pred):
    rle = rle_encoding(pred)
    rle_str = ' '.join(map(str, rle))
    submission.append([i + 1, rle_str])

submission_df = pd.DataFrame(submission, columns=["Id", "Predicted"])
submission_df.to_csv("submission.csv", index=False)

print("Submission file generated: submission.csv")

# Visualize the predicted masks

# Convert RLE-encoded predictions to binary masks
mask_preds = []
for rle in submission_df["Predicted"]:
    mask_pred = rle_to_mask(rle, shape=(128, 128, 65))
    mask_preds.append(mask_pred)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=len(mask_preds), ncols=2, figsize=(8, 16))

for i, (mask_pred, ax) in enumerate(zip(mask_preds, axes)):
    ax[0].imshow(X_test[i, :, :, 32, 0], cmap="gray")
    ax[1].imshow(mask_pred[:, :, 32], cmap="gray")
    ax[0].set_title(f"Input volume {i+1}")
    ax[1].set_title(f"Predicted mask {i+1}")

plt.tight_layout()
plt.show()
