import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tif
from sklearn.model_selection import train_test_split
import cv2
from EDA import load_data



'''
def rle_to_mask(rle, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    rle_data = np.array([int(x) for x in rle.split()]).reshape(-1, 2)
    for start, length in rle_data:
        mask.ravel()[start:start + length] = 1
    return mask
'''


def rle_to_mask(rle, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    rle_data = np.array([int(x) for x in rle.split()])
    for i in range(0, len(rle_data), 2):
        start = rle_data[i] - 1
        length = rle_data[i+1]
        end = start + length
        mask.ravel()[start:end] = 1
    return mask


def preprocess_data(surface_volume, mask, inklabels, inklabels_mask, ir, apply_augmentation=True):
    # Resize the data
    new_shape = (128, 128, 64)
    surface_volume_resized = cv2.resize(surface_volume, new_shape[:2], interpolation=cv2.INTER_LINEAR)
    mask_resized = cv2.resize(mask, new_shape[:2], interpolation=cv2.INTER_NEAREST)
    inklabels_resized = cv2.resize(inklabels, new_shape[:2], interpolation=cv2.INTER_NEAREST)
    print("preproccess:",inklabels_mask)
    inklabels_mask_resized = cv2.resize(inklabels_mask, new_shape[:2], interpolation=cv2.INTER_NEAREST)
    ir_resized = cv2.resize(ir, new_shape[:2], interpolation=cv2.INTER_NEAREST)

    # Normalize the data by dividing by the maximum value (255)
    surface_volume_normalized = surface_volume_resized / 255.0

    # Apply data augmentation if needed
    if apply_augmentation:
        # Random rotation (0, 90, 180, or 270 degrees)
        k = np.random.randint(0, 4)
        surface_volume_normalized = np.rot90(surface_volume_normalized, k)
        mask_resized = np.rot90(mask_resized, k)
        inklabels_resized = np.rot90(inklabels_resized, k)
        inklabels_mask_resized = np.rot90(inklabels_mask_resized, k)
        ir_resized = np.rot90(ir_resized, k)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            surface_volume_normalized = np.fliplr(surface_volume_normalized)
            mask_resized = np.fliplr(mask_resized)
            inklabels_resized = np.fliplr(inklabels_resized)
            inklabels_mask_resized = np.fliplr(inklabels_mask_resized)
            ir_resized = np.fliplr(ir_resized)

    return surface_volume_normalized, mask_resized, inklabels_resized, inklabels_mask_resized, ir_resized


if __name__ == "__main__":
    # Set up parameters
    base_path = "/Users/ahmed/Desktop/Kaggle Volcano Text/vesuvius-challenge-ink-detection/train"
    fragment_ids = [str(i) for i in range(1, 4)]
    apply_augmentation = True

    # Load and preprocess data
    X = []
    y = []
    for fragment_id in fragment_ids:
        surface_volume, mask, inklabels, inklabels_mask, ir = load_data(base_path, fragment_id)
        surface_volume_preprocessed, mask_preprocessed, inklabels_preprocessed, inklabels_mask_preprocessed, ir_preprocessed = preprocess_data(
            surface_volume, mask, inklabels, inklabels_mask, ir, apply_augmentation
        )
        X.append(surface_volume_preprocessed)
        y.append(inklabels_preprocessed)

    X = np.stack(X, axis=0)
    y = np.stack(y, axis=0)


    # Save preprocessed data to file
    np.savez_compressed("preprocessed_data.npz", X=X, y=y)




