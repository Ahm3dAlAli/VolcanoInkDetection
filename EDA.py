import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
import imageio
import tifffile 

def rle_to_mask(rle, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    rle_data = np.array([int(x) for x in rle.split()])
    for i in range(0, len(rle_data), 2):
        start = rle_data[i] - 1
        length = rle_data[i+1]
        end = start + length
        mask.ravel()[start:end] = 1
    return mask


def load_data(base_path, fragment_id, is_train=True):
    fragment_path = os.path.join(base_path, str(fragment_id))
    
    # Load surface_volume
    surface_volume_path = os.path.join(fragment_path, "surface_volume")
    surface_volume_files = sorted(os.listdir(surface_volume_path))
    surface_volume_files=surface_volume_files[1:]
    surface_volume = [imread(os.path.join(surface_volume_path, f)) for f in surface_volume_files]
    surface_volume = np.stack(surface_volume, axis=-1)

    # Load mask
    mask = imread(os.path.join(fragment_path, "mask.png"))

    if is_train:
        # Load inklabels, inklabels_rle, and ir for train data
        inklabels = imread(os.path.join(fragment_path, "inklabels.png"))
        inklabels_rle = np.loadtxt(os.path.join(fragment_path, "inklabels_rle.csv"), delimiter=',', dtype=str).reshape(-1, 1)
        inklabels_rle=inklabels_rle[3]
        inklabels_mask = [rle_to_mask(rle, inklabels.shape[:2]) for rle in inklabels_rle]
        ir = imread(os.path.join(fragment_path, "ir.png"))
        return surface_volume, mask, inklabels, inklabels_mask, ir
    else:
        return surface_volume, mask


def visualize_data(xray_slices, ir_image, ink_mask):
    # Visualize a sample X-ray slice
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(xray_slices[:, :, 32], cmap='gray')
    plt.title("X-ray Slice")

    # Visualize the infrared image
    plt.subplot(1, 3, 2)
    plt.imshow(ir_image, cmap='gray')
    plt.title("Infrared Image")

    # Visualize the ink mask
    plt.subplot(1, 3, 3)
    plt.imshow(ink_mask, cmap='gray')
    plt.title("Ink Mask")

    plt.show()

def analyze_data(base_path):
    bp=base_path
    # Calculate some statistics and visualize the data
    train_fragment_ids = [1]#,2,3]
    num_fragments = len(train_fragment_ids)

    for fragment_id in train_fragment_ids:

        surface_volume, mask, inklabels, inklabels_mask, ir = load_data(bp, fragment_id)
        
        # Calculate ink coverage
        ink_coverage = np.sum(inklabels) / np.sum(mask)
        print(f"Ink coverage for fragment {fragment_id}: {ink_coverage:.2%}")

        # Visualize the data
        for ink_mask in inklabels_mask:
            visualize_data(surface_volume, ir, ink_mask)

if __name__ == "__main__":
    train_base_path = "/Users/ahmed/Desktop/Kaggle Volcano Text/vesuvius-challenge-ink-detection/train/"
    analyze_data(train_base_path)
