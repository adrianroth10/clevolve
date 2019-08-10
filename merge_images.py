#!/usr/bin/env python3
import cv2
import glob
import os.path as osp
import numpy as np
import sys

def merge_images():
    segmented_images[0] = np.ones(segmented_images[0].shape, dtype=bool)
    for i in range(1, len(transformed_image_files)):
        transformed_images[i][~segmented_images[i]] = 0
        segmented_images[0] = segmented_images[0] & ~segmented_images[i]
    transformed_images[0][~segmented_images[0]] = 0
    
    scaler = np.sum(segmented_images, axis=0)
    scaler = np.repeat(scaler, 3).reshape(scaler.shape + (3,))
    merged_image = np.sum(transformed_images, axis=0) / scaler
    cv2.imwrite(osp.join(folder, 'merged.png'), merged_image)


if __name__ == "__main__":
    folder = sys.argv[-1]
    transformed_image_files = glob.glob(osp.join(folder, 'transformed_*.png'))
    segmented_image_files = glob.glob(osp.join(folder, 'segmented_*.png'))
    transformed_image_files = sorted(transformed_image_files)
    segmented_image_files = sorted(segmented_image_files)

    
    transformed_images = None
    segmented_images = None
    for i in range(len(transformed_image_files)):
        transformed_image = cv2.imread(transformed_image_files[i])
        segmented_image = cv2.imread(segmented_image_files[i], 0)
        if transformed_images is None:
            transformed_images = np.zeros((len(transformed_image_files),) + transformed_image.shape, dtype=np.uint8)
            segmented_images = np.zeros((len(segmented_image_files),) + segmented_image.shape, dtype=bool)
        transformed_images[i] = transformed_image
        segmented_images[i] = segmented_image == 255

    merge_images()
