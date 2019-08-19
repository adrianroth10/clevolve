#!/usr/bin/env python3
import cv2
import glob
import os.path as osp
import numpy as np
import sys

def scale_merge_print(segmented_images, filename):
    scaler = np.sum(segmented_images, axis=0)
    scaler = np.repeat(scaler, 3).reshape(scaler.shape + (3,))
    merged_image = np.sum(transformed_images, axis=0) / scaler
    cv2.imwrite(osp.join(folder, filename), merged_image)


def backward_nand_merge():
    first_seg = segmented_images[0].copy()
    cumulative_segmentation = np.zeros(segmented_image[0].shape, dtype=bool)
    segmented_images[0] = np.ones(segmented_images[0].shape, dtype=bool)
    for i in range(len(transformed_image_files) -1, 0, -1):
        segmented_images[i] = segmented_images[i] & ~cumulative_segmentation
        transformed_images[i][~segmented_images[i]] = 0
        segmented_images[0] = segmented_images[0] & ~segmented_images[i]
        cumulative_segmentation = segmented_images[i] | cumulative_segmentation
    first_seg = first_seg & ~cumulative_segmentation
    transformed_images[0][~segmented_images[0]] = 0
    
    scale_merge_print(segmented_images, 'merge_backward_nand.png')

def forward_nand_merge():
    cumulative_segmentation = segmented_images[0].copy()
    segmented_images[0] = np.ones(segmented_images[0].shape, dtype=bool)
    for i in range(1, len(transformed_image_files)):
        segmented_images[i] = segmented_images[i] & ~cumulative_segmentation
        transformed_images[i][~segmented_images[i]] = 0
        segmented_images[0] = segmented_images[0] & ~segmented_images[i]
        cumulative_segmentation = segmented_images[i] | cumulative_segmentation
    transformed_images[0][~segmented_images[0]] = 0
    
    scale_merge_print(segmented_images, 'merge_forward_nand.png')

def simple_merge():
    first_seg = segmented_images[0].copy()
    segmented_images[0] = np.ones(segmented_images[0].shape, dtype=bool)
    for i in range(1, len(transformed_image_files)):
        transformed_images[i][~segmented_images[i]] = 0
        segmented_images[0] = segmented_images[0] & ~segmented_images[i]
    segmented_images[0] = segmented_images[0] | first_seg
    transformed_images[0][~segmented_images[0]] = 0
    
    scale_merge_print(segmented_images, 'merge_simple.png')


def merge_images(ty=0):
    if ty == 0:
        simple_merge()
    elif ty == 1:
        forward_nand_merge()
    elif ty == 2:
        backward_nand_merge()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        folder = sys.argv[-2]
        ty = int(sys.argv[-1])
    else:
        folder = sys.argv[-1]
        ty = 0
    transformed_image_files = glob.glob(osp.join(folder, 'transformed_*.png'))
    segmented_image_files = glob.glob(osp.join(folder, 'segmented_transformed_*.png'))
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

    merge_images(ty)
