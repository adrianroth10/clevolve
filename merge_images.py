#!/usr/bin/env python3
import cv2
import glob
import os.path as osp
# import matplotlib.pyplot as plt
import numpy as np
import sys
import arpack.cxx_wrapper as cxx

def scale_merge_print(background_image, segmented_images, transformed_images, filename):
    scaler = np.sum(segmented_images, axis=0)
    rest = 1 - scaler
    rest[rest < 0] = 0
    background_image = background_image * np.repeat(rest, 3).reshape(transformed_image.shape)
    scaler = np.repeat(scaler + rest, 3).reshape(scaler.shape + (3,))
    merged_image = ((np.sum(transformed_images, axis=0) + background_image) / scaler).astype(np.uint8)
    cv2.imwrite(osp.join(folder, filename), merged_image)

def sigmoid_it(segmented_image, transformed_image=None):
    dm = cxx.depth_map(~(segmented_image > 0.5))
    outside = dm < -5
    inside = dm > 5
    dm[outside] = 0
    sigmoid = (1 / (1 + np.exp(-dm / 2)))
    sigmoid[outside] = 0
    sigmoid[inside] = 1
    if transformed_image is None:
        return sigmoid
    else:
        return [sigmoid,
                (transformed_image * np.repeat(sigmoid, 3).reshape(transformed_image.shape))]

def forward_nand_merge(segmented_images, transformed_images):
    segmented_images = np.array(segmented_images, dtype=float)
    transformed_images = np.array(transformed_images, dtype=float)
    background_image = transformed_images[0].copy()
    cumulative_segmentation = np.zeros(segmented_images[0].shape)
    for i in range(len(transformed_image_files)):
        print(i)
        segmented_images[i] = segmented_images[i] - cumulative_segmentation
        segmented_images[i][segmented_images[i] < 0] = 0
        segmented_images[i], transformed_images[i] = sigmoid_it(segmented_images[i], transformed_images[i])
        cumulative_segmentation = segmented_images[i] + cumulative_segmentation
    
    scale_merge_print(background_image, segmented_images, transformed_images, 'merge_forward_nand.png')

def simple_merge(segmented_images, transformed_images):
    background_image = transformed_images[0].copy()
    first_seg = segmented_images[0].copy()
    segmented_images[0] = np.ones(segmented_images[0].shape, dtype=bool)
    for i in range(1, len(transformed_image_files)):
        transformed_images[i][~segmented_images[i]] = 0
        segmented_images[0] = segmented_images[0] & ~segmented_images[i]
    segmented_images[0] = segmented_images[0] | first_seg
    transformed_images[0][~segmented_images[0]] = 0
    
    scale_merge_print(background_image, segmented_images, transformed_images, 'merge_simple.png')


def merge_images(ty, segmented_images, transformed_images):
    if ty == 0:
        simple_merge(segmented_images, transformed_images)
    elif ty == 1:
        forward_nand_merge(segmented_images, transformed_images)

if __name__ == "__main__":
    if len(sys.argv) > 2:
        folder = sys.argv[-2]
        ty = int(sys.argv[-1])
    else:
        folder = sys.argv[-1]
        ty = 1
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

    merge_images(ty, segmented_images, transformed_images)
