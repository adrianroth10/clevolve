#!/usr/bin/env python3
import cv2
import glob
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import sys

def transform_images():
    pts_dst = points[0]
    shape_dst = loaded_images[0].shape
    for i in range(1, len(images)):
        h, status = cv2.findHomography(points[i], pts_dst)
        print(i)
        print(h)
        im_out = cv2.warpPerspective(loaded_images[i], h, (shape_dst[1], shape_dst[0]))

        directory, filename = osp.split(images[i])
        cv2.imwrite(osp.join(directory, 'transformed_' + filename), im_out)

def onclick(event):
    global i, j
    ix, iy = event.xdata, event.ydata
    points[i, j, 0] = iy
    points[i, j, 1] = ix

    i += 1
    if i == len(images):
        i = 0
        j += 1
    if j == n_points:
        fig.canvas.mpl_disconnect(cid)
        plt.close()
        transform_images()

    ax.imshow(loaded_images[i])
    if j > 0:
        ax.plot(points[i, j - 1, 1], points[i, j - 1, 0], '.', linewidth=7)
    fig.canvas.draw()


if __name__ == "__main__":
    folder = sys.argv[-1]
    images = []
    images += glob.glob(osp.join(folder, '*.jpg'))
    images += glob.glob(osp.join(folder, '*.png'))
    images = images[:2]

    loaded_images = None
    for i, im in enumerate(images):
        load_im = cv2.imread(im)
        if loaded_images is None:
            loaded_images = np.zeros((len(images),) + load_im.shape, dtype=int)
        loaded_images[i] = load_im
        
    n_points = 4
    points = np.zeros((len(images), n_points, 2))

    i = 0
    j = 0

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(loaded_images[0])
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
