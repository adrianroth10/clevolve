#!/usr/bin/env python3
import cv2
import glob
import os.path as osp
import matplotlib.pyplot as plt
import numpy as np
import sys

def transform_images(hs=None):
    if hs is None:
        pts_dst = points[0]
        hs = np.zeros((3 * len(images), 3))
        hs[0:3, :] = np.eye(3)
        for i in range(1, len(images)):
            h, status = cv2.findHomography(points[i], pts_dst)
            hs[3*i:3*(i + 1), :] = h

        np.savetxt(osp.join(folder, 'homographies.txt'), hs)

    shape_dst = loaded_images[0].shape
    for i in range(len(images)):
        h = hs[3*i:3*(i+1), :]
        im_out = cv2.warpPerspective(loaded_images[i], h, (shape_dst[1], shape_dst[0]))
        directory, filename = osp.split(images[i])
        file_base = osp.splitext(filename)[0]
        new_name = osp.join(directory, 'transformed_' + file_base + '.png')
        cv2.imwrite(new_name, im_out)

def onclick(event):
    global i, j
    ix, iy = event.xdata, event.ydata
    points[i, j, 0] = ix
    points[i, j, 1] = iy

    i += 1
    if i == len(images):
        i = 0
        j += 1
    if j == n_points:
        fig.canvas.mpl_disconnect(cid)
        plt.close()
        transform_images()
    else:
        ax.imshow(loaded_images[i])
        if j > 0:
            ax.plot(points[i, j - 1, 0], points[i, j - 1, 1], '.', linewidth=7)
        fig.canvas.draw()


if __name__ == "__main__":
    folder = sys.argv[-1]
    images = []
    images += glob.glob(osp.join(folder, '*.jpg'))
    images += glob.glob(osp.join(folder, '*.png'))
    images = [im for im in images if 'transformed' not in im]
    images = sorted(images)

    loaded_images = None
    for i, im in enumerate(images):
        load_im = cv2.imread(im)
        if loaded_images is None:
            loaded_images = np.zeros((len(images),) + load_im.shape, dtype=np.uint8)
        loaded_images[i] = load_im

    n_points = 4
    points = np.zeros((len(images), n_points, 2))

    hs_file = osp.join(folder, 'homographies.txt')
    if osp.isfile(hs_file):
        hs = np.loadtxt(hs_file)
        transform_images(hs)
    else:
        i = 0
        j = 0

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.imshow(loaded_images[0])
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
