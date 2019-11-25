# -*- coding: utf-8 -*-
"""
@created on: 11/24/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import cv2
import glob
import numpy as np
import os

sizes = set()


def resize_images_and_save(image_folder, results_csv, label, width_list=(1024, 512, 256, 768), variable=False):
    result_path = '/Users/badgod/badgod_documents/ImageDataset/same_size_input/images'
    for i, image_path in enumerate(glob.glob(image_folder + '/*')):

        image = cv2.imread(image_path)
        if image is None:
            continue
        if variable:
            new_width = np.random.choice(width_list)
            image_height = image.shape[0]
            image_width = image.shape[1]
            aspect_ratio = new_width / image_width
            new_image_width = int(image_width * aspect_ratio)
            new_image_height = int(image_height * aspect_ratio)
            resized_image = cv2.resize(image, (new_image_width, new_image_height))
        else:
            new_width = 320
            new_height = 320
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            sizes.add(resized_image.shape)
        new_image_path = result_path + '/' + image_path.split('/')[-1]
        cv2.imwrite(new_image_path, resized_image)
        print(new_image_path, ',', label, file=results_csv)
        print('Done ', i)


csv_results_path = '/Users/badgod/badgod_documents/ImageDataset/same_size_input/csv/result.csv'
if os.path.exists(csv_results_path):
    csv_results_file = open(csv_results_path, 'a')
else:
    csv_results_file = open(csv_results_path, 'w')
    print('Images', ',', 'Label', file=csv_results_file)

base_folder = '/Users/badgod/Downloads/flowers/'
folders = ['dandelion', 'rose', 'tulip', 'daisy', 'sunflower']
for folder in folders:
    print('-------', folder, '--------')
    image_folder = base_folder + folder
    resize_images_and_save(image_folder, csv_results_file, folder)

print(sizes)