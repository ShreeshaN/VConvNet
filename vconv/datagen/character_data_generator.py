# -*- coding: utf-8 -*-
"""
@created on: 11/27/19,
@author: Shreesha N,
@version: v0.0.1
@system name: badgod
Description:

..todo::

"""

import random
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from fontTools.ttLib import TTFont
import numpy as np
import os
import PIL
import uuid

font_path = "/Users/badgod/badgod_documents/Softwares/open-sans_fonts/"
fonts = ["OpenSans-BoldItalic.ttf",
         "OpenSans-ExtraBold.ttf",
         "OpenSans-ExtraBoldItalic.ttf",
         "OpenSans-Italic.ttf",
         "OpenSans-Light.ttf",
         "OpenSans-LightItalic.ttf",
         "OpenSans-Regular.ttf",
         "OpenSans-Semibold.ttf",
         "OpenSans-SemiboldItalic.ttf"]
characters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
font_size = 100
background = 'white'
image_width = 128
image_height = 128

font_size_for_variable_images = [32, 64, 128]
image_width_variable_images = [128, 256, 344]
image_height_variable_images = [96, 192, 258]  # maintaining 4:3 aspect ratio
basepath = "/Users/badgod/badgod_documents/Datasets/CharacterDataset/variable_size/"
image_savepath = basepath + "images/"
csv_filepath = basepath + "csv/result.csv"
csv_file = open(csv_filepath, "w")
number_of_samples = 10000
train_split = 0.8
header = "Image_name,Label"


def generate(variable=False):
    for i in range(number_of_samples):
        if i == 0:
            print(header, file=csv_file)
        if i % 200 == 0:
            print('Done - ', i, '/', number_of_samples)
        font = random.choice(fonts)
        character = random.choice(characters)
        if variable:
            font_index = random.choice(range(len(font_size_for_variable_images)))
            image_size_index = random.choice(range(len(image_width_variable_images)))
            current_font_size = font_size_for_variable_images[font_index]
            current_image_height = image_height_variable_images[image_size_index]
            current_image_width = image_width_variable_images[image_size_index]

            image = create_image_for_text(character, font_path + font, current_font_size, background,
                                          current_image_height, current_image_width)
            image = image.resize((current_image_width, current_image_height), PIL.Image.ANTIALIAS)
        else:
            image = create_image_for_text(character, font_path + font, font_size, background, image_height, image_width)
            image = image.resize((image_width, image_height), PIL.Image.ANTIALIAS)
        image_name = uuid.uuid4().hex + '.jpg'
        image.save(image_savepath + image_name)
        str_row = image_name + ',' + character
        print(str_row, file=csv_file)


def create_image_for_text(text, font, font_size, background, image_height, image_width):
    gen_draw = ImageDraw.Draw(Image.fromarray(np.zeros((1, 1), dtype=np.uint8)))
    font = ImageFont.truetype(font, font_size)
    w, h = gen_draw.textsize(text, font=font)
    H = image_height
    W = w + image_width
    image = Image.new("RGB", (W, H), background)
    draw = ImageDraw.Draw(image)
    draw.text(((W - w) // 2, (H - h) // 2), text, font=font, fill=(0, 0, 0))
    return image


def create_train_test_split():
    data = pd.read_csv(csv_filepath)
    data = data.sample(frac=1)
    tr = data[:int(len(data) * train_split)]
    te = data[-int(len(data) * (1 - train_split)):]
    path = '/'.join(csv_filepath.split('/')[:-1])
    tr.to_csv(path + '/train.csv', index=False)
    te.to_csv(path + '/test.csv', index=False)


if __name__ == '__main__':
    print('Generating character data from a total of ', len(characters), 'classes')
    generate(variable=True)
    print("Done generating data")
    csv_file.close()
    print('Creating train,test splits')
    create_train_test_split()
    print('Data ready to use. Saved in folder ', csv_filepath)
