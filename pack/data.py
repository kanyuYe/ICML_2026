from .config import project_root  

import csv
import os

import numpy as np

from .crypto import encrypt_data
from .utils import judge


def read_image(batch_size):
    filePath = os.path.join(project_root, "PackCNN", "data", "test_batch.bin")
    IMAGE_SIZE = 3072
    LABEL_SIZE = 1
    RECORD_SIZE = LABEL_SIZE + IMAGE_SIZE
    batch_image = np.zeros((batch_size, IMAGE_SIZE))
    imageVector = np.zeros((batch_size, IMAGE_SIZE))
    batch_label = np.zeros(batch_size)
    try:
        with open(filePath, "rb") as file:
            for index in range(batch_size):
                file.seek(index * RECORD_SIZE)
                label = file.read(LABEL_SIZE)
                if not label:
                    raise ValueError("Failed to read label.")
                batch_label[index] = int.from_bytes(label, byteorder="big")
                image_data = file.read(IMAGE_SIZE)
                batch_image[index] = np.frombuffer(image_data, dtype=np.uint8)
                for channel in range(3):
                    for i in range(1024):
                        pixel = float(batch_image[index][channel * 1024 + i]) / 255.0
                        if channel == 0:
                            pixel = (pixel - 0.4914) / 0.2023
                        elif channel == 1:
                            pixel = (pixel - 0.4822) / 0.1994
                        elif channel == 2:
                            pixel = (pixel - 0.4465) / 0.2010
                        imageVector[index][channel * 1024 + i] = pixel
        return imageVector, batch_label
    except FileNotFoundError:
        print(f"Failed to open the file: {filePath}")


def batch_input(mode1, batch_size, openfhe_context, cryptoContext):
    slots = 2 ** 15
    DATA_DIR = os.environ["DATA_DIR"]
    dir_path = DATA_DIR
    # dir_path =os.path.join(project_root, "PackCNN", "data")

    group_num = 8
    input = np.zeros((group_num, 3, 3, 16384 * 2))
    imageVector, batch_label = read_image(batch_size)
    imageVector1 = np.pad(imageVector.reshape(batch_size, 3, 32, 32),
                          ((0, 0), (0, 0), (0, 1), (0, 1)),
                          mode='constant', constant_values=0).reshape(batch_size, -1)
    output_channel = 16
    input_channel = 3
    channel_batch = output_channel * batch_size
    for c in range(input_channel):
        for g in range(group_num):
            for wi in range(3):
                for wo in range(3):
                    for i in range(batch_size):
                        temp_couter = 0
                        for j in range(16):
                            if judge(group_num, g, j):
                                if j * 24 + 33 * 2 * temp_couter + wi + wo * 33 + 3 * g >= 33 * 33:
                                    input[g][wo][wi][j * batch_size + i + c * channel_batch] = 0
                                else:
                                    input[g][wo][wi][j * batch_size + i + c * channel_batch] = imageVector1[i][
                                        j * 24 + 33 * 2 * temp_couter + wi + wo * 33 + c * 33 * 33 + 3 * g]
                            else:
                                temp_couter += 1
                                if j * 24 + 33 * 2 * temp_couter + wi + wo * 33 + 3 * g >= 33 * 33:
                                    input[g][wo][wi][j * batch_size + i + c * channel_batch] = 0
                                else:
                                    input[g][wo][wi][j * batch_size + i + c * channel_batch] = imageVector1[i][
                                        j * 24 + 33 * 2 * temp_couter + wi + wo * 33 + c * 33 * 33 + 3 * g]
    for g in range(group_num):
        for wi in range(3):
            for wo in range(3):
                input[g][wi][wo][channel_batch * input_channel:channel_batch * (input_channel + 1)] = 0
    q = input.shape[-1] // 4
    blk = input[..., :q].copy()
    input[...] = np.tile(blk, (1, 1, 1, 4))
    input_encrypt = encrypt_data(mode1, input, dir_path, slots, openfhe_context, cryptoContext)
    return input_encrypt, batch_label
