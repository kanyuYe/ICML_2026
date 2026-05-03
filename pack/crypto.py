"""Encryption and homomorphic activation helpers."""

from .config import project_root  # Ensures runtime paths and DATA_DIR match the original script.

import time

import numpy as np
from examples.utils import approx
import torch.fhe as fhe

from .encoding import read_values_from_file, use_checkpoint


def encrypt_data(mode, input_data, dir_path, slots, openfhe_context, cryptoContext):
    cipher_list = None
    if mode in ["run", "save", "verify"]:
        print("reading input data...")
        cipher_list = np.empty((input_data.shape[0], input_data.shape[1], input_data.shape[2]), dtype=object)
        print("begin encrypting data...")
        begin_time = time.time()
        for t in range(input_data.shape[0]):
            for i in range(input_data.shape[1]):
                for j in range(input_data.shape[2]):
                    cipher_list[t][i][j] = openfhe_context.encrypt(input_data[t][i][j], "cuda", 1, cryptoContext.L - 20,
                                                                   slots)
        end_time = time.time()
        print("Encryption time:", end_time - begin_time)
    if mode == "load":
        seq_len = 200
    cipher_list = handle_modes(mode, cipher_list, [input_data.shape[0], input_data.shape[1], input_data.shape[2]],
                               dir_path, "encrypt_input", "Encrypting Data")
    return cipher_list


def handle_modes(mode, cipher_list, ct_list_shape,
                 dir_path, file_name, string="Processing"):
    if mode in ["run"]:
        print(f"{string} data...")
        return cipher_list
    elif mode in ["save"]:
        print(f"{string} and saving checkpoint...")
        file_name = dir_path + f"/{file_name}.pkl"
        use_checkpoint(file_name, "SAVE_CHECKPOINT", cipher_list, ct_list_shape)
        return cipher_list
    elif mode in ["verify"]:
        print(f"{string}, saving, and then loading checkpoint for verification...")
        file_name = dir_path + f"/{file_name}.pkl"
        use_checkpoint(file_name, "SAVE_CHECKPOINT", cipher_list, ct_list_shape)
        loaded_cipher_list = use_checkpoint(file_name, "LOAD_CHECKPOINT", None, None)
        if loaded_cipher_list is None:
            raise ValueError(f"Failed to load {file_name}")
        return loaded_cipher_list
    elif mode in ["load"]:
        print(f"Loading {string} data from checkpoint...")
        file_name = dir_path + f"/{file_name}.pkl"
        cipher_list = use_checkpoint(file_name, "LOAD_CHECKPOINT", None, None)
        if cipher_list is None:
            raise ValueError(f"Failed to load {file_name}")
        return cipher_list
    else:
        raise ValueError("Invalid mode. Please choose one of: 'run', 'save', 'verify', 'load'.")


def homo_relu(ciphertext, scale, degree, cryptoContext):
    def scaled_relu_function(x):
        return 0 if x < 0 else (1 / scale) * x

    result = approx.eval_chebyshev_function(scaled_relu_function, ciphertext, -1, 1, degree, cryptoContext)
    return result


def homo_Aespa_perfect_square(i, j, k, x, filename, cryptoContext, left_mask, bottom_mask, left_=0, bottom_=0):
    if left_ == 0 and bottom_ == 0:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        temp = np.ones(32768)
        n1 = read_values_from_file(i, j, k, n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext, left_mask,
                                   scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i, j, k, n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots,
                                   cryptoContext, left_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
    elif left_ == 1 and bottom_ == 0:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        n1 = read_values_from_file(i, j, k, n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext, left_mask,
                                   scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i, j, k, n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots,
                                   cryptoContext, left_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
    elif left_ == 0 and bottom_ == 1:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        n1 = read_values_from_file(i, j, k, n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext,
                                   bottom_mask, scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i, j, k, n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots,
                                   cryptoContext, bottom_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
    elif left_ == 1 and bottom_ == 1:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        n1 = read_values_from_file(i, j, k, n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext,
                                   left_mask * bottom_mask, scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i, j, k, n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots,
                                   cryptoContext, left_mask * bottom_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res


@fhe.utils.profile_python_function
def batch_homo_relu(input, filename, cryptoContext, left_mask, bottom_mask):
    temp = np.ones(32768)
    for i in range(input.shape[0]):
        if i == 0:
            for j in range(3):
                for k in range(3):
                    if j == 0 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp, temp)
                    if j == 0 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp, temp)
                    if j == 0 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 0)
                    if j == 1 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp, temp)
                    if j == 1 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp, temp)
                    if j == 1 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 0)
                    if j == 2 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   0, 1)
                    if j == 2 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   0, 1)
                    if j == 2 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 1)
        else:
            temp1 = np.ones(32768)
            block = 16 * 128
            start = 15 * 128
            end = 16 * 128
            for base in range(0, len(temp1), block):
                temp1[base + start: base + end] = 0
            for j in range(3):
                for k in range(3):
                    if j == 0 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp1, temp1)
                    if j == 0 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp1, temp1)
                    if j == 0 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0] * temp1, 1, 0)
                    if j == 1 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp1, temp1)
                    if j == 1 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp1, temp1)
                    if j == 1 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0], 1, 0)
                    if j == 2 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0],
                                                                   bottom_mask[i, j, k][0][0] * temp1, 0, 1)
                    if j == 2 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0],
                                                                   bottom_mask[i, j, k][0][0] * temp1, 0, 1)
                    if j == 2 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0], 1, 1)
    return input


@fhe.utils.profile_python_function
def batch_homo_relu3(input, filename, cryptoContext, left_mask, bottom_mask):
    temp = np.ones(32768)
    temp1 = np.ones(32768)
    start = 3 * 128
    end = 4 * 128
    block = 4 * 128
    for i in range(0, len(temp1), block):
        temp1[i + start:i + end] = 0
    for i in range(input.shape[0]):
        for j in range(3):
            for k in range(3):
                if j == 0 and k == 0:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext, temp1,
                                                               temp1)
                if j == 0 and k == 1:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext, temp1,
                                                               temp1)
                if j == 0 and k == 2:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                               left_mask[i, j, k][0][0] * temp1,
                                                               bottom_mask[i, j, k][0][0] * temp1, 1, 0)
                if j == 1 and k == 0:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext, temp1,
                                                               temp1)
                if j == 1 and k == 1:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext, temp1,
                                                               temp1)
                if j == 1 and k == 2:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                               left_mask[i, j, k][0][0] * temp1,
                                                               bottom_mask[i, j, k][0][0] * temp1, 1, 0)
                if j == 2 and k == 0:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                               left_mask[i, j, k][0][0] * temp1,
                                                               bottom_mask[i, j, k][0][0] * temp1, 0, 1)
                if j == 2 and k == 1:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                               left_mask[i, j, k][0][0] * temp1,
                                                               bottom_mask[i, j, k][0][0] * temp1, 0, 1)
                if j == 2 and k == 2:
                    input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                               left_mask[i, j, k][0][0] * temp1,
                                                               bottom_mask[i, j, k][0][0] * temp1, 1, 1)
    return input


@fhe.utils.profile_python_function
def batch_homo_relu2(input, filename, cryptoContext, left_mask, bottom_mask):
    temp1 = np.ones(32768)
    block = 8 * 128
    start = 7 * 128
    end = 8 * 128
    temp = np.ones(32768)
    for base in range(0, len(temp1), block):
        temp1[base + start: base + end] = 0
    for i in range(input.shape[0]):
        if i == 0:
            for j in range(3):
                for k in range(3):
                    if j == 0 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp, temp)
                    if j == 0 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 0)
                    if j == 0 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 0)
                    if j == 1 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   0, 1)
                    if j == 1 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 1)
                    if j == 1 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 1)
                    if j == 2 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   0, 1)
                    if j == 2 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 1)
                    if j == 2 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0], bottom_mask[i, j, k][0][0],
                                                                   1, 1)
        else:
            for j in range(3):
                for k in range(3):
                    if j == 0 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   temp1, temp1)
                    if j == 0 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0], 1, 0)
                    if j == 0 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0], 1, 0)
                    if j == 1 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0],
                                                                   bottom_mask[i, j, k][0][0] * temp1, 0, 1)
                    if j == 1 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0], 1, 1)
                    if j == 1 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0] * temp1,
                                                                   bottom_mask[i, j, k][0][0], 1, 1)
                    if j == 2 and k == 0:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0],
                                                                   bottom_mask[i, j, k][0][0] * temp1, 0, 1)
                    if j == 2 and k == 1:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0],
                                                                   bottom_mask[i, j, k][0][0] * temp1, 1, 1)
                    if j == 2 and k == 2:
                        input[i][j][k] = homo_Aespa_perfect_square(i, j, k, input[i][j][k], filename, cryptoContext,
                                                                   left_mask[i, j, k][0][0],
                                                                   bottom_mask[i, j, k][0][0] * temp1, 1, 1)
    return input


# @fhe.utils.profile_python_function
def batch_homo_bs(input, index, logBsSlots_list, levelBudget_list, cryptoContext):
    output = np.empty((input.shape[0], input.shape[1], input.shape[2]), dtype=object)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                output[i][j][k] = fhe.homo_bootstrap(input[i][j][k], cryptoContext.L - index, logBsSlots_list[0],
                                                     levelBudget_list[0], cryptoContext)
    return output
