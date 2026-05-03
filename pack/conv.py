from .config import block_num1

import math

import numpy as np
import torch.fhe as fhe

from .encoding import read_values_from_file
from .utils import ceil_power_of_2, min_padding_to_next_multiple_of_k, perfect_square_split


def judge_is_edge(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo))
    edge_list = np.empty((int(height_pad / wi), 2))
    for i in range(1, int(height_pad / wi) + 1):
        edge_list[i - 1][1] = int((i * int(width_pad / wo) - 1) % group_num)
        edge_list[i - 1][0] = int(np.ceil(((i * int(width_pad / wo))) / group_num) - 1)
    return edge_list


def output_judge_is_edge(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    edge_list = np.empty((int(height_pad / wi), 2))
    for i in range(1, int(height_pad / wi) + 1):
        edge_list[i - 1][1] = int((i * int(width_pad / wo) - 1) % group_num)
        edge_list[i - 1][0] = int(np.ceil(((i * int(width_pad / wo))) / group_num) - 1)
    temp = [[] for _ in range(group_num)]
    for i in range(group_num):
        for j in range(int(height_pad / wi)):
            if edge_list[j][1] == i:
                temp[i].append(int(edge_list[j][0]))
    return temp


def output_judge_is_edge2(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    edge_list = np.empty((int(height_pad / wi) + 1, 2))
    for i in range(1, int(height_pad / wi) + 2):
        edge_list[i - 1][1] = int(((i * int(width_pad / wo) - 1) - int(width_pad / wo) + 1) % group_num)
        edge_list[i - 1][0] = int(
            np.ceil(((i * int(width_pad / wo)) - int(width_pad / wo) + 1) / group_num) - 1)
    temp = [[] for _ in range(group_num)]
    for i in range(group_num):
        for j in range(int(height_pad / wi)):
            if edge_list[j][1] == i:
                temp[i].append(int(edge_list[j][0]))
    return temp


def output_judge_is_edge3(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    edge_list = np.empty((int(height_pad / wi), 2))
    for i in range(0, int(height_pad / wi)):
        edge_list[i][1] = ((int(width_pad / wo) * int((width_pad / wo) - 1)) + i) % group_num
        edge_list[i][0] = np.ceil((((int(width_pad / wo) * int((width_pad / wo) - 1)) + i) + 1) / group_num)
    temp = [[] for _ in range(group_num)]
    for i in range(group_num):
        for j in range(int(height_pad / wi)):
            if edge_list[j][1] == i:
                temp[i].append(int(edge_list[j][0]) - 1)
    return temp


def output_judge_is_edge4(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    edge_list = np.empty((int(height_pad / wi), 2))
    for i in range(0, int(height_pad / wi)):
        edge_list[i][1] = ((int(width_pad / wo) * int((width_pad / wo) - 1)) + i + int(
            width_pad / wo)) % group_num
        edge_list[i][0] = (np.ceil((((int(width_pad / wo) * int((width_pad / wo) - 1)) + i) + 1 + int(
            width_pad / wo)) / group_num)) % num_in_cipher
    temp = [[] for _ in range(group_num)]
    for i in range(group_num):
        for j in range(int(height_pad / wi)):
            if edge_list[j][1] == i:
                temp[i].append((int(edge_list[j][0]) - 1) % num_in_cipher)
    return temp


def zero_num(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    zero_num = np.empty(group_num)
    for i in range(group_num):
        num = int(np.floor(block_num / group_num))
        if i < block_num % group_num:
            num += 1
        zero_num[i] = num_in_cipher - num
    return zero_num


def down_need(wi, wo, height, width, batch_size, in_channel, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher_before = int(slots / (in_channel * batch_size))
    num_in_cipher = int(slots / (in_channel * batch_size))
    height_after = height / 2
    width_after = width / 2
    pad_after = min_padding_to_next_multiple_of_k(height_after, 3)
    height_after_pad = height_after + pad_after
    width_after_pad = width_after + pad_after
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    group_num_after = int(np.ceil(height_after_pad * width_after_pad / (num_in_cipher_before * wi * wo)))
    down_need = [[] for _ in range(group_num)]

    def judge_odd(index):
        column1 = index % (width_pad // wo)
        row1 = math.floor(index / (height_pad / wi))
        if column1 % 2 == 0 and row1 % 2 == 0:
            return 0
        elif column1 % 2 == 0 and row1 % 2 == 1:
            return 2
        elif column1 % 2 == 1 and row1 % 2 == 0:
            return 1
        elif column1 % 2 == 1 and row1 % 2 == 1:
            return 3

    for i in range(group_num):
        for j in range(num_in_cipher):
            down_need[i].append(judge_odd(j * group_num + i))
    temp = [list(set(group)) for group in down_need]
    down_need_final = [[] for _ in range(group_num)]
    for i in range(group_num):
        if 0 in temp[i]:
            down_need_final[i].extend([0, 2, 6, 8])
        if 1 in temp[i]:
            down_need_final[i].extend([1, 7])
        if 2 in temp[i]:
            down_need_final[i].extend([3, 5])
        if 3 in temp[i]:
            down_need_final[i].append(4)
    result = []
    for sublist in down_need_final:
        row_col_indices = [(val // 3, val % 3) for val in sublist]
        result.append(row_col_indices)
    return result


@fhe.utils.profile_python_function
def conv_downsampling(layer, input, weight, bias, pre_bias_list2, pre_bias_list3, pre_bias_list4, cryptoContext,
                      openfhe_context, height, width, wi, wo, batch_size, in_channel, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)
    giant, baby = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output = np.empty((group_num, wi, wo), dtype=object)
    input_rotate_list = np.zeros((group_num, wi, wo, giant), dtype=object)
    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for b in range(giant // 2):
                    if b == 0:
                        input_rotate_list[i][q][r][b] = input[i][q][r]
                    else:
                        input_rotate_list[i][q][r][b] = fhe.homo_rotate(input[i][q][r], 2 * b * temp, cryptoContext)

    def conv2(o, p, q, r, input, weight, cryptoContext):
        weight_encode = np.empty((baby, giant), dtype=object)
        output_giant = np.empty(baby, dtype=object)
        if cryptoContext.config.SAVE_MIDDLE == False:
            for i in range(baby):
                for j in range(giant // 2):
                    name = f"downsample_{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(cryptoContext.pre_encoded[name], name,
                                                     cryptoContext.L - input[0].cur_limbs, slots, False, cryptoContext)
        else:
            for i in range(baby):
                for j in range(giant // 2):
                    name = f"downsample_{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(weight[i][2 * j], name, cryptoContext.L - input[0].cur_limbs,
                                                     slots,
                                                     False, cryptoContext)
        for g in range(giant // 2):
            input_temp = input[g]
            if g == 0:
                for b in range(baby):
                    output_giant[b] = fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext)
            else:
                for b in range(baby):
                    output_giant[b] = fhe.homo_add(output_giant[b],
                                                   fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext),
                                                   cryptoContext)
        for b in range(baby):
            if b == 0:
                output = output_giant[b]
            else:
                output = fhe.homo_add(output, fhe.homo_rotate(output_giant[b], -giant * b * temp, cryptoContext),
                                      cryptoContext)
        return output

    for i in range(group_num):
        for j in range(wi):
            for k in range(wo):
                output[i][j][k] = conv2(0, 0, 0, 0, input_rotate_list[i][j][k], weight[0][0], cryptoContext)
    for i in range(group_num):
        for j in range(wi):
            for k in range(wo):
                output[i][j][k] = fhe.homo_rescale(output[i][j][k], 1, cryptoContext)
    if cryptoContext.config.SAVE_MIDDLE == True:
        for i in range(group_num):
            encode_temp = fhe.encode(bias[i], f"{layer}_{i}_{0}", cryptoContext.L - output[i][0][0].cur_limbs, slots,
                                     False,
                                     cryptoContext)
            output[i][0][0] = fhe.homo_add_pt(output[i][0][0], encode_temp, cryptoContext)
            output[i][1][0] = fhe.homo_add_pt(output[i][1][0], fhe.encode(pre_bias_list3[i], f"{layer}_{i}_{1}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][2][0] = fhe.homo_add_pt(output[i][2][0], fhe.encode(pre_bias_list3[i], f"{layer}_{i}_{2}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][0][2] = fhe.homo_add_pt(output[i][0][2], fhe.encode(pre_bias_list2[i], f"{layer}_{i}_{3}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][0][1] = fhe.homo_add_pt(output[i][0][1], fhe.encode(pre_bias_list2[i], f"{layer}_{i}_{4}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][1][2] = fhe.homo_add_pt(output[i][1][2], fhe.encode(pre_bias_list4[i], f"{layer}_{i}_{5}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][1][1] = fhe.homo_add_pt(output[i][1][1], fhe.encode(pre_bias_list4[i], f"{layer}_{i}_{6}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][2][1] = fhe.homo_add_pt(output[i][2][1], fhe.encode(pre_bias_list4[i], f"{layer}_{i}_{7}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
            output[i][2][2] = fhe.homo_add_pt(output[i][2][2], fhe.encode(pre_bias_list4[i], f"{layer}_{i}_{8}",
                                                                          cryptoContext.L -
                                                                          output[i][0][0].cur_limbs,
                                                                          slots, False, cryptoContext),
                                              cryptoContext)
        return output
    for i in range(group_num):
        encode_temp = fhe.encode(bias[i], "", cryptoContext.L - output[i][0][0].cur_limbs, slots, False,
                                 cryptoContext)
        output[i][0][0] = fhe.homo_add_pt(output[i][0][0], encode_temp, cryptoContext)
        output[i][1][0] = fhe.homo_add_pt(output[i][1][0], fhe.encode(pre_bias_list3[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][2][0] = fhe.homo_add_pt(output[i][2][0], fhe.encode(pre_bias_list3[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][0][2] = fhe.homo_add_pt(output[i][0][2], fhe.encode(pre_bias_list2[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][0][1] = fhe.homo_add_pt(output[i][0][1], fhe.encode(pre_bias_list2[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][1][2] = fhe.homo_add_pt(output[i][1][2], fhe.encode(pre_bias_list4[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][1][1] = fhe.homo_add_pt(output[i][1][1], fhe.encode(pre_bias_list4[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][2][1] = fhe.homo_add_pt(output[i][2][1], fhe.encode(pre_bias_list4[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
        output[i][2][2] = fhe.homo_add_pt(output[i][2][2], fhe.encode(pre_bias_list4[i], "",
                                                                      cryptoContext.L -
                                                                      output[i][0][0].cur_limbs,
                                                                      slots, False, cryptoContext),
                                          cryptoContext)
    return output


@fhe.utils.profile_python_function
def conv_batch2(input, zero_num_temp, pre_weight_list, pre_weight_list_edge, edge_list, pre_weight_list2,
                pre_weight_list3, pre_weight_list4, pre_weight_list5, pre_bias_list, pre_bias_list2, pre_bias_list3,
                pre_bias_list4, batch_size, in_channel, output_channel, height, width, wi, wo, cryptoContext,
                openfhe_context, isresnet, res_initial, layer):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)
    baby, giant = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    output = np.empty((num_in_cipher, slots), dtype=object)
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output_cipher = np.empty((group_num, wi, wo), dtype=object)
    middle_input = np.empty((group_num, 3, 3, num_in_cipher), dtype=object)
    middle_input.fill(None)

    for i in range(group_num):
        for j in range(wo):
            if j < (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][5][0] != 0:
                            if middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] is None:
                                middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] = fhe.homo_rotate(
                                    input[edge_list[i][5][1]][2][0], batch_size * edge_list[i][5][0], cryptoContext)

                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][4][0] != 0:
                            if middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] is None:
                                middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] = fhe.homo_rotate(
                                    input[edge_list[i][4][1]][2][2], batch_size * edge_list[i][4][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

            if j == (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        pass
            if j > (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                        if edge_list[i][7][0] != 0:
                            if middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] is None:
                                middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] = fhe.homo_rotate(
                                    input[edge_list[i][7][1]][0][0], batch_size * edge_list[i][7][0], cryptoContext)

                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                        if edge_list[i][6][0] != 0:
                            if middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] is None:
                                middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] = fhe.homo_rotate(
                                    input[edge_list[i][6][1]][0][2], batch_size * edge_list[i][6][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

    input_rotate_list = np.zeros((group_num, wi, wo, giant), dtype=object)
    input_rotate_list2 = np.zeros((group_num, wi, wo, num_in_cipher, giant), dtype=object)
    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for b in range(giant):
                    if b == 0:
                        input_rotate_list[i][q][r][b] = input[i][q][r]
                    else:
                        input_rotate_list[i][q][r][b] = fhe.homo_rotate(input[i][q][r], b * temp, cryptoContext)

    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for k in range(num_in_cipher):
                    if middle_input[i][q][r][k] is not None:
                        for b in range(giant):
                            if b == 0:
                                input_rotate_list2[i][q][r][k][b] = middle_input[i][q][r][k].deep_copy()
                            else:
                                input_rotate_list2[i][q][r][k][b] = fhe.homo_rotate(middle_input[i][q][r][k], b * temp,
                                                                                    cryptoContext)

    def rotate_weight(weight, index):
        output_weight = np.empty((baby, giant, slots))
        for i in range(baby):
            for j in range(giant):
                output_weight[i][j] = np.roll(weight[i][j], -batch_size * index)
        return output_weight

    def conv2(o, p, q, r, input, weight, cryptoContext):
        weight_encode = np.empty((baby, giant), dtype=object)
        output_giant = np.empty(baby, dtype=object)
        if cryptoContext.config.SAVE_MIDDLE == False:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(cryptoContext.pre_encoded[name], name,
                                                     cryptoContext.L - input[0].cur_limbs, slots, False, cryptoContext)
        else:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(weight[i][j], name, cryptoContext.L - input[0].cur_limbs, slots,
                                                     False, cryptoContext)
        for g in range(giant):
            input_temp = input[g]
            if g == 0:
                for b in range(baby):
                    output_giant[b] = fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext)
            else:
                for b in range(baby):
                    output_giant[b] = fhe.homo_add(output_giant[b],
                                                   fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext),
                                                   cryptoContext)
        return output_giant

    for i in range(group_num):
        for j in range(wo):
            if j < (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        if edge_list[i][0][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][0] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][1] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][0] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][0],
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][1],
                                                 cryptoContext)
                        if edge_list[i][5][0] != 0:
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                       0)][max(edge_list[i][5][0], 0)][0][2] *
                                                               pre_weight_list3[edge_list[i][5][1]][0][2],
                                                               edge_list[i][5][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                     max(edge_list[i][5][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][5][1]][0][2],
                                                 cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][1], pre_weight_list_edge[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][2], pre_weight_list_edge[i][1][1],
                                             cryptoContext)
                        if edge_list[i][3][0] != 0:
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][1][1], pre_weight_list_edge[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][1][2], pre_weight_list_edge[i][2][1],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 0:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][4][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                       0)][
                                                                   max(edge_list[i][4][0], 0)][0][0],
                                                               edge_list[i][4][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                     max(edge_list[i][4][0], 0)][0][0],
                                                 cryptoContext)
                        if edge_list[i][0][0] != 0:
                            temp_list[1] = conv2(i, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][2],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1],
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][2],
                                                 cryptoContext)
                        if edge_list[i][2][0] != 0:
                            temp_list[3] = conv2(i, j, k, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][2][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0],
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][2][0],
                                                 cryptoContext)
                        temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][0][0], pre_weight_list[0][0][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][0][1], pre_weight_list[0][0][1][2],
                                             cryptoContext)
                        temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][1][0], pre_weight_list[0][0][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][1][1], pre_weight_list[0][0][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][0][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][0],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[1] = conv2(i, 0, 0, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[2] = conv2(i, 0, 0, 2,
                                                 input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][2],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][0],
                                                 cryptoContext)
                            temp_list[1] = conv2(i, 0, 0, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1],
                                                 cryptoContext)
                            temp_list[2] = conv2(i, 0, 0, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][2],
                                                 cryptoContext)
                        temp_list[3] = conv2(0, j, k, 3, input_rotate_list[i][0][0], pre_weight_list[0][0][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(0, 0, 0, 4, input_rotate_list[i][0][1], pre_weight_list[0][0][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(0, 0, 0, 5, input_rotate_list[i][0][2], pre_weight_list[0][0][1][2],
                                             cryptoContext)
                        temp_list[6] = conv2(0, j, k, 6, input_rotate_list[i][1][0], pre_weight_list[0][0][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(0, 0, 0, 7, input_rotate_list[i][1][1], pre_weight_list[0][0][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(0, 0, 0, 8, input_rotate_list[i][1][2], pre_weight_list[0][0][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][3][0] != 0:
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][0][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][1], pre_weight_list_edge[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][2], pre_weight_list_edge[i][0][1],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][1], pre_weight_list_edge[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][2], pre_weight_list_edge[i][1][1],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][1], pre_weight_list_edge[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][2], pre_weight_list_edge[i][2][1],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 0:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][2][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][max(edge_list[i][2][0], 0)][0][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[3] = conv2(i, 0, 0, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[6] = conv2(i, 0, 0, 6,
                                                 input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][2][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][0][0],
                                                 cryptoContext)
                            temp_list[3] = conv2(i, 0, 0, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0],
                                                 cryptoContext)
                            temp_list[6] = conv2(i, 0, 0, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][2][0],
                                                 cryptoContext)
                        temp_list[1] = conv2(0, j, k, 1, input_rotate_list[i][0][0], pre_weight_list[0][0][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(0, j, k, 2, input_rotate_list[i][0][1], pre_weight_list[0][0][0][2],
                                             cryptoContext)
                        temp_list[4] = conv2(0, 0, 0, 4, input_rotate_list[i][1][0], pre_weight_list[0][0][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(0, 0, 0, 5, input_rotate_list[i][1][1], pre_weight_list[0][0][1][2],
                                             cryptoContext)
                        temp_list[7] = conv2(0, 0, 0, 7, input_rotate_list[i][2][0], pre_weight_list[0][0][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(0, 0, 0, 8, input_rotate_list[i][2][1], pre_weight_list[0][0][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list[0] = conv2(0, j, k, 0, input_rotate_list[i][0][0], pre_weight_list[0][0][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(0, 1, 0, 1, input_rotate_list[i][0][1], pre_weight_list[0][0][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(0, 1, 0, 2, input_rotate_list[i][0][2], pre_weight_list[0][0][0][2],
                                             cryptoContext)
                        temp_list[3] = conv2(0, 0, 1, 3, input_rotate_list[i][1][0], pre_weight_list[0][0][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(0, 0, 0, 4, input_rotate_list[i][1][1], pre_weight_list[0][0][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(0, 0, 0, 5, input_rotate_list[i][1][2], pre_weight_list[0][0][1][2],
                                             cryptoContext)
                        temp_list[6] = conv2(0, 0, 1, 6, input_rotate_list[i][2][0], pre_weight_list[0][0][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(0, 0, 0, 7, input_rotate_list[i][2][1], pre_weight_list[0][0][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(0, 0, 0, 8, input_rotate_list[i][2][2], pre_weight_list[0][0][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][3][0] != 0:
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][0][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                        if edge_list[i][7][0] != 0:
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][7][1]][0][0][edge_list[i][7][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][7][0]),
                                                                       0)][
                                                                   max(edge_list[i][7][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][7][1]][2][2] *
                                                               pre_weight_list5[edge_list[i][7][1]][2][2],
                                                               edge_list[i][7][0]),
                                                 cryptoContext)
                        else:
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                     max(edge_list[i][7][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][7][1]][2][2] *
                                                 pre_weight_list5[edge_list[i][7][1]][2][2],
                                                 cryptoContext)
                        if edge_list[i][1][0] != 0:
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][0] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][0] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][0],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][0] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][0] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][0],
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][2],
                                             pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][1],
                                             pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][1],
                                             pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][2],
                                             pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 0:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][1][0] != 0:
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][2] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][2],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][2] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][2],
                                                 cryptoContext)
                        if edge_list[i][6][0] != 0:
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                       0)][
                                                                   max(edge_list[i][6][0], 0)][2][0] *
                                                               pre_weight_list5[edge_list[i][6][1]][2][0],
                                                               edge_list[i][6][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                     max(edge_list[i][6][0], 0)][2][0] *
                                                 pre_weight_list5[edge_list[i][6][1]][2][0],
                                                 cryptoContext)
                        if edge_list[i][2][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][0][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][0][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][0][0],
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][1][0],
                                                 cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][0],
                                             pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][1][1],
                                             pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][0],
                                             pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][2][1],
                                             pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][1][0] != 0:
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][0] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][0],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][2] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][2],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][0] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][0],
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][2] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][2],
                                                 cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][0],
                                             pre_weight_list[0][0][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][1],
                                             pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][1][2],
                                             pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][0],
                                             pre_weight_list[0][0][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][1],
                                             pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][2][2],
                                             pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
    if isresnet == True:
        for i in range(group_num):
            for j in range(wi):
                for k in range(wo):
                    output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
    else:
        left, right = layer.split("_", 1)
        a = int(left)
        b = int(right)
        if (a - 1) * block_num1 + int(np.floor((b + 1) / 2)) == block_num1 + 1 or (a - 1) * block_num1 + int(
            np.floor((b + 1) / 2)) == block_num1 * 2 + 1:
            for i in range(group_num):
                for j in range(wi):
                    for k in range(wo):
                        output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k],
                                                              cryptoContext)
        else:
            temp = np.ones(slots)
            A2 = read_values_from_file(0, 0, 0,
                                       f"layer{(a - 1) * block_num1 + int(np.floor((b + 1) / 2))}-conv{2}bn{2}-A2",
                                       cryptoContext.L - res_initial[0][0][0].cur_limbs,
                                       2 ** 15,
                                       cryptoContext, temp, 1)
            for i in range(group_num):
                for j in range(wi):
                    for k in range(wo):
                        res_initial[i][j][k] = fhe.homo_mul_pt(res_initial[i][j][k], A2, cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k],
                                                              cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
    return output_cipher


@fhe.utils.profile_python_function
def conv_batch2_mod2(input, zero_num_temp, pre_weight_list, pre_weight_list_edge, edge_list, pre_weight_list2,
                     pre_weight_list3, pre_weight_list4, pre_weight_list5, pre_bias_list, pre_bias_list2,
                     pre_bias_list3,
                     pre_bias_list4, batch_size, in_channel, output_channel, height, width, wi, wo, cryptoContext,
                     openfhe_context, isresnet, res_initial, layer):
    N = 65536
    slots = int(N / 2)

    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)
    baby, giant = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    output = np.empty((num_in_cipher, slots), dtype=object)
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output_cipher = np.empty((group_num, wi, wo), dtype=object)
    middle_input = np.empty((group_num, 3, 3, num_in_cipher), dtype=object)
    middle_input.fill(None)

    for i in range(group_num):
        for j in range(wo):
            if j < (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][5][0] != 0:
                            if middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] is None:
                                middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] = fhe.homo_rotate(
                                    input[edge_list[i][5][1]][2][0], batch_size * edge_list[i][5][0], cryptoContext)

                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][4][0] != 0:
                            if middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] is None:
                                middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] = fhe.homo_rotate(
                                    input[edge_list[i][4][1]][2][2], batch_size * edge_list[i][4][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

            if j == (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        pass
            if j > (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                        if edge_list[i][7][0] != 0:
                            if middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] is None:
                                middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] = fhe.homo_rotate(
                                    input[edge_list[i][7][1]][0][0], batch_size * edge_list[i][7][0], cryptoContext)

                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                        if edge_list[i][6][0] != 0:
                            if middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] is None:
                                middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] = fhe.homo_rotate(
                                    input[edge_list[i][6][1]][0][2], batch_size * edge_list[i][6][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

    input_rotate_list = np.zeros((group_num, wi, wo, giant), dtype=object)
    input_rotate_list2 = np.zeros((group_num, wi, wo, num_in_cipher, giant), dtype=object)
    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for b in range(giant):
                    if b == 0:
                        input_rotate_list[i][q][r][b] = input[i][q][r]
                    else:
                        input_rotate_list[i][q][r][b] = fhe.homo_rotate(input[i][q][r], b * temp, cryptoContext)

    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for k in range(num_in_cipher):
                    if middle_input[i][q][r][k] is not None:
                        for b in range(giant):
                            if b == 0:
                                input_rotate_list2[i][q][r][k][b] = middle_input[i][q][r][k].deep_copy()
                            else:
                                input_rotate_list2[i][q][r][k][b] = fhe.homo_rotate(middle_input[i][q][r][k], b * temp,
                                                                                    cryptoContext)

    def rotate_weight(weight, index):
        output_weight = np.empty((baby, giant, slots))
        for i in range(baby):
            for j in range(giant):
                output_weight[i][j] = np.roll(weight[i][j], -batch_size * index)
        return output_weight

    def conv2(o, p, q, r, input, weight, cryptoContext):
        weight_encode = np.empty((baby, giant), dtype=object)
        output_giant = np.empty(baby, dtype=object)
        if cryptoContext.config.SAVE_MIDDLE == False:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(cryptoContext.pre_encoded[name], name,
                                                     cryptoContext.L - input[0].cur_limbs, slots, False, cryptoContext)
        else:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(weight[i][j], name, cryptoContext.L - input[0].cur_limbs, slots,
                                                     False, cryptoContext)
        for g in range(giant):
            input_temp = input[g]
            if g == 0:
                for b in range(baby):
                    output_giant[b] = fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext)
            else:
                for b in range(baby):
                    output_giant[b] = fhe.homo_add(output_giant[b],
                                                   fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext),
                                                   cryptoContext)
        return output_giant
        for g in range(giant):
            if g == 0:
                output = output_giant[g]
            else:

                output = fhe.homo_add(output, fhe.homo_rotate(output_giant[g], -baby * g * temp, cryptoContext),
                                      cryptoContext)
        return output

    for i in range(group_num):
        for j in range(wo):
            if j < (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        if edge_list[i][0][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][0] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][1] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][0] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][0],
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][1],
                                                 cryptoContext)
                        if edge_list[i][5][0] != 0:
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                       0)][max(edge_list[i][5][0], 0)][0][2] *
                                                               pre_weight_list3[edge_list[i][5][1]][0][2],
                                                               edge_list[i][5][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                     max(edge_list[i][5][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][5][1]][0][2],
                                                 cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][1], pre_weight_list_edge[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][2], pre_weight_list_edge[i][1][1],
                                             cryptoContext)
                        if edge_list[i][3][0] != 0:
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][1][1], pre_weight_list_edge[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][1][2], pre_weight_list_edge[i][2][1],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 0:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][4][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                       0)][
                                                                   max(edge_list[i][4][0], 0)][0][0],
                                                               edge_list[i][4][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                     max(edge_list[i][4][0], 0)][0][0],
                                                 cryptoContext)
                        if edge_list[i][0][0] != 0:
                            temp_list[1] = conv2(i, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][2],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1],
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][2],
                                                 cryptoContext)
                        if edge_list[i][2][0] != 0:
                            temp_list[3] = conv2(i, j, k, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][2][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0],
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][2][0],
                                                 cryptoContext)
                        temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][0][0], pre_weight_list[0][0][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][0][1], pre_weight_list[0][0][1][2],
                                             cryptoContext)
                        temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][1][0], pre_weight_list[0][0][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][1][1], pre_weight_list[0][0][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][0][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][0] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][1] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][2] *
                                                               pre_weight_list2[edge_list[i][0][1]][0][2],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][0] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][0],
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][1],
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][2] *
                                                 pre_weight_list2[edge_list[i][0][1]][0][2],
                                                 cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][0], pre_weight_list_edge[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][1], pre_weight_list_edge[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][0][2], pre_weight_list_edge[i][1][2],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][1][0], pre_weight_list_edge[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][1][1], pre_weight_list_edge[i][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[i][1][2], pre_weight_list_edge[i][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][3][0] != 0:
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][0][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][2][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][2][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][1],
                                             pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][2],
                                             pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][1],
                                             pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][2],
                                             pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][1],
                                             pre_weight_list_edge[i][2][0] * pre_weight_list4[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][2],
                                             pre_weight_list_edge[i][2][1] * pre_weight_list4[i][2][1],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 0:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][2][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][max(edge_list[i][2][0], 0)][0][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][2][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][2][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][0][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][0][0],
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][1][0],
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][2][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][2][0],
                                                 cryptoContext)
                        temp_list[1] = conv2(int(i != 0), j, k, 1, input_rotate_list[i][0][0],
                                             pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(int(i != 0), j, k, 2, input_rotate_list[i][0][1],
                                             pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[4] = conv2(int(i != 0), j, k, 4, input_rotate_list[i][1][0],
                                             pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(int(i != 0), j, k, 5, input_rotate_list[i][1][1],
                                             pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        temp_list[7] = conv2(int(i != 0), j, k, 7, input_rotate_list[i][2][0],
                                             pre_weight_list[0][0][2][1] * pre_weight_list4[i][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(int(i != 0), j, k, 8, input_rotate_list[i][2][1],
                                             pre_weight_list[0][0][2][2] * pre_weight_list4[i][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][0],
                                             pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][1],
                                             pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][0][2],
                                             pre_weight_list_edge[i][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][0],
                                             pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][1],
                                             pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][1][2],
                                             pre_weight_list_edge[i][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][0],
                                             pre_weight_list_edge[i][2][0] * pre_weight_list4[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][1],
                                             pre_weight_list_edge[i][2][1] * pre_weight_list4[i][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[i][2][2],
                                             pre_weight_list_edge[i][2][2] * pre_weight_list4[i][2][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][3][0] != 0:
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][0][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                               pre_weight_list4[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                        if edge_list[i][7][0] != 0:
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][7][1]][0][0][edge_list[i][7][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][7][0]),
                                                                       0)][
                                                                   max(edge_list[i][7][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][7][1]][2][2] *
                                                               pre_weight_list5[edge_list[i][7][1]][2][2],
                                                               edge_list[i][7][0]),
                                                 cryptoContext)
                        else:
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                     max(edge_list[i][7][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][7][1]][2][2] *
                                                 pre_weight_list5[edge_list[i][7][1]][2][2],
                                                 cryptoContext)
                        if edge_list[i][1][0] != 0:
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][0] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][0] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][0],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][0] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][0] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][0],
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][2],
                                             pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][1],
                                             pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][1],
                                             pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][2],
                                             pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 0:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][1][0] != 0:
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][2] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][2],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][2] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][2],
                                                 cryptoContext)
                        if edge_list[i][6][0] != 0:
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                       0)][
                                                                   max(edge_list[i][6][0], 0)][2][0] *
                                                               pre_weight_list5[edge_list[i][6][1]][2][0],
                                                               edge_list[i][6][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                     max(edge_list[i][6][0], 0)][2][0] *
                                                 pre_weight_list5[edge_list[i][6][1]][2][0],
                                                 cryptoContext)
                        if edge_list[i][2][0] != 0:
                            temp_list[0] = conv2(i, j, k, 0,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][0][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0] *
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][0][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][0][0],
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0] *
                                                 pre_weight_list4[edge_list[i][2][1]][1][0],
                                                 cryptoContext)
                        temp_list[1] = conv2(int(i != 0), j, k, 1, input_rotate_list[i][1][0],
                                             pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(int(i != 0), j, k, 2, input_rotate_list[i][1][1],
                                             pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[4] = conv2(int(i != 0), j, k, 4, input_rotate_list[i][2][0],
                                             pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(int(i != 0), j, k, 5, input_rotate_list[i][2][1],
                                             pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                    if k == 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][1][0] != 0:
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][0] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][0] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][0],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][2] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][2] *
                                                               pre_weight_list2[edge_list[i][1][1]][2][2],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][0] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][0] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][0],
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][2] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][2] *
                                                 pre_weight_list2[edge_list[i][1][1]][2][2],
                                                 cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][0],
                                             pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][1],
                                             pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][1][2],
                                             pre_weight_list_edge[i][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][0],
                                             pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][1],
                                             pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][2][2],
                                             pre_weight_list_edge[i][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        output_giant_ = np.empty(baby, dtype=object)
                        for x in range(baby):
                            for y in range(9):
                                if y == 0:
                                    output_giant_[x] = temp_list[y][x]
                                else:
                                    output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                    cryptoContext)
                        for g in range(baby):
                            if g == 0:
                                output_cipher[i, j, k] = output_giant_[g]
                            else:

                                output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
    if isresnet == True:
        for i in range(group_num):
            for j in range(wi):
                for k in range(wo):
                    output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
    else:
        left, right = layer.split("_", 1)
        a = int(left)
        b = int(right)
        if (a - 1) * block_num1 + int(np.floor((b + 1) / 2)) == block_num1 + 1 or (a - 1) * block_num1 + int(
            np.floor((b + 1) / 2)) == block_num1 * 2 + 1:
            for i in range(group_num):
                for j in range(wi):
                    for k in range(wo):
                        output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k],
                                                              cryptoContext)
        else:
            A2 = read_values_from_file(1, 1, 1,
                                       f"layer{(a - 1) * block_num1 + int(np.floor((b + 1) / 2))}-conv{2}bn{2}-A2",
                                       cryptoContext.L - res_initial[0][0][0].cur_limbs,
                                       2 ** 15,
                                       cryptoContext, np.ones(slots), 1)
            for i in range(group_num):
                for j in range(wi):
                    for k in range(wo):
                        res_initial[i][j][k] = fhe.homo_mul_pt(res_initial[i][j][k], A2, cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k],
                                                              cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
    return output_cipher


@fhe.utils.profile_python_function
def conv2_batch_extend(input, zero_num_temp, pre_weight_list, pre_weight_list_edge, edge_list, pre_weight_list2,
                       pre_weight_list3, pre_weight_list4, pre_weight_list5, pre_bias_list, pre_bias_list2,
                       pre_bias_list3,
                       pre_bias_list4, batch_size, in_channel, output_channel, height, width, wi, wo, cryptoContext,
                       openfhe_context, isresnet, res_initial, layer, down_need, pre_weight_lista,
                       pre_weight_list_edgea, pre_weight_list2a, pre_weight_list3a, pre_weight_list4a,
                       pre_weight_list5a, pre_bias_lista, pre_bias_list2a, pre_bias_list3a, pre_bias_list4a
                       ):
    N = 65536
    slots = int(N / 2)

    num_in_cipher = int(slots / (in_channel * batch_size))
    repeat = int(output_channel / in_channel)
    baby, giant = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    output = np.empty((num_in_cipher, slots), dtype=object)
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output_cipher = np.empty((group_num * repeat, wi, wo), dtype=object)
    middle_input = np.empty((group_num, 3, 3, num_in_cipher), dtype=object)
    middle_input.fill(None)

    for i in range(group_num):
        for j in range(wo):
            if j < (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][5][0] != 0:
                            if middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] is None:
                                middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] = fhe.homo_rotate(
                                    input[edge_list[i][5][1]][2][0], batch_size * edge_list[i][5][0], cryptoContext)

                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][4][0] != 0:
                            if middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] is None:
                                middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] = fhe.homo_rotate(
                                    input[edge_list[i][4][1]][2][2], batch_size * edge_list[i][4][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

            if j == (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        pass
            if j > (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                        if edge_list[i][7][0] != 0:
                            if middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] is None:
                                middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] = fhe.homo_rotate(
                                    input[edge_list[i][7][1]][0][0], batch_size * edge_list[i][7][0], cryptoContext)

                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                        if edge_list[i][6][0] != 0:
                            if middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] is None:
                                middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] = fhe.homo_rotate(
                                    input[edge_list[i][6][1]][0][2], batch_size * edge_list[i][6][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

    input_rotate_list = np.zeros((group_num, wi, wo, giant), dtype=object)
    input_rotate_list2 = np.zeros((group_num, wi, wo, num_in_cipher, giant), dtype=object)
    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for b in range(giant):
                    if b == 0:
                        input_rotate_list[i][q][r][b] = input[i][q][r]
                    else:
                        input_rotate_list[i][q][r][b] = fhe.homo_rotate(input[i][q][r], b * temp, cryptoContext)

    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for k in range(num_in_cipher):
                    if middle_input[i][q][r][k] is not None:
                        for b in range(giant):
                            if b == 0:
                                input_rotate_list2[i][q][r][k][b] = middle_input[i][q][r][k].deep_copy()
                            else:
                                input_rotate_list2[i][q][r][k][b] = fhe.homo_rotate(middle_input[i][q][r][k], b * temp,
                                                                                    cryptoContext)

    def rotate_weight(weight, index):
        output_weight = np.empty((baby, giant, slots))
        for i in range(baby):
            for j in range(giant):
                output_weight[i][j] = np.roll(weight[i][j], -batch_size * index)
        return output_weight

    def conv2(o, p, q, r, input, weight, cryptoContext):
        weight_encode = np.empty((baby, giant), dtype=object)
        output_giant = np.empty(baby, dtype=object)
        if cryptoContext.config.SAVE_MIDDLE == False:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(cryptoContext.pre_encoded[name], name,
                                                     cryptoContext.L - input[0].cur_limbs, slots, False, cryptoContext)
        else:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(weight[i][j], name, cryptoContext.L - input[0].cur_limbs, slots,
                                                     False, cryptoContext)
        for g in range(giant):
            input_temp = input[g]
            if g == 0:
                for b in range(baby):
                    output_giant[b] = fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext)
            else:
                for b in range(baby):
                    output_giant[b] = fhe.homo_add(output_giant[b],
                                                   fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext),
                                                   cryptoContext)
        return output_giant
        for g in range(giant):
            if g == 0:
                output = output_giant[g]
            else:

                output = fhe.homo_add(output, fhe.homo_rotate(output_giant[g], -baby * g * temp, cryptoContext),
                                      cryptoContext)
        return output

    for i in range(group_num):
        for l in range(len(down_need[i])):
            j, k = down_need[i][l]
            if j < (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                    if edge_list[i][5][0] != 0:
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                   0)][max(edge_list[i][5][0], 0)][0][2] *
                                                           pre_weight_list3[edge_list[i][5][1]][0][2],
                                                           edge_list[i][5][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                 max(edge_list[i][5][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][5][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][1], pre_weight_list_edge[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][2], pre_weight_list_edge[i][1][1],
                                         cryptoContext)
                    if edge_list[i][3][0] != 0:
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][1][1], pre_weight_list_edge[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][1][2], pre_weight_list_edge[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][4][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                   0)][
                                                               max(edge_list[i][4][0], 0)][0][0],
                                                           edge_list[i][4][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                 max(edge_list[i][4][0], 0)][0][0],
                                             cryptoContext)
                    if edge_list[i][0][0] != 0:
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0],
                                             cryptoContext)
                    temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][0][0], pre_weight_list[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][0][1], pre_weight_list[0][0][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][1][0], pre_weight_list[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][1][1], pre_weight_list[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(0, j, k, 3, input_rotate_list[i][0][0], pre_weight_list[0][0][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][0][1], pre_weight_list[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][0][2], pre_weight_list[0][0][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(0, j, k, 6, input_rotate_list[i][1][0], pre_weight_list[0][0][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][1][1], pre_weight_list[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][1][2], pre_weight_list[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][1], pre_weight_list_edge[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][2], pre_weight_list_edge[i][0][1],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][1], pre_weight_list_edge[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][2], pre_weight_list_edge[i][1][1],
                                         cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][1], pre_weight_list_edge[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][2], pre_weight_list_edge[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][max(edge_list[i][2][0], 0)][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0],
                                             cryptoContext)
                    temp_list[1] = conv2(0, j, k, 1, input_rotate_list[i][0][0], pre_weight_list[0][0][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(0, j, k, 2, input_rotate_list[i][0][1], pre_weight_list[0][0][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][1][0], pre_weight_list[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][1][1], pre_weight_list[0][0][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][2][0], pre_weight_list[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][2][1], pre_weight_list[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    temp_list[0] = conv2(0, j, k, 0, input_rotate_list[i][0][0], pre_weight_list[0][0][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(0, j, k, 1, input_rotate_list[i][0][1], pre_weight_list[0][0][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(0, j, k, 2, input_rotate_list[i][0][2], pre_weight_list[0][0][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(0, j, k, 3, input_rotate_list[i][1][0], pre_weight_list[0][0][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][1][1], pre_weight_list[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][1][2], pre_weight_list[0][0][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(0, j, k, 6, input_rotate_list[i][2][0], pre_weight_list[0][0][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][2][1], pre_weight_list[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][2][2], pre_weight_list[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                    if edge_list[i][7][0] != 0:
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][7][1]][0][0][edge_list[i][7][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][7][0]),
                                                                   0)][
                                                               max(edge_list[i][7][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][7][1]][2][2] *
                                                           pre_weight_list5[edge_list[i][7][1]][2][2],
                                                           edge_list[i][7][0]),
                                             cryptoContext)
                    else:
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                 max(edge_list[i][7][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][7][1]][2][2] *
                                             pre_weight_list5[edge_list[i][7][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list2[edge_list[i][1][1]][2][0] *
                                             pre_weight_list5[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list2[edge_list[i][1][1]][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][2],
                                         pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][1],
                                         pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][1],
                                         pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][2],
                                         pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][6][0] != 0:
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                   0)][
                                                               max(edge_list[i][6][0], 0)][2][0] *
                                                           pre_weight_list5[edge_list[i][6][1]][2][0],
                                                           edge_list[i][6][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                 max(edge_list[i][6][0], 0)][2][0] *
                                             pre_weight_list5[edge_list[i][6][1]][2][0],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][0],
                                         pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][1][1],
                                         pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][0],
                                         pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][2][1],
                                         pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list5[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][0],
                                         pre_weight_list[0][0][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][1],
                                         pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][1][2],
                                         pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][0],
                                         pre_weight_list[0][0][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][1],
                                         pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][2][2],
                                         pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
    for i in range(group_num):
        for l in range(len(down_need[i])):
            j, k = down_need[i][l]
            if j < (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                    if edge_list[i][5][0] != 0:
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                   0)][max(edge_list[i][5][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][5][1]][0][2],
                                                           edge_list[i][5][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                 max(edge_list[i][5][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][5][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][0][1],
                                         pre_weight_list_edgea[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][0][2],
                                         pre_weight_list_edgea[i][1][1],
                                         cryptoContext)
                    if edge_list[i][3][0] != 0:
                        temp_list[5] = conv2(i + group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][4][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                   0)][
                                                               max(edge_list[i][4][0], 0)][0][0],
                                                           edge_list[i][4][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                 max(edge_list[i][4][0], 0)][0][0],
                                             cryptoContext)
                    if edge_list[i][0][0] != 0:
                        temp_list[1] = conv2(i + group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[3] = conv2(i + group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0],
                                             cryptoContext)
                    temp_list[4] = conv2(group_num, j, k, 4, input_rotate_list[i][0][0], pre_weight_lista[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(group_num, j, k, 5, input_rotate_list[i][0][1], pre_weight_lista[0][0][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(group_num, j, k, 7, input_rotate_list[i][1][0], pre_weight_lista[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(group_num, j, k, 8, input_rotate_list[i][1][1], pre_weight_lista[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(group_num, j, k, 3, input_rotate_list[i][0][0], pre_weight_lista[0][0][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(group_num, j, k, 4, input_rotate_list[i][0][1], pre_weight_lista[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(group_num, j, k, 5, input_rotate_list[i][0][2], pre_weight_lista[0][0][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(group_num, j, k, 6, input_rotate_list[i][1][0], pre_weight_lista[0][0][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(group_num, j, k, 7, input_rotate_list[i][1][1], pre_weight_lista[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(group_num, j, k, 8, input_rotate_list[i][1][2], pre_weight_lista[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][0][1],
                                         pre_weight_list_edgea[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][0][2],
                                         pre_weight_list_edgea[i][0][1],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][1][1],
                                         cryptoContext)
                    temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[i][2][2],
                                         pre_weight_list_edgea[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][max(edge_list[i][2][0], 0)][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0],
                                             cryptoContext)
                    temp_list[1] = conv2(group_num, j, k, 1, input_rotate_list[i][0][0], pre_weight_lista[0][0][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(group_num, j, k, 2, input_rotate_list[i][0][1], pre_weight_lista[0][0][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(group_num, j, k, 4, input_rotate_list[i][1][0], pre_weight_lista[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(group_num, j, k, 5, input_rotate_list[i][1][1], pre_weight_lista[0][0][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(group_num, j, k, 7, input_rotate_list[i][2][0], pre_weight_lista[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(group_num, j, k, 8, input_rotate_list[i][2][1], pre_weight_lista[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    temp_list[0] = conv2(group_num, j, k, 0, input_rotate_list[i][0][0], pre_weight_lista[0][0][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(group_num, j, k, 1, input_rotate_list[i][0][1], pre_weight_lista[0][0][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(group_num, j, k, 2, input_rotate_list[i][0][2], pre_weight_lista[0][0][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(group_num, j, k, 3, input_rotate_list[i][1][0], pre_weight_lista[0][0][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(group_num, j, k, 4, input_rotate_list[i][1][1], pre_weight_lista[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(group_num, j, k, 5, input_rotate_list[i][1][2], pre_weight_lista[0][0][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(group_num, j, k, 6, input_rotate_list[i][2][0], pre_weight_lista[0][0][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(group_num, j, k, 7, input_rotate_list[i][2][1], pre_weight_lista[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(group_num, j, k, 8, input_rotate_list[i][2][2], pre_weight_lista[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                    if edge_list[i][7][0] != 0:
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][7][1]][0][0][edge_list[i][7][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][7][0]),
                                                                   0)][
                                                               max(edge_list[i][7][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][7][1]][2][2] *
                                                           pre_weight_list5a[edge_list[i][7][1]][2][2],
                                                           edge_list[i][7][0]),
                                             cryptoContext)
                    else:
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                 max(edge_list[i][7][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][7][1]][2][2] *
                                             pre_weight_list5a[edge_list[i][7][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][2][2],
                                         pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[7] = conv2(i + group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][6][0] != 0:
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                   0)][
                                                               max(edge_list[i][6][0], 0)][2][0] *
                                                           pre_weight_list5a[edge_list[i][6][1]][2][0],
                                                           edge_list[i][6][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                 max(edge_list[i][6][0], 0)][2][0] *
                                             pre_weight_list5a[edge_list[i][6][1]][2][0],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][1][0],
                                         pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[i][1][1],
                                         pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][2][0],
                                         pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[i][2][1],
                                         pre_weight_lista[0][0][1][2] * pre_weight_list4a[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][1][0],
                                         pre_weight_lista[0][0][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][1][1],
                                         pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[i][1][2],
                                         pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][2][0],
                                         pre_weight_lista[0][0][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][2][1],
                                         pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[i][2][2],
                                         pre_weight_lista[0][0][1][2] * pre_weight_list4a[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
    if isresnet == True:
        for i in range(group_num):
            for l in range(len(down_need[i])):
                j, k = down_need[i][l]
                output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
        for i in range(group_num):
            for l in range(len(down_need[i])):
                j, k = down_need[i][l]
                output_cipher[i + group_num][j][k] = fhe.homo_rescale(output_cipher[i + group_num][j][k], 1,
                                                                      cryptoContext)
    else:
        for i in range(group_num):
            for j in range(wi):
                for k in range(wo):
                    output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                    output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k], cryptoContext)
    return output_cipher


@fhe.utils.profile_python_function
def conv2_batch_extend2(input, zero_num_temp, pre_weight_list, pre_weight_list_edge, edge_list, pre_weight_list2,
                        pre_weight_list3, pre_weight_list4, pre_weight_list5, pre_bias_list, pre_bias_list2,
                        pre_bias_list3,
                        pre_bias_list4, batch_size, in_channel, output_channel, height, width, wi, wo, cryptoContext,
                        openfhe_context, isresnet, res_initial, layer, down_need, pre_weight_lista,
                        pre_weight_list_edgea, pre_weight_list2a, pre_weight_list3a, pre_weight_list4a,
                        pre_weight_list5a, pre_bias_lista, pre_bias_list2a, pre_bias_list3a, pre_bias_list4a
                        ):
    N = 65536
    slots = int(N / 2)

    num_in_cipher = int(slots / (in_channel * batch_size))
    repeat = int(output_channel / in_channel)
    baby, giant = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    output = np.empty((num_in_cipher, slots), dtype=object)
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output_cipher = np.empty((group_num * repeat, wi, wo), dtype=object)
    middle_input = np.empty((group_num, 3, 3, num_in_cipher), dtype=object)
    middle_input.fill(None)

    for i in range(group_num):
        for j in range(wo):
            if j < (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][5][0] != 0:
                            if middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] is None:
                                middle_input[edge_list[i][5][1]][2][0][edge_list[i][5][0]] = fhe.homo_rotate(
                                    input[edge_list[i][5][1]][2][0], batch_size * edge_list[i][5][0], cryptoContext)

                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                        if edge_list[i][4][0] != 0:
                            if middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] is None:
                                middle_input[edge_list[i][4][1]][2][2][edge_list[i][4][0]] = fhe.homo_rotate(
                                    input[edge_list[i][4][1]][2][2], batch_size * edge_list[i][4][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][0][0] != 0:
                            if middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][0][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][0], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][1][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][1], batch_size * edge_list[i][0][0], cryptoContext)

                            if middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] is None:
                                middle_input[edge_list[i][0][1]][2][2][edge_list[i][0][0]] = fhe.homo_rotate(
                                    input[edge_list[i][0][1]][2][2], batch_size * edge_list[i][0][0], cryptoContext)

            if j == (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][0][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][0][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][0][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][0][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        pass
            if j > (((wi + 1) / 2) - 1):
                for k in range(wi):
                    if k == wi - 1:
                        if edge_list[i][3][0] != 0:
                            if middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][1][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][1][0], batch_size * edge_list[i][3][0], cryptoContext)

                            if middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] is None:
                                middle_input[edge_list[i][3][1]][2][0][edge_list[i][3][0]] = fhe.homo_rotate(
                                    input[edge_list[i][3][1]][2][0], batch_size * edge_list[i][3][0], cryptoContext)

                        if edge_list[i][7][0] != 0:
                            if middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] is None:
                                middle_input[edge_list[i][7][1]][0][0][edge_list[i][7][0]] = fhe.homo_rotate(
                                    input[edge_list[i][7][1]][0][0], batch_size * edge_list[i][7][0], cryptoContext)

                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

                    if k == 0:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                        if edge_list[i][6][0] != 0:
                            if middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] is None:
                                middle_input[edge_list[i][6][1]][0][2][edge_list[i][6][0]] = fhe.homo_rotate(
                                    input[edge_list[i][6][1]][0][2], batch_size * edge_list[i][6][0], cryptoContext)

                        if edge_list[i][2][0] != 0:
                            if middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][1][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][1][2], batch_size * edge_list[i][2][0], cryptoContext)

                            if middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] is None:
                                middle_input[edge_list[i][2][1]][2][2][edge_list[i][2][0]] = fhe.homo_rotate(
                                    input[edge_list[i][2][1]][2][2], batch_size * edge_list[i][2][0], cryptoContext)

                    if k == 1:
                        if edge_list[i][1][0] != 0:
                            if middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][0][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][0], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][1][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][1], batch_size * edge_list[i][1][0], cryptoContext)

                            if middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] is None:
                                middle_input[edge_list[i][1][1]][0][2][edge_list[i][1][0]] = fhe.homo_rotate(
                                    input[edge_list[i][1][1]][0][2], batch_size * edge_list[i][1][0], cryptoContext)

    input_rotate_list = np.zeros((group_num, wi, wo, giant), dtype=object)
    input_rotate_list2 = np.zeros((group_num, wi, wo, num_in_cipher, giant), dtype=object)
    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for b in range(giant):
                    if b == 0:
                        input_rotate_list[i][q][r][b] = input[i][q][r]
                    else:
                        input_rotate_list[i][q][r][b] = fhe.homo_rotate(input[i][q][r], b * temp, cryptoContext)

    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for k in range(num_in_cipher):
                    if middle_input[i][q][r][k] is not None:
                        for b in range(giant):
                            if b == 0:
                                input_rotate_list2[i][q][r][k][b] = middle_input[i][q][r][k].deep_copy()
                            else:
                                input_rotate_list2[i][q][r][k][b] = fhe.homo_rotate(middle_input[i][q][r][k], b * temp,
                                                                                    cryptoContext)

    def rotate_weight(weight, index):
        output_weight = np.empty((baby, giant, slots))
        for i in range(baby):
            for j in range(giant):
                output_weight[i][j] = np.roll(weight[i][j], -batch_size * index)
        return output_weight

    def conv2(o, p, q, r, input, weight, cryptoContext):
        weight_encode = np.empty((baby, giant), dtype=object)
        output_giant = np.empty(baby, dtype=object)
        if cryptoContext.config.SAVE_MIDDLE == False:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(cryptoContext.pre_encoded[name], name,
                                                     cryptoContext.L - input[0].cur_limbs, slots, False, cryptoContext)
        else:
            for i in range(baby):
                for j in range(giant):
                    name = f"weight_layer{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(weight[i][j], name, cryptoContext.L - input[0].cur_limbs, slots,
                                                     False, cryptoContext)
        for g in range(giant):
            input_temp = input[g]
            if g == 0:
                for b in range(baby):
                    output_giant[b] = fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext)
            else:
                for b in range(baby):
                    output_giant[b] = fhe.homo_add(output_giant[b],
                                                   fhe.homo_mul_pt(input_temp, weight_encode[b][g], cryptoContext),
                                                   cryptoContext)
        return output_giant
        for g in range(giant):
            if g == 0:
                output = output_giant[g]
            else:
                output = fhe.homo_add(output, fhe.homo_rotate(output_giant[g], -baby * g * temp, cryptoContext),
                                      cryptoContext)
        return output

    for i in range(group_num):
        for l in range(len(down_need[i])):
            j, k = down_need[i][l]
            if j < (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                    if edge_list[i][5][0] != 0:
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                   0)][max(edge_list[i][5][0], 0)][0][2] *
                                                           pre_weight_list3[edge_list[i][5][1]][0][2],
                                                           edge_list[i][5][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                 max(edge_list[i][5][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][5][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][1], pre_weight_list_edge[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][2], pre_weight_list_edge[i][1][1],
                                         cryptoContext)
                    if edge_list[i][3][0] != 0:
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][1][1], pre_weight_list_edge[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][1][2], pre_weight_list_edge[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][4][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                   0)][
                                                               max(edge_list[i][4][0], 0)][0][0],
                                                           edge_list[i][4][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                 max(edge_list[i][4][0], 0)][0][0],
                                             cryptoContext)
                    if edge_list[i][0][0] != 0:
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0],
                                             cryptoContext)
                    temp_list[4] = conv2(0, j, k, 4, input_rotate_list[i][0][0], pre_weight_list[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(0, j, k, 5, input_rotate_list[i][0][1], pre_weight_list[0][0][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(0, j, k, 7, input_rotate_list[i][1][0], pre_weight_list[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(0, j, k, 8, input_rotate_list[i][1][1], pre_weight_list[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][2] *
                                                           pre_weight_list2[edge_list[i][0][1]][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2] *
                                             pre_weight_list2[edge_list[i][0][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][0], pre_weight_list_edge[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][1], pre_weight_list_edge[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][0][2], pre_weight_list_edge[i][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][1][0], pre_weight_list_edge[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][1][1], pre_weight_list_edge[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(i, j, k, 8, input_rotate_list[i][1][2], pre_weight_list_edge[i][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][2][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][3][1]][2][2] *
                                             pre_weight_list4[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][1],
                                         pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][2],
                                         pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][1],
                                         pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][2],
                                         pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][1],
                                         pre_weight_list_edge[i][2][0] * pre_weight_list4[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][2],
                                         pre_weight_list_edge[i][2][1] * pre_weight_list4[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0] *
                                             pre_weight_list4[edge_list[i][2][1]][2][0],
                                             cryptoContext)
                    temp_list[1] = conv2(int(i != 0), j, k, 1, input_rotate_list[i][0][0],
                                         pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(int(i != 0), j, k, 2, input_rotate_list[i][0][1],
                                         pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[4] = conv2(int(i != 0), j, k, 4, input_rotate_list[i][1][0],
                                         pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(int(i != 0), j, k, 5, input_rotate_list[i][1][1],
                                         pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(int(i != 0), j, k, 7, input_rotate_list[i][2][0],
                                         pre_weight_list[0][0][2][1] * pre_weight_list4[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(int(i != 0), j, k, 8, input_rotate_list[i][2][1],
                                         pre_weight_list[0][0][2][2] * pre_weight_list4[i][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][0],
                                         pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][1],
                                         pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][0][2],
                                         pre_weight_list_edge[i][0][2] * pre_weight_list4[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][0],
                                         pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][1],
                                         pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][1][2],
                                         pre_weight_list_edge[i][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][0],
                                         pre_weight_list_edge[i][2][0] * pre_weight_list4[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][1],
                                         pre_weight_list_edge[i][2][1] * pre_weight_list4[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(i, j, k, 8, input_rotate_list[i][2][2],
                                         pre_weight_list_edge[i][2][2] * pre_weight_list4[i][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2] *
                                                           pre_weight_list4[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                    if edge_list[i][7][0] != 0:
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][7][1]][0][0][edge_list[i][7][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][7][0]),
                                                                   0)][
                                                               max(edge_list[i][7][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][7][1]][2][2] *
                                                           pre_weight_list5[edge_list[i][7][1]][2][2],
                                                           edge_list[i][7][0]),
                                             cryptoContext)
                    else:
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                 max(edge_list[i][7][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][7][1]][2][2] *
                                             pre_weight_list5[edge_list[i][7][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list2[edge_list[i][1][1]][2][0] *
                                             pre_weight_list5[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list2[edge_list[i][1][1]][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][2],
                                         pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][1],
                                         pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][1],
                                         pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][2],
                                         pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][6][0] != 0:
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                   0)][
                                                               max(edge_list[i][6][0], 0)][2][0] *
                                                           pre_weight_list5[edge_list[i][6][1]][2][0],
                                                           edge_list[i][6][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                 max(edge_list[i][6][0], 0)][2][0] *
                                             pre_weight_list5[edge_list[i][6][1]][2][0],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                    temp_list[1] = conv2(int(i != 0), j, k, 1, input_rotate_list[i][1][0],
                                         pre_weight_list[0][0][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(int(i != 0), j, k, 2, input_rotate_list[i][1][1],
                                         pre_weight_list[0][0][0][2] * pre_weight_list4[i][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(int(i != 0), j, k, 4, input_rotate_list[i][2][0],
                                         pre_weight_list[0][0][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(int(i != 0), j, k, 5, input_rotate_list[i][2][1],
                                         pre_weight_list[0][0][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][2] *
                                                           pre_weight_list2[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list5[edge_list[i][1][1]][2][0] *
                                             pre_weight_list2[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1] *
                                             pre_weight_list2[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5[edge_list[i][1][1]][2][2] *
                                             pre_weight_list2[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][1][0],
                                         pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][1][1],
                                         pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][1][2],
                                         pre_weight_list_edge[i][0][2] * pre_weight_list4[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][2][0],
                                         pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][2][1],
                                         pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][2][2],
                                         pre_weight_list_edge[i][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i, j, k] = output_giant_[g]
                        else:

                            output_cipher[i, j, k] = fhe.homo_add(output_cipher[i, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
    for i in range(group_num):
        for l in range(len(down_need[i])):
            j, k = down_need[i][l]
            if j < (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                    if edge_list[i][5][0] != 0:
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                   0)][max(edge_list[i][5][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][5][1]][0][2],
                                                           edge_list[i][5][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                 max(edge_list[i][5][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][5][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][0][1],
                                         pre_weight_list_edgea[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][0][2],
                                         pre_weight_list_edgea[i][1][1],
                                         cryptoContext)
                    if edge_list[i][3][0] != 0:
                        temp_list[5] = conv2(i + group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][4][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                   0)][
                                                               max(edge_list[i][4][0], 0)][0][0],
                                                           edge_list[i][4][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                 max(edge_list[i][4][0], 0)][0][0],
                                             cryptoContext)
                    if edge_list[i][0][0] != 0:
                        temp_list[1] = conv2(i + group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[3] = conv2(i + group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0],
                                             cryptoContext)
                    temp_list[4] = conv2(group_num, j, k, 4, input_rotate_list[i][0][0], pre_weight_lista[0][0][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(group_num, j, k, 5, input_rotate_list[i][0][1], pre_weight_lista[0][0][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(group_num, j, k, 7, input_rotate_list[i][1][0], pre_weight_lista[0][0][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(group_num, j, k, 8, input_rotate_list[i][1][1], pre_weight_lista[0][0][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][2] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][0][0],
                                         pre_weight_list_edgea[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][0][1],
                                         pre_weight_list_edgea[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[i][0][2],
                                         pre_weight_list_edgea[i][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[i][1][0],
                                         pre_weight_list_edgea[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][2][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][2][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][0][1],
                                         pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][0][2],
                                         pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][2][0] * pre_weight_list4a[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[i][2][2],
                                         pre_weight_list_edgea[i][2][1] * pre_weight_list4a[i][2][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][2][0],
                                             cryptoContext)
                    temp_list[1] = conv2(int(i != 0) + group_num, j, k, 1, input_rotate_list[i][0][0],
                                         pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(int(i != 0) + group_num, j, k, 2, input_rotate_list[i][0][1],
                                         pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(int(i != 0) + group_num, j, k, 4, input_rotate_list[i][1][0],
                                         pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(int(i != 0) + group_num, j, k, 5, input_rotate_list[i][1][1],
                                         pre_weight_lista[0][0][1][2] * pre_weight_list4a[i][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(int(i != 0) + group_num, j, k, 7, input_rotate_list[i][2][0],
                                         pre_weight_lista[0][0][2][1] * pre_weight_list4a[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(int(i != 0) + group_num, j, k, 8, input_rotate_list[i][2][1],
                                         pre_weight_lista[0][0][2][2] * pre_weight_list4a[i][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][0][0],
                                         pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][0][1],
                                         pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[i][0][2],
                                         pre_weight_list_edgea[i][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][1][0],
                                         pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][1][2] * pre_weight_list4a[i][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[i][2][0],
                                         pre_weight_list_edgea[i][2][0] * pre_weight_list4a[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][2][1] * pre_weight_list4a[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[i][2][2],
                                         pre_weight_list_edgea[i][2][2] * pre_weight_list4a[i][2][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i + group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                    if edge_list[i][7][0] != 0:
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][7][1]][0][0][edge_list[i][7][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][7][0]),
                                                                   0)][
                                                               max(edge_list[i][7][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][7][1]][2][2] *
                                                           pre_weight_list5a[edge_list[i][7][1]][2][2],
                                                           edge_list[i][7][0]),
                                             cryptoContext)
                    else:
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                 max(edge_list[i][7][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][7][1]][2][2] *
                                             pre_weight_list5a[edge_list[i][7][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][2][2],
                                         pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[7] = conv2(i + group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][6][0] != 0:
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                   0)][
                                                               max(edge_list[i][6][0], 0)][2][0] *
                                                           pre_weight_list5a[edge_list[i][6][1]][2][0],
                                                           edge_list[i][6][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                 max(edge_list[i][6][0], 0)][2][0] *
                                             pre_weight_list5a[edge_list[i][6][1]][2][0],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i + group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                    temp_list[1] = conv2(int(i != 0) + group_num, j, k, 1, input_rotate_list[i][1][0],
                                         pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(int(i != 0) + group_num, j, k, 2, input_rotate_list[i][1][1],
                                         pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(int(i != 0) + group_num, j, k, 4, input_rotate_list[i][2][0],
                                         pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(int(i != 0) + group_num, j, k, 5, input_rotate_list[i][2][1],
                                         pre_weight_lista[0][0][1][2] * pre_weight_list4a[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i + group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][2] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i + group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][0] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i + group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i + group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][2] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i + group_num, j, k, 0, input_rotate_list[i][1][0],
                                         pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i + group_num, j, k, 1, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i + group_num, j, k, 2, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i + group_num, j, k, 3, input_rotate_list[i][2][0],
                                         pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i + group_num, j, k, 4, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i + group_num, j, k, 5, input_rotate_list[i][2][2],
                                         pre_weight_list_edgea[i][1][2] * pre_weight_list4a[i][1][2],
                                         cryptoContext)
                    output_giant_ = np.empty(baby, dtype=object)
                    for x in range(baby):
                        for y in range(9):
                            if y == 0:
                                output_giant_[x] = temp_list[y][x]
                            else:
                                output_giant_[x] = fhe.homo_add(output_giant_[x], temp_list[y][x],
                                                                cryptoContext)
                    for g in range(baby):
                        if g == 0:
                            output_cipher[i + group_num, j, k] = output_giant_[g]
                        else:

                            output_cipher[i + group_num, j, k] = fhe.homo_add(output_cipher[i + group_num, j, k],
                                                                              fhe.homo_rotate(output_giant_[g],
                                                                                              -giant * g * temp,
                                                                                              cryptoContext),
                                                                              cryptoContext)
    if isresnet == True:
        for i in range(group_num):
            for l in range(len(down_need[i])):
                j, k = down_need[i][l]
                output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
        for i in range(group_num):
            for l in range(len(down_need[i])):
                j, k = down_need[i][l]
                output_cipher[i + group_num][j][k] = fhe.homo_rescale(output_cipher[i + group_num][j][k], 1,
                                                                      cryptoContext)
    else:
        for i in range(group_num):
            for j in range(wi):
                for k in range(wo):
                    output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                    output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k], cryptoContext)
    return output_cipher


def is_middle(index, wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo))
    temp_list = []
    for i in range(int(width_pad / wo)):
        temp_list.append(i)
        temp_list.append(i * (width_pad / wo))
        temp_list.append((i + 1) * (width_pad / wo) - 1)
    for i in range(num_in_cipher):
        if (i * group_num + index) not in temp_list:
            return i


def output_location(index, wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo))
    if index == None:
        return [None, None]
    if index >= block_num:
        return [None, None]
    locate_list = np.empty(2, dtype=int)
    locate_list[0] = np.floor(index / group_num)
    locate_list[1] = (index % group_num)
    return locate_list


def pre_mask1(input, wi, wo, height, width, batch_size, in_channel, cryptoContext):
    N = 65536
    slots = int(N / 2)
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    num_in_cipher = ceil_power_of_2(block_num)
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    channel_in_cipher = slots / (batch_size * num_in_cipher)
    output_preweight = np.zeros((num_in_cipher, slots), dtype=float)
    output_preweight_encode = np.zeros((num_in_cipher, slots), dtype=object)
    for i in range(num_in_cipher):
        for j in range(int(channel_in_cipher)):
            output_preweight[i][i * batch_size + j * (num_in_cipher * batch_size):(i + 1) * batch_size + j * (
                num_in_cipher * batch_size)] = 1
    for i in range(num_in_cipher):
        output_preweight_encode[i] = fhe.encode(output_preweight[i], "", 0, slots, False, cryptoContext)
    return output_preweight_encode


def pre_mask2(wi, wo, height, width, batch_size, in_channel, cryptoContext, index):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = ceil_power_of_2(int(slots / (in_channel * batch_size)))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output_preweight = np.zeros((num_in_cipher, slots), dtype=float)
    output_preweight_encode = np.zeros((num_in_cipher), dtype=object)
    for i in range(num_in_cipher):
        for j in range(int(in_channel)):
            output_preweight[i][i * batch_size + j * (num_in_cipher * batch_size):(i + 1) * batch_size + j * (
                num_in_cipher * batch_size)] = 1
    if cryptoContext.config.SAVE_MIDDLE == True:
        for i in range(num_in_cipher):
            name = f"pre_mask2_{i}_{index}"
            output_preweight_encode[i] = fhe.encode(output_preweight[i], name, 0, slots, False, cryptoContext)
    if cryptoContext.config.SAVE_MIDDLE == False:
        for i in range(num_in_cipher):
            name = f"pre_mask2_{i}_{index}"
            output_preweight_encode[i] = fhe.encode(cryptoContext.pre_encoded[name], name, 0, slots, False,
                                                    cryptoContext)
    return output_preweight_encode


@fhe.utils.profile_python_function
def downsampling(input, pre_mask, wi, wo, height, width, batch_size, in_channel, output_channel, plan, cryptoContext,
                 openfhe_context):
    N = 65536
    slots = int(N / 2)
    num_in_cipher_before = int(slots / (in_channel * batch_size))
    num_in_cipher = int(slots / (output_channel * batch_size))
    height_after = height / 2
    width_after = width / 2
    pad_after = min_padding_to_next_multiple_of_k(height_after, 3)
    height_after_pad = height_after + pad_after
    width_after_pad = width_after + pad_after
    pad = min_padding_to_next_multiple_of_k(height, 3)
    extend_in_out = int(output_channel / in_channel)
    extend_in_out = 2
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher_before * wi * wo)))
    group_num_after = int(np.ceil(height_after_pad * width_after_pad / (num_in_cipher * wi * wo)))
    output = np.empty((group_num_after, wi, wo), dtype=object)

    def remap_index(old_idx):
        if output_channel == 32:
            r = old_idx // 12
            c = old_idx % 12
            if c == 11:
                return None
            return int(r * 11 + c)
        if output_channel == 64:
            r = old_idx // 6
            c = old_idx % 6
            if c == 11:
                return None
            return int(r * 6 + c)

    def to_next_even(x):
        return (x + 1) // 2 * 2

    def judge_row_column():
        row_pad = int(to_next_even(height_pad / wi))
        temp = np.empty((2, 2, group_num_after, num_in_cipher, 2))
        for t in range(2):
            for i in range(2):
                for j in range(group_num_after):
                    num = 0
                    for k in range(num_in_cipher):
                        temp[t][i][j][k] = output_location(
                            remap_index(row_pad * t + i + 2 * j + k * 2 * group_num_after + num * row_pad), wi, wo,
                            height, width, batch_size, in_channel)
                        if (row_pad * t + i + 2 * j + k * 2 * group_num_after + num * row_pad) // row_pad != (
                            row_pad * t + i + 2 * j + (k + 1) * 2 * group_num_after + num * row_pad) // row_pad:
                            num += 1
        return temp

    down_fx = judge_row_column()
    if plan == 2:
        for i in range(group_num_after):
            for j in range(4):
                if j == 0:
                    temp1 = down_fx[0][0][i]
                    flag = 0
                    for x in range(extend_in_out):
                        if x == 0:
                            for k in range(num_in_cipher):
                                if np.isnan(temp1[k][0]):
                                    flag += 1
                                else:
                                    break
                        for k in range(num_in_cipher):
                            if not np.isnan(temp1[k][0]):
                                if x == 0 and k == flag:
                                    temp_output1 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][0][0],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                    temp_output2 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][0][2],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                    temp_output3 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][2][0],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                    temp_output4 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][2][2],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                else:
                                    temp_output1 = fhe.homo_add(temp_output1, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][0][0],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                                    temp_output2 = fhe.homo_add(temp_output2, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][0][2],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                                    temp_output3 = fhe.homo_add(temp_output3, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][2][0],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                                    temp_output4 = fhe.homo_add(temp_output4, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][2][2],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                    output[i][0][0] = temp_output1
                    output[i][0][1] = temp_output2
                    output[i][1][0] = temp_output3
                    output[i][1][1] = temp_output4
                if j == 1:
                    temp1 = down_fx[0][1][i]
                    flag = 0
                    for x in range(extend_in_out):
                        if x == 0:
                            for k in range(num_in_cipher):
                                if np.isnan(temp1[k][0]):
                                    flag += 1
                                else:
                                    break
                        for k in range(num_in_cipher):
                            if not np.isnan(temp1[k][0]):
                                if x == 0 and k == flag:
                                    temp_output1 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][0][1],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                    temp_output4 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][2][1],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                else:
                                    temp_output1 = fhe.homo_add(temp_output1, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][0][1],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                                    temp_output4 = fhe.homo_add(temp_output4, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][2][1],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                    output[i][0][2] = temp_output1
                    output[i][1][2] = temp_output4
                if j == 2:
                    temp1 = down_fx[1][0][i]
                    flag = 0
                    for x in range(extend_in_out):
                        if x == 0:
                            for k in range(num_in_cipher):
                                if np.isnan(temp1[k][0]):
                                    flag += 1
                                else:
                                    break
                        for k in range(num_in_cipher):
                            if not np.isnan(temp1[k][0]):
                                if x == 0 and k == flag:
                                    temp_output1 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][1][0],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                    temp_output4 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][1][2],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                else:
                                    temp_output1 = fhe.homo_add(temp_output1, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][1][0],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                                    temp_output4 = fhe.homo_add(temp_output4, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][1][2],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                    output[i][2][0] = temp_output1
                    output[i][2][1] = temp_output4
                if j == 3:
                    temp1 = down_fx[1][1][i]
                    flag = 0
                    for x in range(extend_in_out):
                        if x == 0:
                            for k in range(num_in_cipher):
                                if np.isnan(temp1[k][0]):
                                    flag += 1
                                else:
                                    break
                        for k in range(num_in_cipher):
                            if not np.isnan(temp1[k][0]):
                                if x == 0 and k == flag:
                                    temp_output1 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][1][1],
                                                                                   pre_mask[int(temp1[k][0])],
                                                                                   cryptoContext),
                                                                   batch_size * (int(temp1[k][0]) - k), cryptoContext)
                                else:
                                    temp_output1 = fhe.homo_add(temp_output1, fhe.homo_rotate(
                                        fhe.homo_mul_pt(input[int(temp1[k][1]) + x * group_num][1][1],
                                                        pre_mask[int(temp1[k][0])], cryptoContext),
                                        batch_size * (int(temp1[k][0]) - k) - x * num_in_cipher * batch_size,
                                        cryptoContext), cryptoContext)
                    output[i][2][2] = temp_output1
    for i in range(group_num_after):
        for j in range(3):
            for k in range(3):
                output[i][j][k] = fhe.homo_rescale(output[i][j][k], 1, cryptoContext)
    return output


def judge_rotate(wi, wo, height, width, batch_size, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    locate_list = np.empty((group_num, 8, 2), dtype=int)
    for i in range(group_num):
        locate = is_middle(i, wi, wo, height, width, batch_size, output_channel)
        locate_temp = locate * group_num + i
        locate_temp = i
        locate_list[i, 0, 0] = \
            output_location(locate_temp - int(height_pad / wi), wi, wo, height, width, batch_size, output_channel)[0] - \
            output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 1, 0] = \
            output_location(locate_temp + int(height_pad / wi), wi, wo, height, width, batch_size, output_channel)[0] - \
            output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 2, 0] = output_location(locate_temp - 1, wi, wo, height, width, batch_size, output_channel)[0] - \
                               output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 3, 0] = output_location(locate_temp + 1, wi, wo, height, width, batch_size, output_channel)[0] - \
                               output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 0, 1] = \
            output_location(locate_temp - int(height_pad / wi), wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 1, 1] = \
            output_location(locate_temp + int(height_pad / wi), wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 2, 1] = output_location(locate_temp - 1, wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 3, 1] = output_location(locate_temp + 1, wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 4, 0] = \
            output_location(locate_temp - int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[
                0] - \
            output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 5, 0] = \
            output_location(locate_temp - int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[
                0] - \
            output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 6, 0] = \
            output_location(locate_temp + int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[
                0] - \
            output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 7, 0] = \
            output_location(locate_temp + int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[
                0] - \
            output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 4, 1] = \
            output_location(locate_temp - int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[
                1]
        locate_list[i, 5, 1] = \
            output_location(locate_temp - int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[
                1]
        locate_list[i, 6, 1] = \
            output_location(locate_temp + int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[
                1]
        locate_list[i, 7, 1] = \
            output_location(locate_temp + int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[
                1]
    return locate_list
