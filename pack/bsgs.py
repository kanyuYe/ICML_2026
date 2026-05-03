import numpy as np

from .model import get_weight_bias
from .utils import min_padding_to_next_multiple_of_k, perfect_square_split


def pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,
               cryptoContext):
    N = 65536
    slots = int(N / 2)
    repeat = int(output_channel / in_channel)
    num_in_cipher = int(slots / (output_channel * batch_size))
    if layer == "downsample0" or layer == "downsample1":
        weight101, bias101 = get_weight_bias(layer, state_dict, index, cryptoContext)
    else:
        weight101 = get_weight_bias(layer, state_dict, index, cryptoContext)
    if layer == "layer2[0]" and index == 2:
        weight_temp = weight101.copy()
        half = output_channel // 2
        new_order = np.stack((np.arange(half), np.arange(half, output_channel)), axis=1).ravel()
        a_rearranged = weight_temp[:, new_order, :, :]
        weight101 = a_rearranged
    if layer == "layer3[0]" and index == 2:
        weight_temp = weight101.copy()
        half = output_channel // 2
        new_order = np.stack((np.arange(half), np.arange(half, output_channel)), axis=1).ravel()
        a_rearranged = weight_temp[:, new_order, :, :]
        weight101 = a_rearranged
    weight101 = weight101 * scale
    if layer == "":
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    if layer == "downsample0":
        b = np.zeros((32, 32, 3, 3), dtype=arr_expanded101.dtype)
        b[::2, 0::2, :, :] = arr_expanded101[::2, :, :, :]
        b[1::2, 1::2, :, :] = arr_expanded101[1::2, :, :, :]
        arr_expanded101 = b
    if layer == "downsample1":
        b = np.zeros((64, 64, 3, 3), dtype=arr_expanded101.dtype)
        b[::2, 0::2, :, :] = arr_expanded101[::2, :, :, :]
        b[1::2, 1::2, :, :] = arr_expanded101[1::2, :, :, :]
        arr_expanded101 = b
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer == "downsample0":
        giant, baby = perfect_square_split(in_channel)
    elif layer == "downsample1":
        giant, baby = perfect_square_split(in_channel)
    else:
        baby, giant = perfect_square_split(in_channel)
    pre_weight_list = np.empty((wi, wo, baby, giant, slots))
    if layer == "downsample0" or layer == "downsample1":
        repeat_bias = int(slots / output_channel)
        pre_bias_list = np.repeat(bias101, repeat_bias)
        pre_bias_list1 = np.empty((group_num, slots))
        for i in range(group_num):
            pre_bias_list1[i] = pre_bias_list.copy()
    temp = batch_size * num_in_cipher
    for wii in range(wi):
        for woo in range(wo):
            for b in range(baby):
                for g in range(giant):
                    for r in range(int(repeat)):
                        for i in range(in_channel):
                            input = (g + i) % in_channel
                            output = (r * in_channel + (b * giant + i)) % output_channel
                            pre_weight_list[wii][woo][b][g][
                            i * temp + r * in_channel * temp:(i + 1) * temp + r * in_channel * temp] = \
                                arr_expanded101[output][input][wii][woo]
    if layer == "downsample0" or layer == "downsample1":
        for i in range(group_num):
            num = int(np.floor(block_num / group_num))
            if i < block_num % group_num:
                num += 1
            for j in range(output_channel):
                pre_bias_list1[i][j * temp + num * batch_size:(j + 1) * temp] = 0
        return pre_weight_list, pre_bias_list1
    return pre_weight_list


def pre_weight2(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,
                cryptoContext):
    N = 65536
    slots = int(N / 2)
    repeat = int(output_channel / in_channel)
    num_in_cipher = int(slots / (output_channel * batch_size))
    if layer == "downsample0" or layer == "downsample1":
        weight101, bias101 = get_weight_bias(layer, state_dict, index, cryptoContext)
    else:
        weight101 = get_weight_bias(layer, state_dict, index, cryptoContext)
    weight101 = weight101 * scale
    if layer == "":
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    if layer == "layer2[0]":
        arr_expanded1 = arr_expanded101[:16]
        arr_expanded2 = arr_expanded101[16:]
    if layer == "layer3[0]":
        arr_expanded1 = arr_expanded101[:32]
        arr_expanded2 = arr_expanded101[32:]
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    baby, giant = perfect_square_split(in_channel)
    repeat_bias = int(slots / output_channel)
    pre_weight_list = np.empty((wi, wo, baby, giant, slots))
    pre_weight_list2 = np.empty((wi, wo, baby, giant, slots))
    if layer == "downsample0" or layer == "downsample1":
        pre_bias_list1 = np.repeat(bias1, repeat_bias)
        pre_bias_list2 = np.repeat(bias2, repeat_bias)
        pre_bias_list11 = np.empty((group_num, slots))
        pre_bias_list22 = np.empty((group_num, slots))
        for i in range(group_num):
            pre_bias_list11[i] = pre_bias_list1.copy()
        for i in range(group_num):
            pre_bias_list22[i] = pre_bias_list2.copy()
    temp = batch_size * num_in_cipher
    for wi in range(3):
        for wo in range(3):
            for b in range(baby):
                for g in range(giant):
                    for r in range(int(repeat)):
                        for i in range(in_channel):
                            input = (g + i) % in_channel
                            output = (r * in_channel + (b * giant + i)) % output_channel
                            pre_weight_list[wi][wo][b][g][
                            i * temp + r * in_channel * temp:(i + 1) * temp + r * in_channel * temp] = \
                                arr_expanded1[output][input][wi][wo]
    for wi in range(3):
        for wo in range(3):
            for b in range(baby):
                for g in range(giant):
                    for r in range(int(repeat)):
                        for i in range(in_channel):
                            input = (g + i) % in_channel
                            output = (r * in_channel + (b * giant + i)) % output_channel
                            pre_weight_list2[wi][wo][b][g][
                            i * temp + r * in_channel * temp:(i + 1) * temp + r * in_channel * temp] = \
                                arr_expanded2[output][input][wi][wo]
    return pre_weight_list, pre_weight_list2


def pre_weight_edge(pre_weight_list, pre_bias_list, batch_size, in_channel, output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)
    baby, giant = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    pre_weight_list1 = np.empty((num_in_cipher, 3, 3, baby, giant, slots))
    pre_bias_list1 = np.empty((num_in_cipher, slots))
    for i in range(num_in_cipher):
        pre_weight_list1[i] = pre_weight_list.copy()
    for i in range(num_in_cipher):
        pre_bias_list1[i] = pre_bias_list.copy()
    for wi in range(3):
        for wo in range(3):
            for b in range(baby):
                for g in range(giant):
                    for r in range(int(repeat)):
                        for i in range(num_in_cipher):
                            for j in range(in_channel):
                                pre_weight_list1[i][wi][wo][b][g][i * batch_size + temp * j + r * in_channel * temp:(
                                                                                                                        i + 1) * batch_size + temp * j + r * in_channel * temp] = 0
    for r in range(int(repeat)):
        for i in range(num_in_cipher):
            for j in range(in_channel):
                pre_bias_list1[i][i * batch_size + temp * j + r * in_channel * temp:(
                                                                                        i + 1) * batch_size + temp * j + r * in_channel * temp] = 0
    return pre_weight_list1, pre_bias_list1


def pre_weight_edge2(edge_list, pre_weight_list, pre_bias_list, batch_size, in_channel, output_channel, height, width,
                     wi, wo, layer):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)
    if layer == "downsample0":
        giant, baby = perfect_square_split(in_channel)
    else:
        baby, giant = perfect_square_split(in_channel)
    temp = batch_size * num_in_cipher
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    pre_weight_list1 = np.empty((group_num, wi, wo, baby, giant, slots))
    pre_weight_list2 = np.ones((group_num, wi, wo, baby, giant, slots))
    pre_bias_list1 = pre_bias_list.copy()
    pre_bias_list2 = np.ones((group_num, slots))
    for i in range(group_num):
        pre_weight_list1[i] = pre_weight_list.copy()
    for wii in range(wi):
        for woo in range(wo):
            for b in range(baby):
                for g in range(giant):
                    for r in range(int(repeat)):
                        for i in range(group_num):
                            temp_list = edge_list[i]
                            for t in temp_list:
                                for j in range(in_channel):
                                    pre_weight_list1[i][wii][woo][b][g][
                                    t * batch_size + temp * j + r * in_channel * temp:(
                                                                                          t + 1) * batch_size + temp * j + r * in_channel * temp] = 0
                                    pre_weight_list2[i][wii][woo][b][g][
                                    t * batch_size + temp * j + r * in_channel * temp:(
                                                                                          t + 1) * batch_size + temp * j + r * in_channel * temp] = 0
    for r in range(int(repeat)):
        for i in range(group_num):
            temp_list = edge_list[i]
            for t in temp_list:
                for j in range(in_channel):
                    pre_bias_list1[i][t * batch_size + temp * j + r * in_channel * temp:(
                                                                                            t + 1) * batch_size + temp * j + r * in_channel * temp] = 0
                    pre_bias_list2[i][t * batch_size + temp * j + r * in_channel * temp:(
                                                                                            t + 1) * batch_size + temp * j + r * in_channel * temp] = 0
    return pre_weight_list1, pre_bias_list1, pre_weight_list2, pre_bias_list2


def mask_top_or_bottom(pre_weight_list, b, wi, wo, height, width, batch_size, in_channel,
                       output_channel, layer):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    if layer == "downsample0":
        giant, baby = perfect_square_split(in_channel)
    else:
        baby, giant = perfect_square_split(in_channel)
    repeat_bias = int(slots / output_channel)
    repeat = int(output_channel / in_channel)
    temp = batch_size * num_in_cipher
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo))
    final_list = np.empty((num_in_cipher, num_in_cipher, wi, wo, baby, giant, slots))
    for i in range(num_in_cipher):
        for j in range(num_in_cipher):
            final_list[i][j] = pre_weight_list.copy()
    for k in range(num_in_cipher):
        for i in range(num_in_cipher):
            for wii in range(wi):
                for woo in range(wo):
                    for b in range(baby):
                        for g in range(giant):
                            for r in range(int(repeat)):
                                for j in range(in_channel):
                                    final_list[k][i][wii][woo][b][g][
                                    j * temp + r * in_channel * temp:j * temp + r * in_channel * temp + i * batch_size] = 0
                                    final_list[k][i][wii][woo][b][g][
                                    j * temp + r * in_channel * temp + (num_in_cipher - k) * batch_size:(
                                                                                                            j + 1) * temp + r * in_channel * temp] = 0
    return final_list


def final_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                 right_edge, bottom_edge, up_edge, index, cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer == "downsample0" or layer == "downsample1":
        a, b = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale,
                          index, cryptoContext)
    else:
        a = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,
                       cryptoContext)
        b = np.zeros((group_num, 2 ** 15))
    pre_weight_list = mask_top_or_bottom(a, b, wi, wo, height, width, batch_size, in_channel,
                                         output_channel, layer)
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo, layer)
    _, _, mask_weight_list2, mask_bias_list2 = pre_weight_edge2(right_edge, a, b, batch_size, in_channel,
                                                                output_channel, height, width, wi,
                                                                wo, layer)
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo, layer)
    _, _, mask_weight_list4, mask_bias_list4 = pre_weight_edge2(up_edge, a, b, batch_size, in_channel, output_channel,
                                                                height, width, wi,
                                                                wo, layer)
    final_bias = pre_bias_list1 * mask_bias_list3
    return pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, a


def final_weight2(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                  right_edge, bottom_edge, up_edge, index, cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    a, a2 = pre_weight2(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,
                        cryptoContext)
    b = np.zeros((group_num, 2 ** 15))
    b2 = np.zeros((group_num, 2 ** 15))
    pre_weight_list = mask_top_or_bottom(a, b, wi, wo, height, width, batch_size, in_channel,
                                         output_channel, layer)
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo, layer)
    _, _, mask_weight_list2, mask_bias_list2 = pre_weight_edge2(right_edge, a, b, batch_size, in_channel,
                                                                output_channel, height, width, wi,
                                                                wo, layer)
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo, layer)
    _, _, mask_weight_list4, mask_bias_list4 = pre_weight_edge2(up_edge, a, b, batch_size, in_channel, output_channel,
                                                                height, width, wi,
                                                                wo, layer)
    final_bias = pre_bias_list1 * mask_bias_list3
    pre_weight_lista = mask_top_or_bottom(a2, b2, wi, wo, height, width, batch_size, in_channel,
                                          output_channel, layer)
    pre_weight_list1a, pre_bias_list1a, mask_weight_list1a, mask_bias_list1a = pre_weight_edge2(left_edge, a2, b2,
                                                                                                batch_size,
                                                                                                in_channel,
                                                                                                output_channel,
                                                                                                height, width, wi,
                                                                                                wo, layer)
    _, _, mask_weight_list2a, mask_bias_list2a = pre_weight_edge2(right_edge, a2, b2, batch_size, in_channel,
                                                                  output_channel, height, width, wi,
                                                                  wo, layer)
    _, pre_bias_list2a, mask_weight_list3a, mask_bias_list3a = pre_weight_edge2(bottom_edge, a2, b2, batch_size,
                                                                                in_channel,
                                                                                output_channel, height, width, wi,
                                                                                wo, layer)
    _, _, mask_weight_list4a, mask_bias_list4a = pre_weight_edge2(up_edge, a2, b2, batch_size, in_channel,
                                                                  output_channel,
                                                                  height, width, wi,
                                                                  wo, layer)
    final_biasa = pre_bias_list1a * mask_bias_list3a
    return pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, pre_weight_lista, pre_weight_list1a, mask_weight_list1a, mask_weight_list2a, mask_weight_list3a, mask_weight_list4a, b2, pre_bias_list1a, pre_bias_list2a, final_biasa


def final_weight3(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                  right_edge, bottom_edge, up_edge, index, cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer == "downsample0" or layer == "downsample1":
        a, b = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale,
                          index, cryptoContext)
    else:
        a = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,
                       cryptoContext)
        b = np.zeros((group_num, 2 ** 15))
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo, layer)
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo, layer)
    return mask_weight_list1, mask_weight_list3


def final_weight4(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                  right_edge, bottom_edge, up_edge, index, cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer == "downsample0" or layer == "downsample1":
        a, b = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale,
                          index, cryptoContext)
    else:
        a = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,
                       cryptoContext)
        b = np.zeros((group_num, 2 ** 15))
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo, layer)
    _, _, mask_weight_list2, mask_bias_list2 = pre_weight_edge2(right_edge, a, b, batch_size, in_channel,
                                                                output_channel, height, width, wi,
                                                                wo, layer)
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo, layer)
    final_bias = pre_bias_list1 * mask_bias_list3
    return b, pre_bias_list1, pre_bias_list2, final_bias, a
