"""Model parameter extraction and final classifier stages."""

from .config import project_root  # Ensures runtime paths and DATA_DIR match the original script.

import math
import os

import numpy as np
import torch
import torch.fhe as fhe
from examples.resnet.gen_aespa_weights.HerPN import get_Aespa_MutalChannel_PAF_resnet18, get_Aespa_MutalChannel_PAF_resnet20

from .utils import min_padding_to_next_multiple_of_k, perfect_square_split


def get_weight_bias(layer, state_dict, index, cryptoContext, eps=1e-5):
    model = get_Aespa_MutalChannel_PAF_resnet20()
    if layer == "":
        W = model.conv1.weight
    elif layer == "downsample0":
        W = model.layer2[0].downsample[0].weight
    elif layer == "downsample1":
        W = model.layer3[0].downsample[0].weight
    else:
        layer_name, idx_str = layer.split("[")
        idx = int(idx_str.rstrip("]"))
        block = getattr(model, layer_name)[idx]
        W = getattr(block, f"conv{index}").weight
    Weight1 = np.empty((W.shape[0], W.shape[1], W.shape[2], W.shape[3]))
    Bias1 = np.empty((W.shape[0],))
    if layer == "":
        filename = cryptoContext.weight_path + "conv1bn1-A2" + '.bin'
        values = []
        if not os.path.isfile(filename):
            print(f"Failed to open file: {filename}")
            return values
        try:
            with open(filename, 'r') as file:
                for row in file:
                    for value in row.strip().split(','):
                        try:
                            num = float(value)
                            values.append(num * 1)
                        except ValueError:
                            print(f"unconvert:: {value}")
            temp = int(len(values) / W.shape[0])
            A = [0] * W.shape[0]
            for i in range(W.shape[0]):
                A[i] = values[i * temp]
        except IOError as e:
            print(f"error: {e}")
    elif layer == "downsample0":
        conv_weight = state_dict['layer2.0.downsample.0.weight']
        bn_weight = state_dict['layer2.0.downsample.1.weight']
        bn_bias = state_dict['layer2.0.downsample.1.bias']
        bn_running_mean = state_dict['layer2.0.downsample.1.running_mean']
        bn_running_var = state_dict['layer2.0.downsample.1.running_var']
        A = bn_weight / torch.sqrt(bn_running_var + eps)
        b = -(bn_weight * bn_running_mean / torch.sqrt(bn_running_var) + eps) + bn_bias
        A = model.layer2[0].downsample[1].weight / torch.sqrt(
            model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
        b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
            model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + \
            model.layer2[0].downsample[1].bias
        A1 = model.layer2[0].HerPN2.a2.detach() ** 0.5
        A = A * A1
        b = b * A1
    elif layer == "downsample1":
        conv_weight = state_dict['layer3.0.downsample.0.weight']
        bn_weight = state_dict['layer3.0.downsample.1.weight']
        bn_bias = state_dict['layer3.0.downsample.1.bias']
        bn_running_mean = state_dict['layer3.0.downsample.1.running_mean']
        bn_running_var = state_dict['layer3.0.downsample.1.running_var']
        A = bn_weight / torch.sqrt(bn_running_var + eps)
        b = -(bn_weight * bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
        A = model.layer3[0].downsample[1].weight / torch.sqrt(
            model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)
        b = -(model.layer3[0].downsample[1].weight * model.layer3[0].downsample[1].running_mean / torch.sqrt(
            model.layer3[0].downsample[1].running_var + model.layer3[0].downsample[1].eps)) + \
            model.layer3[0].downsample[1].bias
        A1 = model.layer3[0].HerPN2.a2.detach() ** 0.5
        A = A * A1
        b = b * A1
    else:
        layer_name, idx_str = layer.split("[")
        idx = int(idx_str.rstrip("]"))
        block = getattr(model, layer_name)[idx]
        temp_PAF = getattr(block, f"HerPN{index}")
        A = temp_PAF.a2.detach() ** 0.5
    if layer == "downsample0":
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for p in range(1):
                    for q in range(1):
                        Weight1[i, j, p, q] = W[i, j].reshape(1)[q + p * 3].detach() * A[i].detach()
        for i in range(W.shape[0]):
            Bias1[i] = b[i].item()
        return Weight1, Bias1
    elif layer == "downsample1":
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for p in range(1):
                    for q in range(1):
                        Weight1[i, j, p, q] = W[i, j].reshape(1)[q + p * 3].detach() * A[i].detach()
        for i in range(W.shape[0]):
            Bias1[i] = b[i].item()
        return Weight1, Bias1
    else:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for p in range(3):
                    for q in range(3):
                        Weight1[i, j, p, q] = W[i, j].reshape(9)[q + p * 3].detach() * A[i]
        return Weight1


@fhe.utils.profile_python_function
def averagepool(input, batch_size, height, width, wi, wo, in_channel, output_channel, cryptoContext, openfhe_context):
    N = 65536
    slots = int(N / 2)
    repeat = int(output_channel / in_channel)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    for i in range(wi):
        for j in range(wo):
            for k in range(group_num):
                if i == 0 and j == 0 and k == 0:
                    output = input[i][j][k]
                else:
                    output = fhe.homo_add(output, input[i][j][k], cryptoContext)
    for i in range(1, num_in_cipher):
        output = fhe.homo_add(output, fhe.homo_rotate(output.deep_copy(), i * batch_size, cryptoContext), cryptoContext)
    return output


@fhe.utils.profile_python_function
def fc(input_cipher, batch_size, height, width, wi, wo, in_channel, output_channel, cryptoContext, openfhe_context):
    model = get_Aespa_MutalChannel_PAF_resnet20()
    N = 65536
    slots = int(N / 2)
    output_channel = 16
    repeat = int(slots / output_channel)
    num_in_cipher = int(slots / (in_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    fc_weight = model.fc.weight.detach().numpy()
    fc_weight = fc_weight * (1 / 64)
    fc_bias = model.fc.bias.reshape(-1).detach().cpu().numpy()
    weight_pad = np.pad(fc_weight, ((0, 6), (0, 0)), mode='constant', constant_values=0)
    fc_pad = np.repeat(fc_bias, 512)
    baby = 4
    giant = 4
    temp = batch_size * num_in_cipher
    templist = np.zeros((baby, giant, slots))
    templist_encode = np.zeros((baby, giant), dtype=object)
    output_giant = np.empty(baby, dtype=object)
    for b in range(baby):
        for g in range(giant):
            for r in range(int(4)):
                for i in range(output_channel):
                    input = (r * output_channel + g + i) % in_channel
                    output = (b * giant + i) % output_channel
                    templist[b][g][
                    i * temp + r * output_channel * temp:(i + 1) * temp + r * output_channel * temp] = \
                        weight_pad[output][input]
    for i in range(baby):
        for j in range(giant):
            templist_encode[i][j] = fhe.encode(templist[i][j], "", 1, slots, False, cryptoContext)
    for g in range(giant):
        if g != 0:
            input_temp = fhe.homo_rotate(input_cipher.deep_copy(), g * temp, cryptoContext)
        else:
            input_temp = input_cipher.deep_copy()
        if g == 0:
            for b in range(baby):
                output_giant[b] = fhe.homo_mul_pt(input_temp, templist_encode[b][g], cryptoContext)
        else:
            for b in range(baby):
                output_giant[b] = fhe.homo_add(output_giant[b],
                                               fhe.homo_mul_pt(input_temp, templist_encode[b][g], cryptoContext),
                                               cryptoContext)
    for b in range(baby):
        if b == 0:
            output = output_giant[b]
        else:
            output = fhe.homo_add(output,
                                  fhe.homo_rotate(output_giant[b].deep_copy(), -giant * b * temp, cryptoContext),
                                  cryptoContext)
    output = fhe.homo_rescale(output, 1, cryptoContext)
    temp_cipher1 = output.deep_copy()
    temp_cipher2 = output.deep_copy()
    temp_cipher3 = output.deep_copy()
    output = fhe.homo_add(output, fhe.homo_rotate(temp_cipher1, 16 * temp, cryptoContext), cryptoContext)
    output = fhe.homo_add(output, fhe.homo_rotate(temp_cipher2, 16 * 2 * temp, cryptoContext), cryptoContext)
    output = fhe.homo_add(output, fhe.homo_rotate(temp_cipher3, 16 * 3 * temp, cryptoContext), cryptoContext)
    output = fhe.homo_add_pt(output, fhe.encode(fc_pad, "", 1, slots, False, cryptoContext), cryptoContext)
    return output


def choose_conv(batch_size, resnet):
    slots = 2 << 15
    ch1 = 16
    ch2 = 32
    ch3 = 64
    bs = 252
    alist = []
    blist = []
    clist = []

    def tets21(x):
        return math.ceil(x / 16)

    def tets12(x):
        return math.ceil(121 * x / 2048)

    for i in range(2048):
        if tets12(i) == tets21(i):
            alist.append(i)

    def tets1(x):
        return math.ceil(x / 16)

    def tets2(x):
        return math.ceil(9 * x / 256)

    for i in range(2048):
        if tets2(i) == tets1(i):
            blist.append(i)

    def tets3(x):
        return math.ceil(x / 32)

    def tets4(x):
        return math.ceil(9 * x / 512)

    for i in range(2048):
        if tets4(i) == tets3(i):
            clist.append(i)
    if resnet == 20:
        block_num = 6
    if resnet == 32:
        block_num = 10
    if resnet == 44:
        block_num = 14
    if resnet == 56:
        block_num = 18
    if resnet == 110:
        block_num = 36
    if batch_size <= 16:
        return [1, 1, 1]
    elif batch_size < 64:
        if batch_size in set(alist).intersection(set(blist), set(clist)):
            return [1, 1, 1]
        elif batch_size not in blist:
            if batch_size not in clist:
                b, g = perfect_square_split(int(2 ** 8 / batch_size))
                b1, g1 = perfect_square_split(16)
                if (batch_size / 2 ** 4) * 9 * (b + g - 2) * block_num + (batch_size / 8) * 4 * (
                    slots / (32 * batch_size)) * (math.ceil(36 / (slots / (32 * batch_size)))) > (
                    batch_size / 2 ** 4) * 9 * (b1 + g1 - 2) * block_num + 2 * 4 * (slots / (32 * batch_size)) * (
                    math.ceil(36 / (slots / (32 * batch_size)))):
                    return [2, 2, 2]
                else:
                    return [1, 2, 2]
            else:
                b, g = perfect_square_split(int(2 ** 8 / batch_size))
                b1, g1 = perfect_square_split(16)
                if (batch_size / 2 ** 4) * 9 * (b + g - 2) * block_num + (batch_size / 8) * 4 * (
                    slots / (32 * batch_size)) * (math.ceil(36 / (slots / (32 * batch_size)))) > (
                    batch_size / 2 ** 4) * 9 * (b1 + g1 - 2) * block_num + 2 * 4 * (slots / (32 * batch_size)) * (
                    math.ceil(36 / (slots / (32 * batch_size)))):
                    return [2, 2, 1]
                else:
                    return [1, 2, 1]
    elif batch_size == 64:
        b, g = perfect_square_split(32)
        b1, g1 = perfect_square_split(int(512 / batch_size))
        plan011 = 4 * 9 * (b + g - 2) * (32 / (2048 / batch_size)) * block_num + 9 * (batch_size / 16) * (
            b1 + g1 - 2) * block_num + 9 * (
                      math.ceil(batch_size / 16) - math.ceil(9 * batch_size / 256)) * bs * block_num + 144
        plan121 = 72 * block_num + 3 * 16 * 4 * 8 + 3 * 9 * 10 * block_num
        plan221 = 4 * 9 * 6 * block_num + 2 * 3 * 16 * 4 + 3 * 9 * 10 * block_num
        plan111 = 72 * block_num + 2 * 4 * 16 * 9 * 4 + 4 * 9 * 4 * block_num + 9 * (
            math.ceil(batch_size / 16) - math.ceil(9 * batch_size / 256)) * bs * block_num + 144
        min_value = min(plan011, plan121, plan221, plan111)
        if min_value == plan011:
            return [0, 1, 1]
        elif min_value == plan121:
            return [1, 2, 1]
        elif min_value == plan221:
            return [2, 2, 1]
        else:
            return [1, 1, 1]
    elif batch_size > 64:
        if batch_size in alist:
            b, g = perfect_square_split(int(2 ** 8 / batch_size))
            b1, g1 = perfect_square_split(16)
            if (batch_size / 2 ** 4) * 9 * (b + g - 2) * block_num + (batch_size / 8) * 4 * (
                slots / (32 * batch_size)) * (math.ceil(36 / (slots / (32 * batch_size)))) > (
                batch_size / 2 ** 4) * 9 * (b1 + g1 - 2) * block_num + 2 * 4 * (slots / (32 * batch_size)) * (
                math.ceil(36 / (slots / (32 * batch_size)))):
                return [2, 2, 2]
            else:
                return [1, 2, 2]
        else:
            return [2, 2, 2]
