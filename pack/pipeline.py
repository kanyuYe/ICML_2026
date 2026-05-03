"""Main PackCNN inference pipeline."""

from .config import project_root  

import os
import sys
import time

import numpy as np
import torch
import torch.fhe as fhe

from .bsgs import final_weight, final_weight2, final_weight3, final_weight4
from .conv import (
    conv2_batch_extend,
    conv2_batch_extend2,
    conv_batch2,
    conv_batch2_mod2,
    conv_downsampling,
    down_need,
    downsampling,
    judge_rotate,
    output_judge_is_edge,
    output_judge_is_edge2,
    output_judge_is_edge3,
    output_judge_is_edge4,
    pre_mask2,
    zero_num,
)
from .crypto import batch_homo_bs, batch_homo_relu, batch_homo_relu2, batch_homo_relu3
from .data import batch_input
from .encoding import load_weight
from .model import averagepool, fc


def batch_CNN(loadorsave, pkl_dir):
    DATA_DIR = os.environ["DATA_DIR"]
    rotate_index_list = [128, 128 * 2, 128 * 3, 128 * 4, 128 * 5, 128 * 6, 128 * 7, -128, -128 * 2, -128 * 3, -128 * 4,
                         -128 * 5, -128 * 6, -128 * 7, 2048, -2048, 2048 * 2, 2048 * -2, 2048 * 3, 2048 * -4, 2048 * -8,
                         2048 * -12, 1024, -1024, 3072, -2048 * 3, -4096, -2 * 4096, -3 * 4096, -4 * 4096, -5 * 4096,
                         -6 * 4096, -7 * 4096, 512 * 1, 512 * 3, 512 * 5, 512 * 7, 16 * 512, 32 * 512, 48 * 512]
    maxLevelsRemaining = 14
    logBsSlots_list = [15]
    logN = 16
    dnum = 3
    dcrtBits = 52
    firstMod = 55
    levelBudget_list = [[4, 4]]
    rescaleTech = "FIXEDMANUAL"
    relu_degree = 180
    slots = 2 ** 15
    # loadorsave=0
    if loadorsave == 0:
        SAVE_MIDDLE = True
        DIRECT_LOAD = False
    else:
        SAVE_MIDDLE = False
        DIRECT_LOAD = True
    device = "cuda"
    start = time.time()
    print(time.ctime(start))
    config = torch.fhe.config.Config(AUTO_LOAD_KEYS=True,
                                     SAVE_MIDDLE=SAVE_MIDDLE
                                     )
    cryptoContext, openfhe_context = (
        fhe.try_load_context(maxLevelsRemaining, rotate_index_list, logBsSlots_list, logN, dnum, dcrtBits, firstMod,
                             levelBudget_list, "SPARSE_TERNARY", rescaleTech, device, save_dir=DATA_DIR,
                             config=config))
    cryptoContext.DIRECT_LOAD = DIRECT_LOAD
    cryptoContext.weight_path = os.path.join(project_root, "PackCNN", "data", "weights_aespa_20/")
    temp_finename = os.path.join(project_root, "PackCNN", "data", "cifar10_resnet20-4118986f.pt")
    ckpt = torch.load(temp_finename, map_location="cpu")
    pre_encode_type = "middle"
    cryptoContext.pre_encode_type = pre_encode_type
    pkl_path = DATA_DIR + "/encode_20260212_150521.pkl"
    pkl_path = DATA_DIR + pkl_dir
    start_load = time.time()
    print("Performing preprocessing or preloading of weight values for subsequent computation...")
    if loadorsave == 0:
        pass
    else:
        load_weight(pkl_path, cryptoContext)
    end_load = time.time()
    print("load weight time:", f" {end_load - start_load:.4f} seconds.")
    pre_mask = pre_mask2(3, 3, 32, 32, 128, 16, cryptoContext, 1)
    pre_mask22 = pre_mask2(3, 3, 16, 16, 128, 32, cryptoContext, 2)
    state_dict = ckpt.get("state_dict", ckpt)
    left_edge = output_judge_is_edge(3, 3, 32, 32, 128, 16)
    right_edge = output_judge_is_edge2(3, 3, 32, 32, 128, 16)
    bottom_edge = output_judge_is_edge3(3, 3, 32, 32, 128, 16)
    up_edge = output_judge_is_edge4(3, 3, 32, 32, 128, 16)
    edge_list = judge_rotate(3, 3, 32, 32, 128, 16)
    zero_num_temp = zero_num(3, 3, 32, 32, 128, 16)
    left_edge2 = output_judge_is_edge(3, 3, 16, 16, 128, 32)
    right_edge2 = output_judge_is_edge2(3, 3, 16, 16, 128, 32)
    bottom_edge2 = output_judge_is_edge3(3, 3, 16, 16, 128, 32)
    up_edge2 = output_judge_is_edge4(3, 3, 16, 16, 128, 32)
    edge_list2 = judge_rotate(3, 3, 16, 16, 128, 32)
    zero_num_temp2 = zero_num(3, 3, 16, 16, 128, 32)
    left_edge3 = output_judge_is_edge(3, 3, 8, 8, 128, 64)
    right_edge3 = output_judge_is_edge2(3, 3, 8, 8, 128, 64)
    bottom_edge3 = output_judge_is_edge3(3, 3, 8, 8, 128, 64)
    up_edge3 = output_judge_is_edge4(3, 3, 8, 8, 128, 64)
    edge_list3 = judge_rotate(3, 3, 8, 8, 128, 64)
    zero_num_temp3 = zero_num(3, 3, 8, 8, 128, 64)
    if loadorsave == 0:
        input, batch_label = batch_input("save", 128, openfhe_context, cryptoContext)
    else:
        input, batch_label = batch_input("load", 128, openfhe_context, cryptoContext)
    scale = 1
    time1 = time.time()

    def initial_layer0():
        conv1_mode = "save"
        if conv1_mode == "save":
            # NOTE: The required .pkl files have already been generated and stored in advance,
            # so we directly load them by default.
            # If you want to regenerate the .pkl files, please uncomment the code block below.
            # Meanwhile, you must comment out all the subsequent np.zeros() initialization blocks
            pre_weight_list = np.zeros((16, 16, 3, 3, 2, 2, 32768))
            pre_weight_list1 = np.zeros((8, 3, 3, 2, 2, 32768))
            mask_weight_list1 = np.zeros((8, 3, 3, 2, 2, 32768))
            mask_weight_list2 = np.zeros((8, 3, 3, 2, 2, 32768))
            mask_weight_list3 = np.zeros((8, 3, 3, 2, 2, 32768))
            mask_weight_list4 = np.zeros((8, 3, 3, 2, 2, 32768))
            b = np.zeros((8, 32768))
            pre_bias_list1 = np.zeros((8, 32768))
            pre_bias_list2 = np.zeros((8, 32768))
            final_bias = np.zeros((8, 32768))

            # NOTE: If the .pkl files have already been generated, the following code can be commented out.
            if loadorsave == 0:
                pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                    128, 4, state_dict, 16, 32, 32, 3, 3, "", scale, left_edge, right_edge, bottom_edge, up_edge, 1,
                    cryptoContext)
            # NOTE: If the .pkl files have already been generated, the above code can be commented out.

            output = conv_batch2(input, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list, mask_weight_list1,
                                 mask_weight_list2, mask_weight_list3,
                                 mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 4, 16, 32, 32,
                                 3,
                                 3, cryptoContext, openfhe_context, True, None, f"0")
            output = batch_homo_relu(output, "conv1bn1", cryptoContext, mask_weight_list1, mask_weight_list3)
        return output

    def layer1(output, block_num):
        for i in range(block_num):
            templist = output.copy()
            conv2_mode = "save"
            if conv2_mode == "save":
                scale = 1
                pre_weight_list = np.zeros((16, 16, 3, 3, 4, 4, 32768))
                pre_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list2 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list3 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list4 = np.zeros((8, 3, 3, 4, 4, 32768))
                b = np.zeros((8, 32768))
                pre_bias_list1 = np.zeros((8, 32768))
                pre_bias_list2 = np.zeros((8, 32768))
                final_bias = np.zeros((8, 32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave == 0:
                    pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                        128, 16, state_dict, 16, 32, 32, 3, 3, f"layer1[{i}]", scale, left_edge, right_edge,
                        bottom_edge, up_edge, 1, cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2(output, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 16, 16, 32,
                                     32, 3, 3,
                                     cryptoContext, openfhe_context, True, None, f"1_{i * 2}")
                output = batch_homo_relu(output, f"layer{i + 1}-conv{1}bn{1}", cryptoContext, mask_weight_list1,
                                         mask_weight_list3)
            if conv2_mode == "save":
                scale = 1
                pre_weight_list = np.zeros((16, 16, 3, 3, 4, 4, 32768))
                pre_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list2 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list3 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list4 = np.zeros((8, 3, 3, 4, 4, 32768))
                b = np.zeros((8, 32768))
                pre_bias_list1 = np.zeros((8, 32768))
                pre_bias_list2 = np.zeros((8, 32768))
                final_bias = np.zeros((8, 32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave == 0:
                    pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                        128, 16, state_dict, 16, 32, 32, 3, 3, f"layer1[{i}]", scale, left_edge, right_edge,
                        bottom_edge, up_edge, 2, cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2(output, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 16, 16, 32,
                                     32, 3, 3,
                                     cryptoContext, openfhe_context, False, templist, f"1_{i * 2 + 1}")
                output = batch_homo_relu(output, f"layer{i + 1}-conv{2}bn{2}", cryptoContext, mask_weight_list1,
                                         mask_weight_list3)
        return output

    def layer2(output, block_num):
        for i in range(block_num):
            templist = output.copy()
            conv2_mode = "save"
            if i == 0:
                scale = 1
                if conv2_mode == "save":
                    pre_weight_list = np.zeros((16, 16, 3, 3, 4, 4, 32768))
                    pre_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list2 = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list3 = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list4 = np.zeros((8, 3, 3, 4, 4, 32768))
                    b = np.zeros((8, 32768))
                    pre_bias_list1 = np.zeros((8, 32768))
                    pre_bias_list2 = np.zeros((8, 32768))
                    final_bias = np.zeros((8, 32768))
                    pre_weight_lista = np.zeros((16, 16, 3, 3, 4, 4, 32768))
                    pre_weight_list1a = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list1a = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list2a = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list3a = np.zeros((8, 3, 3, 4, 4, 32768))
                    mask_weight_list4a = np.zeros((8, 3, 3, 4, 4, 32768))
                    b2 = np.zeros((8, 32768))
                    pre_bias_list1a = np.zeros((8, 32768))
                    pre_bias_list2a = np.zeros((8, 32768))
                    final_biasa = np.zeros((8, 32768))
                    down_need1 = down_need(3, 3, 32, 32, 128, 16, 32)

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave == 0:
                        pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, pre_weight_lista, pre_weight_list1a, mask_weight_list1a, mask_weight_list2a, mask_weight_list3a, mask_weight_list4a, b2, pre_bias_list1a, pre_bias_list2a, final_biasa = final_weight2(
                            128, 16, state_dict, 16, 32, 32, 3, 3, f"layer2[{i}]", scale, left_edge, right_edge,
                            bottom_edge, up_edge, 1, cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv2_batch_extend(output, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list,
                                                mask_weight_list1,
                                                mask_weight_list2, mask_weight_list3,
                                                mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128,
                                                16, 32, 32,
                                                32, 3, 3,
                                                cryptoContext, openfhe_context, True, None, f"2_{i * 2}", down_need1,
                                                pre_weight_lista, pre_weight_list1a, mask_weight_list1a,
                                                mask_weight_list2a, mask_weight_list3a, mask_weight_list4a, b2,
                                                pre_bias_list1a, pre_bias_list2a, final_biasa)
                output = downsampling(output, pre_mask, 3, 3, 32, 32, 128, 16, 32, 2, cryptoContext, openfhe_context)
                mask_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list3 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list1, mask_weight_list3 = final_weight3(
                    128, 32, state_dict, 32, 16, 16, 3, 3, f"layer2[1]", scale, left_edge2, right_edge2, bottom_edge2,
                    up_edge2, 1, cryptoContext)
                output = batch_homo_relu2(output, f"layer{i + 4}-conv{1}bn{1}", cryptoContext, mask_weight_list1,
                                          mask_weight_list3)
            else:
                if conv2_mode == "save":
                    scale = 1
                    pre_weight_list = np.zeros((8, 8, 3, 3, 8, 4, 32768))
                    pre_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list2 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list3 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list4 = np.zeros((5, 3, 3, 8, 4, 32768))
                    b = np.zeros((5, 32768))
                    pre_bias_list1 = np.zeros((5, 32768))
                    pre_bias_list2 = np.zeros((5, 32768))
                    final_bias = np.zeros((5, 32768))

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave == 0:
                        pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                            128, 32, state_dict, 32, 16, 16, 3, 3, f"layer2[{i}]", scale, left_edge2, right_edge2,
                            bottom_edge2,
                            up_edge2, 1, cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv_batch2_mod2(output, zero_num_temp2, pre_weight_list, pre_weight_list1, edge_list2,
                                              mask_weight_list1,
                                              mask_weight_list2, mask_weight_list3,
                                              mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128,
                                              32, 32, 16,
                                              16, 3, 3,
                                              cryptoContext, openfhe_context, True, None, f"2_{i * 2}",
                                              )
                output = batch_homo_relu2(output, f"layer{i + 4}-conv{1}bn{1}", cryptoContext, mask_weight_list1,
                                          mask_weight_list3)
            if conv2_mode == "save":
                if i == 0:
                    weight_down = np.zeros((1, 1))

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave == 0:
                        bias_down, pre_bias_list2, pre_bias_list3, final_bias, weight_down = final_weight4(128, 32,
                                                                                                           state_dict,
                                                                                                           32, 16, 16,
                                                                                                           3, 3,
                                                                                                           f"downsample0",
                                                                                                           scale,
                                                                                                           left_edge2,
                                                                                                           right_edge2,
                                                                                                           bottom_edge2,
                                                                                                           up_edge2, 2,
                                                                                                           cryptoContext)
                        temp_finename = os.path.join(project_root, "PackCNN", "data", "params1.npz")
                        np.savez(temp_finename,
                                 bias_down=bias_down,
                                 pre_bias_list2=pre_bias_list2,
                                 pre_bias_list3=pre_bias_list3,
                                 final_bias=final_bias)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.
                    temp_finename = os.path.join(project_root, "PackCNN", "data", "params1.npz")
                    data = np.load(temp_finename)
                    bias_down = data["bias_down"]
                    pre_bias_list2 = data["pre_bias_list2"]
                    pre_bias_list3 = data["pre_bias_list3"]
                    final_bias = data["final_bias"]
                    conv2_test = np.empty((templist.shape[0] * 2, *templist.shape[1:]), dtype=object)
                    conv2_test[:templist.shape[0]] = templist
                    conv2_test[templist.shape[0]:] = templist
                    templist = downsampling(conv2_test, pre_mask, 3, 3, 32, 32, 128, 16, 32, 2, cryptoContext,
                                            openfhe_context)
                    templist = conv_downsampling(f"layer2", templist, weight_down, bias_down, pre_bias_list2,
                                                 pre_bias_list3, final_bias, cryptoContext, openfhe_context, 16, 16, 3,
                                                 3, 128, 32, 32)
                scale = 1
                pre_weight_list = np.zeros((8, 8, 3, 3, 8, 4, 32768))
                pre_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list2 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list3 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list4 = np.zeros((5, 3, 3, 8, 4, 32768))
                b = np.zeros((5, 32768))
                pre_bias_list1 = np.zeros((5, 32768))
                pre_bias_list2 = np.zeros((5, 32768))
                final_bias = np.zeros((5, 32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave == 0:
                    pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                        128, 32, state_dict, 32, 16, 16, 3, 3, f"layer2[{i}]", scale, left_edge2, right_edge2,
                        bottom_edge2, up_edge2, 2, cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2_mod2(output, zero_num_temp2, pre_weight_list, pre_weight_list1, edge_list2,
                                          mask_weight_list1,
                                          mask_weight_list2, mask_weight_list3,
                                          mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 32, 32,
                                          16,
                                          16, 3, 3,
                                          cryptoContext, openfhe_context, False, templist, f"2_{i * 2 + 1}")
            if i == 0:
                index = 0
                output = batch_homo_bs(output, index, logBsSlots_list, levelBudget_list, cryptoContext)
            output = batch_homo_relu2(output, f"layer{i + 4}-conv{2}bn{2}", cryptoContext, mask_weight_list1,
                                      mask_weight_list3)
        return output

    def layer3(output, block_num):
        for i in range(block_num):
            templist = output.copy()
            conv2_mode = "save"
            if i == 0:
                if conv2_mode == "save":
                    scale = 1

                    pre_weight_list = np.zeros((8, 8, 3, 3, 8, 4, 32768))
                    pre_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list2 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list3 = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list4 = np.zeros((5, 3, 3, 8, 4, 32768))
                    b = np.zeros((5, 32768))
                    pre_bias_list1 = np.zeros((5, 32768))
                    pre_bias_list2 = np.zeros((5, 32768))
                    final_bias = np.zeros((5, 32768))
                    pre_weight_lista = np.zeros((8, 8, 3, 3, 8, 4, 32768))
                    pre_weight_list1a = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list1a = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list2a = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list3a = np.zeros((5, 3, 3, 8, 4, 32768))
                    mask_weight_list4a = np.zeros((5, 3, 3, 8, 4, 32768))
                    b2 = np.zeros((5, 32768))
                    pre_bias_list1a = np.zeros((5, 32768))
                    pre_bias_list2a = np.zeros((5, 32768))
                    final_biasa = np.zeros((5, 32768))
                    down_need1 = down_need(3, 3, 16, 16, 128, 32, 64)

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave == 0:
                        pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, pre_weight_lista, pre_weight_list1a, mask_weight_list1a, mask_weight_list2a, mask_weight_list3a, mask_weight_list4a, b2, pre_bias_list1a, pre_bias_list2a, final_biasa = final_weight2(
                            128, 32, state_dict, 32, 16, 16, 3, 3, f"layer3[{i}]", scale, left_edge2, right_edge2,
                            bottom_edge2, up_edge2, 1, cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv2_batch_extend2(output, zero_num_temp2, pre_weight_list, pre_weight_list1, edge_list2,
                                                 mask_weight_list1,
                                                 mask_weight_list2, mask_weight_list3,
                                                 mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128,
                                                 32, 64, 16,
                                                 16, 3, 3,
                                                 cryptoContext, openfhe_context, True, None, f"3_{i * 2}", down_need1,
                                                 pre_weight_lista, pre_weight_list1a, mask_weight_list1a,
                                                 mask_weight_list2a, mask_weight_list3a, mask_weight_list4a, b2,
                                                 pre_bias_list1a, pre_bias_list2a, final_biasa)
                output = downsampling(output, pre_mask22, 3, 3, 16, 16, 128, 32, 64, 2, cryptoContext, openfhe_context)
                mask_weight_list1 = np.zeros((3, 3, 3, 8, 8, 32768))
                mask_weight_list3 = np.zeros((3, 3, 3, 8, 8, 32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave == 0:
                    mask_weight_list1, mask_weight_list3 = final_weight3(
                        128, 64, state_dict, 64, 8, 8, 3, 3, f"layer3[1]", scale, left_edge3, right_edge3, bottom_edge3,
                        up_edge3, 1, cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = batch_homo_relu3(output, f"layer{i + 7}-conv{1}bn{1}", cryptoContext, mask_weight_list1,
                                          mask_weight_list3)
            else:
                if conv2_mode == "save":
                    scale = 1

                    pre_weight_list = np.zeros((4, 4, 3, 3, 8, 8, 32768))
                    pre_weight_list1 = np.zeros((3, 3, 3, 8, 8, 32768))
                    mask_weight_list1 = np.zeros((3, 3, 3, 8, 8, 32768))
                    mask_weight_list2 = np.zeros((3, 3, 3, 8, 8, 32768))
                    mask_weight_list3 = np.zeros((3, 3, 3, 8, 8, 32768))
                    mask_weight_list4 = np.zeros((3, 3, 3, 8, 8, 32768))
                    b = np.zeros((3, 32768))
                    pre_bias_list1 = np.zeros((3, 32768))
                    pre_bias_list2 = np.zeros((3, 32768))
                    final_bias = np.zeros((3, 32768))

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave == 0:
                        pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                            128, 64, state_dict, 64, 8, 8, 3, 3, f"layer3[{i}]", scale, left_edge3, right_edge3,
                            bottom_edge3,
                            up_edge3, 1, cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv_batch2(output, zero_num_temp3, pre_weight_list, pre_weight_list1, edge_list3,
                                         mask_weight_list1,
                                         mask_weight_list2, mask_weight_list3,
                                         mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128,
                                         64, 64, 8,
                                         8, 3, 3,
                                         cryptoContext, openfhe_context, True, None, f"3_{i * 2}",
                                         )
                output = batch_homo_relu3(output, f"layer{i + 7}-conv{1}bn{1}", cryptoContext, mask_weight_list1,
                                          mask_weight_list3)
            if conv2_mode == "save":
                if i == 0:
                    weight_down = np.zeros((1, 1))

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave == 0:
                        bias_down, pre_bias_list2, pre_bias_list3, final_bias, weight_down = final_weight4(128, 64,
                                                                                                           state_dict,
                                                                                                           64, 8, 8, 3,
                                                                                                           3,
                                                                                                           f"downsample1",
                                                                                                           scale,
                                                                                                           left_edge3,
                                                                                                           right_edge3,
                                                                                                           bottom_edge3,
                                                                                                           up_edge3, 2,
                                                                                                           cryptoContext)
                        temp_finename = os.path.join(project_root, "PackCNN", "data", "params2.npz")
                        np.savez(temp_finename,
                                 bias_down=bias_down,
                                 pre_bias_list2=pre_bias_list2,
                                 pre_bias_list3=pre_bias_list3,
                                 final_bias=final_bias)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.
                    temp_finename = os.path.join(project_root, "PackCNN", "data", "params2.npz")
                    data = np.load(temp_finename)
                    bias_down = data["bias_down"]
                    pre_bias_list2 = data["pre_bias_list2"]
                    pre_bias_list3 = data["pre_bias_list3"]
                    final_bias = data["final_bias"]
                    conv2_test = np.empty((templist.shape[0] * 2, *templist.shape[1:]), dtype=object)
                    conv2_test[:templist.shape[0]] = templist
                    conv2_test[templist.shape[0]:] = templist
                    templist = downsampling(conv2_test, pre_mask22, 3, 3, 16, 16, 128, 32, 64, 2, cryptoContext,
                                            openfhe_context)
                    templist = conv_downsampling(f"layer3", templist, weight_down, bias_down, pre_bias_list2,
                                                 pre_bias_list3, final_bias, cryptoContext, openfhe_context, 8, 8, 3, 3,
                                                 128, 64, 64)
                scale = 1
                pre_weight_list = np.zeros((4, 4, 3, 3, 8, 8, 32768))
                pre_weight_list1 = np.zeros((3, 3, 3, 8, 8, 32768))
                mask_weight_list1 = np.zeros((3, 3, 3, 8, 8, 32768))
                mask_weight_list2 = np.zeros((3, 3, 3, 8, 8, 32768))
                mask_weight_list3 = np.zeros((3, 3, 3, 8, 8, 32768))
                mask_weight_list4 = np.zeros((3, 3, 3, 8, 8, 32768))
                b = np.zeros((3, 32768))
                pre_bias_list1 = np.zeros((3, 32768))
                pre_bias_list2 = np.zeros((3, 32768))
                final_bias = np.zeros((3, 32768))

                # NOTE: If the .pkl files have already been generated, the floowing code can be commented out.
                if loadorsave == 0:
                    pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, _ = final_weight(
                        128, 64, state_dict, 64, 8, 8, 3, 3, f"layer3[{i}]", scale, left_edge3, right_edge3,
                        bottom_edge3, up_edge3, 2, cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2(output, zero_num_temp3, pre_weight_list, pre_weight_list1, edge_list3,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 64, 64, 8,
                                     8, 3, 3,
                                     cryptoContext, openfhe_context, False, templist, f"3_{i * 2 + 1}")
            if i == 0:
                index = 0
                output = batch_homo_bs(output, index, logBsSlots_list, levelBudget_list, cryptoContext)
            output = batch_homo_relu3(output, f"layer{i + 7}-conv{2}bn{2}", cryptoContext, mask_weight_list1,
                                      mask_weight_list3)
        return output

    time_start = time.time()
    print("Starting privacy-preserving inference...")
    layer0 = initial_layer0()
    print("layer0 finished")
    output = layer1(layer0, 3)
    print("layer1 finished")
    layer2 = layer2(output, 3)
    print("layer2 finished")
    layer3 = layer3(layer2, 3)
    print("layer3 finished")
    average = averagepool(layer3, 128, 8, 8, 3, 3, 32, 64, cryptoContext, openfhe_context)
    fc1 = fc(average, 128, 8, 8, 3, 3, 64, 10, cryptoContext, openfhe_context)
    time_end = time.time() - time_start
    print(f"Ciphertext inference finished. Total time (including encoding): {time_end:.4f} s "
          f"(encoding time is excluded in our experiments for both our method and the baseline).")
    aaa = openfhe_context.decrypt(fc1).cpu().numpy().reshape(-1)
    mat = aaa.reshape(64, 512).T
    sub = mat[:, :10]
    batch_size = 128
    correct = 0
    FHE_result = np.argmax(sub, axis=1)
    for i in range(batch_size):
        print(
            f"For image {i}, the inference result is {FHE_result[i]}, and the correct label is {int(batch_label[i])}.")
        if FHE_result[i] == batch_label[i]:
            correct += 1
    print(f"Correct: {correct}/{batch_size}, Accuracy: {(correct / batch_size) * 100:.2f}%")
    # print(f"\n\ncorrect/total: {correct}/{batch_size}")
    return 0


def main(argv=None):
    if argv is None:
        argv = sys.argv
    loadorsave = int(argv[-2])
    pkl_dir = argv[-1]
    return batch_CNN(loadorsave, pkl_dir)
