from .config import project_root  

import os
import pickle
import time

import numpy as np
import torch
import torch.fhe as fhe


def dump_cts_nd(ct_list, ct_list_shape, file_path):
    assert len(ct_list_shape) >= 1, "ct_list_shape must be non-empty"

    dump_ct = []

    def _dump_one(ct, idx):
        if ct is None:
            print("None at", idx)
            return
        tmp = ct.deep_copy().cpu()
        tmp.cv = [tmp.cv[0].numpy(), tmp.cv[1].numpy()]
        dump_ct.append(tmp)

    def _traverse(node, shape, idx_prefix):
        if len(shape) == 0:
            _dump_one(node, idx_prefix)
            return

        limit = shape[0]
        if node is None:
            print("None subtree at", idx_prefix)
            return

        n = min(limit, len(node))
        for i in range(n):
            _traverse(node[i], shape[1:], idx_prefix + (i,))

    _traverse(ct_list, ct_list_shape, ())

    with open(file_path, "wb") as f:
        pickle.dump({"cts": dump_ct, "ct_list_shape": ct_list_shape}, f)


def dump_cts(ct_list, ct_list_shape, file_path):
    assert len(ct_list_shape) == 2 or len(ct_list_shape) == 1, "only support two/one dim input ct_list now"
    dump_ct = []
    if len(ct_list_shape) == 2:
        for i, row in enumerate(ct_list):
            if i >= ct_list_shape[0]:
                break
            for j, ct in enumerate(row):
                if j >= ct_list_shape[1]:
                    break
                if ct is None:
                    print("index", j)
                else:
                    tmp = ct.deep_copy().cpu()
                    tmp.cv = [tmp.cv[0].numpy(), tmp.cv[1].numpy()]
                    dump_ct.append(tmp)
    else:
        for j, ct in enumerate(ct_list):
            if j >= ct_list_shape[0]:
                break
            if ct is None:
                print("index", j)
            else:
                tmp = ct.deep_copy().cpu()
                tmp.cv = [tmp.cv[0].numpy(), tmp.cv[1].numpy()]
                dump_ct.append(tmp)

    with open(file_path, "wb") as f:
        pickle.dump({"cts": dump_ct, "ct_list_shape": ct_list_shape}, f)


def load_cts(file_path, device="cpu"):
    data = pickle.load(open(file_path, "rb"))
    ct_list, ct_list_shape = data["cts"], data["ct_list_shape"]
    ct_list = reshape_ct_list(ct_list, ct_list_shape)
    print(f"len(ct_list) {len(ct_list)}")
    if (len(ct_list_shape) > 1):
        print(f"len(ct_list[0]) {len(ct_list[0])}")

    for row in ct_list:
        if len(ct_list_shape) == 1:
            row = [row]
        for ct in row:
            ct.cv = [torch.from_numpy(ct.cv[0]).to(device), torch.from_numpy(ct.cv[1]).to(device)]

    return ct_list, ct_list_shape


def reshape_ct_list(ct_list, ct_list_shape):
    if len(ct_list_shape) == 1:
        return ct_list

    if ct_list_shape[0] == 1:
        reshaped_ct_list = [ct_list]
    elif ct_list_shape[0] == 2:
        half_len_ct_list = len(ct_list) // 2
        reshaped_ct_list = [ct_list[:half_len_ct_list], ct_list[half_len_ct_list:]]
    else:
        raise Exception("ct_list shape is not supported.")

    return reshaped_ct_list


def load_cts_nd(file_path, device="cpu", return_as="ndarray"):
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    flat_cts = data["cts"]
    ct_list_shape = data["ct_list_shape"]

    total_elements = np.prod(ct_list_shape)
    assert len(flat_cts) == total_elements, \
        f"number is error"

    for ct in flat_cts:
        if ct is not None:
            ct.cv = [
                torch.from_numpy(ct.cv[0]).to(device),
                torch.from_numpy(ct.cv[1]).to(device)
            ]

    if return_as == "ndarray":
        ct_array = np.array(flat_cts, dtype=object).reshape(ct_list_shape)
        return ct_array, ct_list_shape

    elif return_as == "nested_list":
        def _reshape_flat_to_nested(flat_list, shape):
            if len(shape) == 1:
                return flat_list
            sub_size = np.prod(shape[1:])
            nested = []
            for i in range(shape[0]):
                start = i * sub_size
                end = start + sub_size
                nested.append(_reshape_flat_to_nested(flat_list[start:end], shape[1:]))
            return nested

        ct_nested = _reshape_flat_to_nested(flat_cts, ct_list_shape)
        return ct_nested, ct_list_shape
    else:
        raise ValueError("return_as must be 'ndarray' or 'nested_list'")


def use_checkpoint(file_name, mode, cipher_list, ct_list_shape):
    if mode == "SAVE_CHECKPOINT":
        print("beging saving checkpoint")
        dump_cts_nd(cipher_list, ct_list_shape, file_name)
        print(f"end saving checkpoint into {file_name}")
    elif mode == "LOAD_CHECKPOINT":
        print(f"Loading checkpoint {file_name}")
        if os.path.exists(file_name):
            cipher_list, dim = load_cts_nd(file_name, "cuda")
            print("shape of ct list", dim)
            return cipher_list
        else:
            print("file no exist")

        print("end loading checkpoint")


def load_weight(encode_weight_path, cryptoContext):
    if cryptoContext.DIRECT_LOAD:
        time_open = time.time()
        with open(encode_weight_path, 'rb') as f:
            pre_encoded = pickle.load(f)
        time_over = time.time()
        load_checkpoint = getattr(cryptoContext, "LOAD_CHECKPOINT", False)
        torch.cuda.synchronize()
        for key, _ in pre_encoded.items():
            if cryptoContext.pre_encode_type == "middle":
                if load_checkpoint:
                    _ = fhe.encode(pre_encoded[key], key, 0, pre_encoded[key].slots, False,
                                   cryptoContext)
                pre_encoded[key].encoded_values = torch.tensor(pre_encoded[key].encoded_values, device="cuda")
            elif cryptoContext.pre_encode_type == "end":
                pre_encoded[key].cv = [torch.tensor(pre_encoded[key].cv[0], dtype=torch.uint64, device="cuda")]
        torch.cuda.synchronize()
        cryptoContext.pre_encoded = pre_encoded
        cryptoContext.LOAD_CHECKPOINT = False
    else:
        pass


def read_values_from_file(i, j, k, val_name, level, slots, cryptoContext, mask, scale=1.0):
    if val_name == "layer4-conv1bn1-n1" or val_name == "layer4-conv1bn1-n2":
        if cryptoContext.DIRECT_LOAD:
            full_name = "{}_{}_{}".format(val_name, level, slots)
            if cryptoContext.pre_encode_type == "middle":
                name = f"{val_name}{i}{j}{k}"
            else:
                name = full_name
            return fhe.encode(cryptoContext.pre_encoded[name], name, level, slots, False, cryptoContext)
        else:
            values = []
            filename = cryptoContext.weight_path + val_name + '.bin'
            if not os.path.isfile(filename):
                print(f"Failed to open file: {filename}")
                return values
            try:
                with open(filename, 'r') as file:
                    for row in file:
                        for value in row.strip().split(','):
                            try:
                                num = float(value)
                                values.append(num * scale)
                            except ValueError:
                                print(f"unconvert:: {value}")
            except IOError as e:
                print(f"error: {e}")
            values = np.repeat(values, int(slots / len(values)))
            blocks = values.reshape(32, 1024)
            half = 16
            order = np.ravel(np.column_stack([np.arange(half), np.arange(half, 32)]))
            values = blocks[order].reshape(-1)
            values = values * mask
            name = f"{val_name}{i}{j}{k}"
            encoded = fhe.encode(values, name, level, slots, False, cryptoContext)
            return encoded
    if val_name == "layer7-conv1bn1-n1" or val_name == "layer7-conv1bn1-n2":
        if cryptoContext.DIRECT_LOAD:
            full_name = "{}_{}_{}".format(val_name, level, slots)
            if cryptoContext.pre_encode_type == "middle":
                name = f"{val_name}{i}{j}{k}"
            else:
                name = full_name
            return fhe.encode(cryptoContext.pre_encoded[name], name, level, slots, False, cryptoContext)
        else:
            values = []
            filename = cryptoContext.weight_path + val_name + '.bin'
            if not os.path.isfile(filename):
                print(f"Failed to open file: {filename}")
                return values
            try:
                with open(filename, 'r') as file:
                    for row in file:
                        for value in row.strip().split(','):
                            try:
                                num = float(value)
                                values.append(num * scale)
                            except ValueError:
                                print(f"unconvert:: {value}")
            except IOError as e:
                print(f"error: {e}")
            values = np.repeat(values, int(slots / len(values)))
            blocks = values.reshape(64, 512)
            half = 32
            order = np.ravel(np.column_stack([np.arange(half), np.arange(half, 64)]))
            values = blocks[order].reshape(-1)
            values = values * mask
            name = f"{val_name}{i}{j}{k}"
            encoded = fhe.encode(values, name, level, slots, False, cryptoContext)
            return encoded
    else:
        if cryptoContext.DIRECT_LOAD:
            if cryptoContext.pre_encode_type == "middle":
                name = f"{val_name}{i}{j}{k}"
            return fhe.encode(cryptoContext.pre_encoded[name], name, level, slots, False, cryptoContext)
        else:
            values = []
            filename = cryptoContext.weight_path + val_name + '.bin'
            if not os.path.isfile(filename):
                print(f"Failed to open file: {filename}")
                return values
            try:
                with open(filename, 'r') as file:
                    for row in file:
                        for value in row.strip().split(','):
                            try:
                                num = float(value)
                                values.append(num * scale)
                            except ValueError:
                                print(f"unconvert:: {value}")
            except IOError as e:
                print(f"error: {e}")
            values = np.repeat(values, int(slots / len(values)))
            values = values * mask
            name = f"{val_name}{i}{j}{k}"
            encoded = fhe.encode(values, name, level, slots, False, cryptoContext)
            return encoded
