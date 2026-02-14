import sys, os
from third_party.flatbuffers.python.flatbuffers.packer import float32
sys.path.append("/".join(os.getcwd().split("/")[:-2]))
sys.path.append("/".join(os.getcwd().split("/")[:-3]))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["DATA_DIR"] = os.path.join(project_root, "ICML_2026_PackCNN", "data")
# os.environ["DATA_DIR"] = "/data/test/data"
import pickle
import time
import math
import numpy as np
import csv
from examples.utils import approx
import torch
import torch.fhe as fhe
from examples.resnet.gen_weights.HerPN import  get_Aespa_MutalChannel_PAF_resnet20
block_num1=3


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
    assert len(ct_list_shape) ==2 or len(ct_list_shape) ==1, "only support two/one dim input ct_list now"
    dump_ct = []
    if len(ct_list_shape) ==2:
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
    if (len(ct_list_shape)>1):
        print(f"len(ct_list[0]) {len(ct_list[0])}")

    for row in ct_list:
        if len(ct_list_shape)==1:
            row = [row]
        for ct in row:
            ct.cv = [torch.from_numpy(ct.cv[0]).to(device), torch.from_numpy(ct.cv[1]).to(device)]

    return ct_list, ct_list_shape


def reshape_ct_list(ct_list, ct_list_shape):

    if len(ct_list_shape) == 1:
        return ct_list

    if ct_list_shape[0]==1:
        reshaped_ct_list = [ct_list]
    elif ct_list_shape[0]==2:
        half_len_ct_list = len(ct_list)//2
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
            cipher_list, dim =load_cts_nd(file_name, "cuda")
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
def encrypt_data(mode, input_data, dir_path, slots, openfhe_context,cryptoContext):
    cipher_list = None
    if mode in ["run", "save", "verify"]:
        print("reading input data...")
        cipher_list = np.empty((input_data.shape[0], input_data.shape[1], input_data.shape[2]), dtype=object)
        print("begin encrypting data...")
        begin_time = time.time()
        for t in range(input_data.shape[0]):
            for i in range(input_data.shape[1]):
                for j in range(input_data.shape[2]):
                    cipher_list[t][i][j] = openfhe_context.encrypt(input_data[t][i][j], "cuda", 1, cryptoContext.L-20, slots)
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
def get_weight_bias(layer, state_dict, index,cryptoContext, eps=1e-5):
    model = get_Aespa_MutalChannel_PAF_resnet20()
    if layer == "":
        W=model.conv1.weight
    elif layer=="downsample0":
        W=model.layer2[0].downsample[0].weight
    elif layer=="downsample1":
        W=model.layer3[0].downsample[0].weight
    else:
        layer_name, idx_str = layer.split("[")
        idx = int(idx_str.rstrip("]"))
        block = getattr(model, layer_name)[idx]
        W = getattr(block, f"conv{index}").weight
    Weight1 = np.empty((W.shape[0], W.shape[1], W.shape[2], W.shape[3]))
    Bias1 = np.empty((W.shape[0],))
    if layer == "":
        filename = cryptoContext.weight_path + "conv1bn1-A2" + '.bin'
        values=[]
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
            temp=int(len(values)/W.shape[0])
            A=[0]*W.shape[0]
            for i in range(W.shape[0]):
                A[i]=values[i*temp]
        except IOError as e:
            print(f"error: {e}")
    elif layer=="downsample0":
        conv_weight = state_dict['layer2.0.downsample.0.weight']  
        bn_weight = state_dict['layer2.0.downsample.1.weight']  
        bn_bias = state_dict['layer2.0.downsample.1.bias']  
        bn_running_mean = state_dict['layer2.0.downsample.1.running_mean']  
        bn_running_var = state_dict['layer2.0.downsample.1.running_var']  
        A=bn_weight/torch.sqrt(bn_running_var + eps)
        b=-(bn_weight*bn_running_mean/torch.sqrt(bn_running_var)+eps)+bn_bias
        A = model.layer2[0].downsample[1].weight / torch.sqrt(
            model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)
        b = -(model.layer2[0].downsample[1].weight * model.layer2[0].downsample[1].running_mean / torch.sqrt(
            model.layer2[0].downsample[1].running_var + model.layer2[0].downsample[1].eps)) + \
            model.layer2[0].downsample[1].bias
        A1 = model.layer2[0].HerPN2.a2.detach() ** 0.5
        A = A * A1
        b = b * A1
    elif layer=="downsample1":
        conv_weight = state_dict['layer3.0.downsample.0.weight']  
        bn_weight = state_dict['layer3.0.downsample.1.weight']  
        bn_bias = state_dict['layer3.0.downsample.1.bias']  
        bn_running_mean = state_dict['layer3.0.downsample.1.running_mean']  
        bn_running_var = state_dict['layer3.0.downsample.1.running_var']  
        A=bn_weight/torch.sqrt(bn_running_var + eps)
        b=-(bn_weight*bn_running_mean/torch.sqrt(bn_running_var+eps))+bn_bias
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
    if layer=="downsample0":
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for p in range(1):
                    for q in range(1):
                        Weight1[i, j, p, q] = W[i, j].reshape(1)[q + p * 3].detach() * A[i].detach()
        for i in range(W.shape[0]):
            Bias1[i] = b[i].item()
        return Weight1,Bias1
    elif layer=="downsample1":
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for p in range(1):
                    for q in range(1):
                        Weight1[i, j, p, q] = W[i, j].reshape(1)[q + p * 3].detach() * A[i].detach()
        for i in range(W.shape[0]):
            Bias1[i] = b[i].item()
        return Weight1,Bias1
    else:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                for p in range(3):
                    for q in range(3):
                        Weight1[i, j, p, q] = W[i, j].reshape(9)[q+p*3].detach() * A[i]
        return Weight1
def read_image(batch_size):
    filePath = os.path.join(project_root, "ICML_2026_PackCNN", "data", "test_batch.bin")
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
def judge(group_num, now_group, index):
    height = 33
    width = 33
    if index == 0:
        return True
    else:
        temp = np.floor(((index - 1) * (3 * group_num) + now_group * 3) / 33)
        if np.floor(((index) * (3 * group_num) + now_group * 3) / 33) == temp:
            return True
        else:
            return False
def batch_input(mode1, batch_size, openfhe_context,cryptoContext):
    slots = 2 ** 15
    DATA_DIR = os.environ["DATA_DIR"]
    dir_path = DATA_DIR
    # dir_path =os.path.join(project_root, "ICML_2026_PackCNN", "data")

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
    input_encrypt = encrypt_data(mode1, input, dir_path, slots, openfhe_context,cryptoContext)
    return input_encrypt, batch_label
def perfect_square_split(n):
    if n <= 0:
        raise ValueError("n must be integater")
    root = int(math.sqrt(n))
    if root * root == n:
        return root, root
    for i in range(root, 0, -1):
        if n % i == 0:
            return  n // i,i
def pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext):
    N = 65536
    slots = int(N / 2)
    repeat = int(output_channel / in_channel)  
    num_in_cipher = int(slots / (output_channel * batch_size))
    if layer=="downsample0"or layer=="downsample1":
        weight101, bias101 = get_weight_bias(layer, state_dict, index,cryptoContext)
    else:
        weight101 = get_weight_bias(layer, state_dict, index,cryptoContext)
    if layer=="layer2[0]" and index==2:
        weight_temp=weight101.copy()
        half = output_channel // 2
        new_order = np.stack((np.arange(half), np.arange(half, output_channel)), axis=1).ravel()
        a_rearranged = weight_temp[:, new_order, :, :]
        weight101=a_rearranged
    if layer=="layer3[0]" and index==2:
        weight_temp=weight101.copy()
        half = output_channel // 2
        new_order = np.stack((np.arange(half), np.arange(half, output_channel)), axis=1).ravel()
        a_rearranged = weight_temp[:, new_order, :, :]
        weight101=a_rearranged
    weight101 = weight101 * scale
    if layer=="":
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    if layer=="downsample0":
        b = np.zeros((32, 32, 3, 3), dtype=arr_expanded101.dtype)
        b[::2, 0::2, :, :] = arr_expanded101[::2, :, :, :]
        b[1::2, 1::2, :, :] = arr_expanded101[1::2, :, :, :]
        arr_expanded101=b
    if layer=="downsample1":
        b = np.zeros((64, 64, 3, 3), dtype=arr_expanded101.dtype)
        b[::2, 0::2, :, :] = arr_expanded101[::2, :, :, :]
        b[1::2, 1::2, :, :] = arr_expanded101[1::2, :, :, :]
        arr_expanded101=b
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))  
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer=="downsample0":
        giant,baby = perfect_square_split(in_channel)  
    elif layer=="downsample1":
        giant,baby = perfect_square_split(in_channel)  
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
    if layer=="downsample0"or layer=="downsample1":
        for i in range(group_num):
            num = int(np.floor(block_num / group_num))
            if i < block_num % group_num:
                num += 1
            for j in range(output_channel):
                pre_bias_list1[i][j * temp + num * batch_size:(j + 1) * temp] = 0
        return pre_weight_list, pre_bias_list1
    return pre_weight_list
def pre_weight2(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext):
    N = 65536
    slots = int(N / 2)
    repeat = int(output_channel / in_channel)  
    num_in_cipher = int(slots / (output_channel * batch_size))
    if layer=="downsample0"or layer=="downsample1":
        weight101, bias101 = get_weight_bias(layer, state_dict, index,cryptoContext)
    else:
        weight101= get_weight_bias(layer, state_dict, index,cryptoContext)
    weight101 = weight101 * scale
    if layer=="":
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 1), (0, 0), (0, 0)), mode='constant', constant_values=0)
    else:
        arr_expanded101 = np.pad(weight101, ((0, 0), (0, 0), (0, 0), (0, 0)), mode='constant', constant_values=0)
    if layer=="layer2[0]":
        arr_expanded1=arr_expanded101[:16]
        arr_expanded2=arr_expanded101[16:]   
    if layer=="layer3[0]":
        arr_expanded1=arr_expanded101[:32]
        arr_expanded2=arr_expanded101[32:]   
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))  
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    baby, giant = perfect_square_split(in_channel)  
    repeat_bias = int(slots / output_channel)
    pre_weight_list = np.empty((wi, wo, baby, giant, slots))
    pre_weight_list2 = np.empty((wi, wo, baby, giant, slots))
    if layer=="downsample0" or layer=="downsample1":
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
    return pre_weight_list,pre_weight_list2
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
                     wi, wo,layer):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)  
    if layer=="downsample0":
        giant,baby = perfect_square_split(in_channel)  
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
def min_padding_to_next_multiple_of_k(n, k):
    r = n % k
    if r == 0:
        return 0
    else:
        return k - r
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
    edge_list = np.empty((int(height_pad / wi)+1, 2))
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
    temp=[list(set(group)) for group in down_need]
    down_need_final=[[] for _ in range(group_num)]
    for i in range(group_num):
        if 0 in temp[i]:
            down_need_final[i].extend([0,2,6,8])
        if 1 in temp[i]:
            down_need_final[i].extend([1,7])
        if 2 in temp[i]:
            down_need_final[i].extend([3,5])
        if 3 in temp[i]:
            down_need_final[i].append(4)
    result = []
    for sublist in down_need_final:
        row_col_indices = [(val // 3, val % 3) for val in sublist]  
        result.append(row_col_indices)
    return result
@fhe.utils.profile_python_function
def conv_downsampling(layer,input,weight,bias,pre_bias_list2,pre_bias_list3,pre_bias_list4,cryptoContext,openfhe_context,height,width,wi,wo,batch_size,in_channel,output_channel):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    repeat = int(output_channel / in_channel)  
    giant,baby = perfect_square_split(in_channel)  
    temp = batch_size * num_in_cipher
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    output = np.empty((group_num,wi,wo), dtype=object)
    input_rotate_list = np.zeros((group_num, wi, wo, giant), dtype=object)
    for i in range(group_num):
        for q in range(3):
            for r in range(3):
                for b in range(giant//2):
                    if b == 0:
                        input_rotate_list[i][q][r][b] = input[i][q][r]
                    else:
                        input_rotate_list[i][q][r][b] = fhe.homo_rotate(input[i][q][r], 2*b * temp, cryptoContext)
    def conv2(o, p, q, r, input, weight, cryptoContext):
        weight_encode = np.empty((baby, giant), dtype=object)
        output_giant = np.empty(baby, dtype=object)
        if cryptoContext.config.SAVE_MIDDLE == False:
            for i in range(baby):
                for j in range(giant//2):
                    name = f"downsample_{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(cryptoContext.pre_encoded[name], name,
                                                     cryptoContext.L - input[0].cur_limbs, slots, False, cryptoContext)
        else:
            for i in range(baby):
                for j in range(giant//2):
                    name = f"downsample_{layer}_{o}_{p}_{q}_{r}_{i}_{j}"
                    weight_encode[i][j] = fhe.encode(weight[i][2*j], name, cryptoContext.L - input[0].cur_limbs, slots,
                                                     False, cryptoContext)
        for g in range(giant//2):
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
                output[i][j][k]=conv2(0,0,0,0,input_rotate_list[i][j][k],weight[0][0],cryptoContext)
    for i in range(group_num):
        for j in range(wi):
            for k in range(wo):
                output[i][j][k]=fhe.homo_rescale(output[i][j][k],1,cryptoContext)
    if cryptoContext.config.SAVE_MIDDLE ==True:
        for i in range(group_num):
            encode_temp = fhe.encode(bias[i], f"{layer}_{i}_{0}", cryptoContext.L - output[i][0][0].cur_limbs, slots, False,
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
            temp=np.ones(slots)
            A2 = read_values_from_file(0,0,0,f"layer{(a - 1) * block_num1 + int(np.floor((b + 1) / 2))}-conv{2}bn{2}-A2",
                                       cryptoContext.L - res_initial[0][0][0].cur_limbs,
                                       2 ** 15,
                                       cryptoContext, temp,1)
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
                            temp_list[0] = conv2(i , j, k, 0,
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
                                                                       0)][max(edge_list[i][0][0], 0)][0][0]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][1]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2,
                                                 input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][max(edge_list[i][0][0], 0)][0][2]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][2],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][0]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                                 cryptoContext)
                            temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                                 cryptoContext)
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][2]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][2],
                                                 cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][0][0], pre_weight_list_edge[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][0][1],pre_weight_list_edge[i][1][1],
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
                                                               pre_weight_list3[edge_list[i][3][1]][0][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][1][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3[edge_list[i][3][1]][2][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][0][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][1][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3[edge_list[i][3][1]][2][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][1], pre_weight_list_edge[i][0][0]* pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][2], pre_weight_list_edge[i][0][1]* pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][1], pre_weight_list_edge[i][1][0]* pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][2], pre_weight_list_edge[i][1][1]* pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][1], pre_weight_list_edge[i][2][0]* pre_weight_list4[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][2], pre_weight_list_edge[i][2][1]* pre_weight_list4[i][2][1],
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
                                                                       0)][max(edge_list[i][2][0], 0)][0][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3,
                                                 input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][1][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6,
                                                 input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                       0)][
                                                                   max(edge_list[i][2][0], 0)][2][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][2][0],
                                                               edge_list[i][2][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][0][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                                 cryptoContext)
                            temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][1][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                                 cryptoContext)
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                     max(edge_list[i][2][0], 0)][2][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][2][0],
                                                 cryptoContext)
                        temp_list[1] = conv2(int(i != 0), j, k, 1, input_rotate_list[i][0][0], pre_weight_list[0][0][0][1]* pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(int(i != 0), j, k, 2, input_rotate_list[i][0][1], pre_weight_list[0][0][0][2]* pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[4] = conv2(int(i != 0), j, k, 4, input_rotate_list[i][1][0], pre_weight_list[0][0][1][1]* pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(int(i != 0), j, k, 5, input_rotate_list[i][1][1], pre_weight_list[0][0][1][2]* pre_weight_list4[i][1][2],
                                             cryptoContext)
                        temp_list[7] = conv2(int(i != 0), j, k, 7, input_rotate_list[i][2][0], pre_weight_list[0][0][2][1]* pre_weight_list4[i][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(int(i != 0), j, k, 8, input_rotate_list[i][2][1], pre_weight_list[0][0][2][2]* pre_weight_list4[i][2][2],
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
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][0], pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][1], pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][0][2], pre_weight_list_edge[i][0][2] * pre_weight_list4[i][0][2],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][0], pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][1], pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][1][2], pre_weight_list_edge[i][1][2] * pre_weight_list4[i][1][2],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][0], pre_weight_list_edge[i][2][0] * pre_weight_list4[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][1], pre_weight_list_edge[i][2][1] * pre_weight_list4[i][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[i][2][2], pre_weight_list_edge[i][2][2] * pre_weight_list4[i][2][2],
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
                                                               pre_weight_list5[edge_list[i][1][1]][2][0]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][0],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][1] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][1]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][1],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8,
                                                 input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_list[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][2] *
                                                               pre_weight_list5[edge_list[i][1][1]][2][2]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][2],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                        else:
                            temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][0] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][0]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][0],
                                                 cryptoContext)
                            temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][1]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                            temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                                 pre_weight_list[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][2] *
                                                 pre_weight_list5[edge_list[i][1][1]][2][2]*
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
        if (a - 1) * block_num1  + int(np.floor((b + 1) / 2))==block_num1+1 or(a - 1) * block_num1  + int(np.floor((b + 1) / 2))==block_num1*2+1:
            for i in range(group_num):
                for j in range(wi):
                    for k in range(wo):
                        output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k], cryptoContext)
        else:
            A2 = read_values_from_file(1,1,1,f"layer{(a - 1) * block_num1 + int(np.floor((b + 1) / 2))}-conv{2}bn{2}-A2",
                                   cryptoContext.L - res_initial[0][0][0].cur_limbs,
                                   2 ** 15,
                                   cryptoContext, np.ones(slots),1)
            for i in range(group_num):
                for j in range(wi):
                    for k in range(wo):
                        res_initial[i][j][k] = fhe.homo_mul_pt(res_initial[i][j][k], A2, cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k], cryptoContext)
                        output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
    return output_cipher
@fhe.utils.profile_python_function
def conv2_batch_extend(input, zero_num_temp, pre_weight_list, pre_weight_list_edge, edge_list, pre_weight_list2,
                pre_weight_list3, pre_weight_list4, pre_weight_list5, pre_bias_list, pre_bias_list2, pre_bias_list3,
                pre_bias_list4, batch_size, in_channel, output_channel, height, width, wi, wo, cryptoContext,
                openfhe_context, isresnet, res_initial, layer,down_need,pre_weight_lista, pre_weight_list_edgea, pre_weight_list2a, pre_weight_list3a, pre_weight_list4a, pre_weight_list5a, pre_bias_lista, pre_bias_list2a, pre_bias_list3a, pre_bias_list4a
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
    output_cipher = np.empty((group_num*repeat, wi, wo), dtype=object)
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
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][0] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1] *
                                                           pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1] *
                                             pre_weight_list2a[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                    if edge_list[i][5][0] != 0:
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                   0)][max(edge_list[i][5][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][5][1]][0][2],
                                                           edge_list[i][5][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                 max(edge_list[i][5][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][5][1]][0][2],
                                             cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][0][1], pre_weight_list_edgea[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][0][2], pre_weight_list_edgea[i][1][1],
                                         cryptoContext)
                    if edge_list[i][3][0] != 0:
                        temp_list[5] = conv2(i+group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[i][1][1], pre_weight_list_edgea[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[i][1][2], pre_weight_list_edgea[i][2][1],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][4][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                   0)][
                                                               max(edge_list[i][4][0], 0)][0][0],
                                                           edge_list[i][4][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                 max(edge_list[i][4][0], 0)][0][0],
                                             cryptoContext)
                    if edge_list[i][0][0] != 0:
                        temp_list[1] = conv2(i+group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][
                                                               max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[3] = conv2(i+group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i+group_num, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][0][1], pre_weight_list_edgea[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][0][2], pre_weight_list_edgea[i][0][1],
                                         cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][1][1], pre_weight_list_edgea[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][1][2], pre_weight_list_edgea[i][1][1],
                                         cryptoContext)
                    temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[i][2][1], pre_weight_list_edgea[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[i][2][2], pre_weight_list_edgea[i][2][1],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][max(edge_list[i][2][0], 0)][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                if k == wi - 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][3][0] != 0:
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][
                                                               max(edge_list[i][3][0], 0)][0][2] *
                                                           pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                                           pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i+group_num, j, k, 5,
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
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                             pre_weight_list4a[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                    if edge_list[i][7][0] != 0:
                        temp_list[8] = conv2(i+group_num, j, k, 8,
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
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                 max(edge_list[i][7][0], 0)][2][2] *
                                             pre_weight_list3a[edge_list[i][7][1]][2][2] *
                                             pre_weight_list5a[edge_list[i][7][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7,
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
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list2a[edge_list[i][1][1]][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                    temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][1][2],
                                         pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][1][1],
                                         pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][2][1],
                                         pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[7] = conv2(i+group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    if edge_list[i][6][0] != 0:
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                   0)][
                                                               max(edge_list[i][6][0], 0)][2][0] *
                                                           pre_weight_list5a[edge_list[i][6][1]][2][0],
                                                           edge_list[i][6][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                 max(edge_list[i][6][0], 0)][2][0] *
                                             pre_weight_list5a[edge_list[i][6][1]][2][0],
                                             cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][0][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0] *
                                                           pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0] *
                                             pre_weight_list4a[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                    temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][1][0],
                                         pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[i][1][1],
                                         pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][2][0],
                                         pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[i][2][1],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                  fhe.homo_rotate(output_giant_[g],
                                                                                  -giant * g * temp,
                                                                                  cryptoContext),
                                                                  cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                             input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][0] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_lista[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_lista[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5a[edge_list[i][1][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][1][0],
                                         pre_weight_lista[0][0][0][0] * pre_weight_list4a[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][1][1],
                                         pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[i][1][2],
                                         pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][2][0],
                                         pre_weight_lista[0][0][1][0] * pre_weight_list4a[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][2][1],
                                         pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[i][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
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
                output_cipher[i+group_num][j][k] = fhe.homo_rescale(output_cipher[i+group_num][j][k], 1, cryptoContext)
    else:
        for i in range(group_num):
            for j in range(wi):
                for k in range(wo):
                    output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                    output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k], cryptoContext)
    return output_cipher
@fhe.utils.profile_python_function
def conv2_batch_extend2(input, zero_num_temp, pre_weight_list, pre_weight_list_edge, edge_list, pre_weight_list2,
                pre_weight_list3, pre_weight_list4, pre_weight_list5, pre_bias_list, pre_bias_list2, pre_bias_list3,
                pre_bias_list4, batch_size, in_channel, output_channel, height, width, wi, wo, cryptoContext,
                openfhe_context, isresnet, res_initial, layer,down_need,pre_weight_lista, pre_weight_list_edgea, pre_weight_list2a, pre_weight_list3a, pre_weight_list4a, pre_weight_list5a, pre_bias_lista, pre_bias_list2a, pre_bias_list3a, pre_bias_list4a
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
    output_cipher = np.empty((group_num*repeat, wi, wo), dtype=object)
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
                                                                   0)][max(edge_list[i][0][0], 0)][0][0]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1,
                                             input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][1]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2,
                                             input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                   0)][max(edge_list[i][0][0], 0)][0][2]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][2],
                                                           edge_list[i][0][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][0]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][1]*
                                                               pre_weight_list2[edge_list[i][0][1]][0][1],
                                             cryptoContext)
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                 max(edge_list[i][0][0], 0)][0][2]*
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
                                                           pre_weight_list3[edge_list[i][3][1]][0][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5,
                                             input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][1][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][1][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                   0)][max(edge_list[i][3][0], 0)][2][2] *
                                                           pre_weight_list3[edge_list[i][3][1]][2][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][2][2],
                                                           edge_list[i][3][0]),
                                             cryptoContext)
                    else:
                        temp_list[2] = conv2(i, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][0][2] *
                                             pre_weight_list3[edge_list[i][3][1]][0][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][0][2],
                                             cryptoContext)
                        temp_list[5] = conv2(i, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][1][2] *
                                             pre_weight_list3[edge_list[i][3][1]][1][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][1][2],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                 max(edge_list[i][3][0], 0)][2][2] *
                                             pre_weight_list3[edge_list[i][3][1]][2][2]*
                                                 pre_weight_list4[edge_list[i][3][1]][2][2],
                                             cryptoContext)
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][1], pre_weight_list_edge[i][0][0]* pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][2], pre_weight_list_edge[i][0][1]* pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][1], pre_weight_list_edge[i][1][0]* pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][2], pre_weight_list_edge[i][1][1]* pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][1], pre_weight_list_edge[i][2][0]* pre_weight_list4[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][2], pre_weight_list_edge[i][2][1]* pre_weight_list4[i][2][1],
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
                                                                   0)][max(edge_list[i][2][0], 0)][0][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3,
                                             input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][1][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6,
                                             input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                   0)][
                                                               max(edge_list[i][2][0], 0)][2][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][2][0],
                                                           edge_list[i][2][0]),
                                             cryptoContext)
                    else:
                        temp_list[0] = conv2(i, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][0][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][1][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][1][0],
                                             cryptoContext)
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                 max(edge_list[i][2][0], 0)][2][0]*
                                                               pre_weight_list4[edge_list[i][2][1]][2][0],
                                             cryptoContext)
                    temp_list[1] = conv2(int(i != 0), j, k, 1, input_rotate_list[i][0][0], pre_weight_list[0][0][0][1]* pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(int(i != 0), j, k, 2, input_rotate_list[i][0][1], pre_weight_list[0][0][0][2]* pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[4] = conv2(int(i != 0), j, k, 4, input_rotate_list[i][1][0], pre_weight_list[0][0][1][1]* pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(int(i != 0), j, k, 5, input_rotate_list[i][1][1], pre_weight_list[0][0][1][2]* pre_weight_list4[i][1][2],
                                         cryptoContext)
                    temp_list[7] = conv2(int(i != 0), j, k, 7, input_rotate_list[i][2][0], pre_weight_list[0][0][2][1]* pre_weight_list4[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(int(i != 0), j, k, 8, input_rotate_list[i][2][1], pre_weight_list[0][0][2][2]* pre_weight_list4[i][2][2],
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
                    temp_list[0] = conv2(i, j, k, 0, input_rotate_list[i][0][0], pre_weight_list_edge[i][0][0] * pre_weight_list4[i][0][0],
                                         cryptoContext)
                    temp_list[1] = conv2(i, j, k, 1, input_rotate_list[i][0][1], pre_weight_list_edge[i][0][1] * pre_weight_list4[i][0][1],
                                         cryptoContext)
                    temp_list[2] = conv2(i, j, k, 2, input_rotate_list[i][0][2], pre_weight_list_edge[i][0][2] * pre_weight_list4[i][0][2],
                                         cryptoContext)
                    temp_list[3] = conv2(i, j, k, 3, input_rotate_list[i][1][0], pre_weight_list_edge[i][1][0] * pre_weight_list4[i][1][0],
                                         cryptoContext)
                    temp_list[4] = conv2(i, j, k, 4, input_rotate_list[i][1][1], pre_weight_list_edge[i][1][1] * pre_weight_list4[i][1][1],
                                         cryptoContext)
                    temp_list[5] = conv2(i, j, k, 5, input_rotate_list[i][1][2], pre_weight_list_edge[i][1][2] * pre_weight_list4[i][1][2],
                                         cryptoContext)
                    temp_list[6] = conv2(i, j, k, 6, input_rotate_list[i][2][0], pre_weight_list_edge[i][2][0] * pre_weight_list4[i][2][0],
                                         cryptoContext)
                    temp_list[7] = conv2(i, j, k, 7, input_rotate_list[i][2][1], pre_weight_list_edge[i][2][1] * pre_weight_list4[i][2][1],
                                         cryptoContext)
                    temp_list[8] = conv2(i, j, k, 8, input_rotate_list[i][2][2], pre_weight_list_edge[i][2][2] * pre_weight_list4[i][2][2],
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
                                                           pre_weight_list5[edge_list[i][1][1]][2][0]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][0],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7,
                                             input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][1] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][1]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][1],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8,
                                             input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                             rotate_weight(pre_weight_list[
                                                               max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                   0)][
                                                               max(edge_list[i][1][0], 0)][2][2] *
                                                           pre_weight_list5[edge_list[i][1][1]][2][2]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][2],
                                                           edge_list[i][1][0]),
                                             cryptoContext)
                    else:
                        temp_list[6] = conv2(i, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][0] *
                                             pre_weight_list5[edge_list[i][1][1]][2][0]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][1] *
                                             pre_weight_list5[edge_list[i][1][1]][2][1]*
                                                 pre_weight_list2[edge_list[i][1][1]][2][1],
                                             cryptoContext)
                        temp_list[8] = conv2(i, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                             pre_weight_list[
                                                 max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                 max(edge_list[i][1][0], 0)][2][2] *
                                             pre_weight_list5[edge_list[i][1][1]][2][2]*
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
                            temp_list[0] = conv2(i+group_num, j, k, 0,
                                                 input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][0] *
                                                               pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                            temp_list[1] = conv2(i+group_num, j, k, 1,
                                                 input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                       0)][
                                                                   max(edge_list[i][0][0], 0)][0][1] *
                                                               pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                               edge_list[i][0][0]),
                                                 cryptoContext)
                        else:
                            temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][1],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][0] *
                                                 pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                 cryptoContext)
                            temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][2],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                     max(edge_list[i][0][0], 0)][0][1] *
                                                 pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                 cryptoContext)
                        if edge_list[i][5][0] != 0:
                            temp_list[2] = conv2(i+group_num, j, k, 2,
                                                 input_rotate_list2[edge_list[i][5][1]][2][0][edge_list[i][5][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][5][0]),
                                                                       0)][max(edge_list[i][5][0], 0)][0][2] *
                                                               pre_weight_list3a[edge_list[i][5][1]][0][2],
                                                               edge_list[i][5][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][5][1]][2][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][5][0]), 0)][
                                                     max(edge_list[i][5][0], 0)][0][2] *
                                                 pre_weight_list3a[edge_list[i][5][1]][0][2],
                                                 cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][0][1], pre_weight_list_edgea[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][0][2], pre_weight_list_edgea[i][1][1],
                                             cryptoContext)
                        if edge_list[i][3][0] != 0:
                            temp_list[5] = conv2(i+group_num, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i+group_num, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[i][1][1], pre_weight_list_edgea[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[i][1][2], pre_weight_list_edgea[i][2][1],
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
                                output_cipher[i+group_num, j, k] = output_giant_[g]
                            else:
                                
                                
                                output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][4][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                            input_rotate_list2[edge_list[i][4][1]][2][2][edge_list[i][4][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][4][0]),
                                                                0)][
                                                            max(edge_list[i][4][0], 0)][0][0],
                                                        edge_list[i][4][0]),
                                            cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][4][1]][2][2],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][4][0]), 0)][
                                                max(edge_list[i][4][0], 0)][0][0],
                                            cryptoContext)
                    if edge_list[i][0][0] != 0:
                        temp_list[1] = conv2(i+group_num, j, k, 1,
                                            input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                0)][
                                                            max(edge_list[i][0][0], 0)][0][1],
                                                        edge_list[i][0][0]),
                                            cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                            input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                0)][
                                                            max(edge_list[i][0][0], 0)][0][2],
                                                        edge_list[i][0][0]),
                                            cryptoContext)
                    else:
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][0],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                max(edge_list[i][0][0], 0)][0][1],
                                            cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][1],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                max(edge_list[i][0][0], 0)][0][2],
                                            cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[3] = conv2(i+group_num, j, k, 3,
                                            input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                0)][
                                                            max(edge_list[i][2][0], 0)][1][0],
                                                        edge_list[i][2][0]),
                                            cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                            input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                0)][
                                                            max(edge_list[i][2][0], 0)][2][0],
                                                        edge_list[i][2][0]),
                                            cryptoContext)
                    else:
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][0][2],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                max(edge_list[i][2][0], 0)][1][0],
                                            cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][1][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                fhe.homo_rotate(output_giant_[g],
                                                                                -giant * g * temp,
                                                                                cryptoContext),
                                                                cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][0][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                            input_rotate_list2[edge_list[i][0][1]][2][0][edge_list[i][0][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                0)][max(edge_list[i][0][0], 0)][0][0]*
                                                        pre_weight_list2a[edge_list[i][0][1]][0][0],
                                                        edge_list[i][0][0]),
                                            cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1,
                                            input_rotate_list2[edge_list[i][0][1]][2][1][edge_list[i][0][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                0)][max(edge_list[i][0][0], 0)][0][1]*
                                                        pre_weight_list2a[edge_list[i][0][1]][0][1],
                                                        edge_list[i][0][0]),
                                            cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2,
                                            input_rotate_list2[edge_list[i][0][1]][2][2][edge_list[i][0][0]],
                                            rotate_weight(pre_weight_lista[
                                                            max(int(zero_num_temp[i] - edge_list[i][0][0]),
                                                                0)][max(edge_list[i][0][0], 0)][0][2]*
                                                        pre_weight_list2a[edge_list[i][0][1]][0][2],
                                                        edge_list[i][0][0]),
                                            cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][0][1]][2][0],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                max(edge_list[i][0][0], 0)][0][0]*
                                                        pre_weight_list2a[edge_list[i][0][1]][0][0],
                                            cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[edge_list[i][0][1]][2][1],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                max(edge_list[i][0][0], 0)][0][1]*
                                                        pre_weight_list2a[edge_list[i][0][1]][0][1],
                                            cryptoContext)
                        temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][0][1]][2][2],
                                            pre_weight_lista[
                                                max(int(zero_num_temp[i] - edge_list[i][0][0]), 0)][
                                                max(edge_list[i][0][0], 0)][0][2]*
                                                        pre_weight_list2a[edge_list[i][0][1]][0][2],
                                            cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][0][0], pre_weight_list_edgea[i][1][0],
                                        cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][0][1],pre_weight_list_edgea[i][1][1],
                                        cryptoContext)
                    temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[i][0][2], pre_weight_list_edgea[i][1][2],
                                        cryptoContext)
                    temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[i][1][0], pre_weight_list_edgea[i][2][0],
                                        cryptoContext)
                    temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[i][1][1], pre_weight_list_edgea[i][2][1],
                                        cryptoContext)
                    temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[i][1][2], pre_weight_list_edgea[i][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                fhe.homo_rotate(output_giant_[g],
                                                                                -giant * g * temp,
                                                                                cryptoContext),
                                                                cryptoContext)
            if j == (((wi + 1) / 2) - 1):
                if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][3][0] != 0:
                            temp_list[2] = conv2(i+group_num, j, k, 2,
                                                 input_rotate_list2[edge_list[i][3][1]][0][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][0][2] *
                                                               pre_weight_list3a[edge_list[i][3][1]][0][2]*
                                                 pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i+group_num, j, k, 5,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][1][2] *
                                                               pre_weight_list3a[edge_list[i][3][1]][1][2]*
                                                 pre_weight_list4a[edge_list[i][3][1]][1][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[8] = conv2(i+group_num, j, k, 8,
                                                 input_rotate_list2[edge_list[i][3][1]][2][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][max(edge_list[i][3][0], 0)][2][2] *
                                                               pre_weight_list3a[edge_list[i][3][1]][2][2]*
                                                 pre_weight_list4a[edge_list[i][3][1]][2][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                        else:
                            temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][0][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][0][2]*
                                                 pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][1][2]*
                                                 pre_weight_list4a[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                            temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][2][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][2][2]*
                                                 pre_weight_list4a[edge_list[i][3][1]][2][2],
                                                 cryptoContext)
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][0][1], pre_weight_list_edgea[i][0][0]* pre_weight_list4a[i][0][0],
                                             cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][0][2], pre_weight_list_edgea[i][0][1]* pre_weight_list4a[i][0][1],
                                             cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][1][1], pre_weight_list_edgea[i][1][0]* pre_weight_list4a[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][1][2], pre_weight_list_edgea[i][1][1]* pre_weight_list4a[i][1][1],
                                             cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[i][2][1], pre_weight_list_edgea[i][2][0]* pre_weight_list4a[i][2][0],
                                             cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[i][2][2], pre_weight_list_edgea[i][2][1]* pre_weight_list4a[i][2][1],
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
                                output_cipher[i+group_num, j, k] = output_giant_[g]
                            else:
                                
                                
                                output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                                input_rotate_list2[edge_list[i][2][1]][0][2][edge_list[i][2][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                    0)][max(edge_list[i][2][0], 0)][0][0]*
                                                            pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                            edge_list[i][2][0]),
                                                cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3,
                                                input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                    0)][
                                                                max(edge_list[i][2][0], 0)][1][0]*
                                                            pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                            edge_list[i][2][0]),
                                                cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                                input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                    0)][
                                                                max(edge_list[i][2][0], 0)][2][0]*
                                                            pre_weight_list4a[edge_list[i][2][1]][2][0],
                                                            edge_list[i][2][0]),
                                                cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][0][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                    max(edge_list[i][2][0], 0)][0][0]*
                                                            pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][1][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                    max(edge_list[i][2][0], 0)][1][0]*
                                                            pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                cryptoContext)
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][2][1]][2][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                    max(edge_list[i][2][0], 0)][2][0]*
                                                            pre_weight_list4a[edge_list[i][2][1]][2][0],
                                                cryptoContext)
                    temp_list[1] = conv2(int(i != 0)+group_num, j, k, 1, input_rotate_list[i][0][0], pre_weight_lista[0][0][0][1]* pre_weight_list4a[i][0][1],
                                            cryptoContext)
                    temp_list[2] = conv2(int(i != 0)+group_num, j, k, 2, input_rotate_list[i][0][1], pre_weight_lista[0][0][0][2]* pre_weight_list4a[i][0][2],
                                            cryptoContext)
                    temp_list[4] = conv2(int(i != 0)+group_num, j, k, 4, input_rotate_list[i][1][0], pre_weight_lista[0][0][1][1]* pre_weight_list4a[i][1][1],
                                            cryptoContext)
                    temp_list[5] = conv2(int(i != 0)+group_num, j, k, 5, input_rotate_list[i][1][1], pre_weight_lista[0][0][1][2]* pre_weight_list4a[i][1][2],
                                            cryptoContext)
                    temp_list[7] = conv2(int(i != 0)+group_num, j, k, 7, input_rotate_list[i][2][0], pre_weight_lista[0][0][2][1]* pre_weight_list4a[i][2][1],
                                            cryptoContext)
                    temp_list[8] = conv2(int(i != 0)+group_num, j, k, 8, input_rotate_list[i][2][1], pre_weight_lista[0][0][2][2]* pre_weight_list4a[i][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                    fhe.homo_rotate(output_giant_[g],
                                                                                    -giant * g * temp,
                                                                                    cryptoContext),
                                                                    cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][0][0], pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                            cryptoContext)
                    temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][0][1], pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                            cryptoContext)
                    temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[i][0][2], pre_weight_list_edgea[i][0][2] * pre_weight_list4a[i][0][2],
                                            cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][1][0], pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                            cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][1][1], pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                            cryptoContext)
                    temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[i][1][2], pre_weight_list_edgea[i][1][2] * pre_weight_list4a[i][1][2],
                                            cryptoContext)
                    temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[i][2][0], pre_weight_list_edgea[i][2][0] * pre_weight_list4a[i][2][0],
                                            cryptoContext)
                    temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[i][2][1], pre_weight_list_edgea[i][2][1] * pre_weight_list4a[i][2][1],
                                            cryptoContext)
                    temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[i][2][2], pre_weight_list_edgea[i][2][2] * pre_weight_list4a[i][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                    fhe.homo_rotate(output_giant_[g],
                                                                                    -giant * g * temp,
                                                                                    cryptoContext),
                                                                    cryptoContext)
            if j > (((wi + 1) / 2) - 1):
                if k == wi - 1:
                        temp_list = np.empty((9), dtype=object)
                        temp_list1 = np.empty((9), dtype=object)
                        if edge_list[i][3][0] != 0:
                            temp_list[2] = conv2(i+group_num, j, k, 2,
                                                 input_rotate_list2[edge_list[i][3][1]][1][0][edge_list[i][3][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][3][0]),
                                                                       0)][
                                                                   max(edge_list[i][3][0], 0)][0][2] *
                                                               pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                                               pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                               edge_list[i][3][0]),
                                                 cryptoContext)
                            temp_list[5] = conv2(i+group_num, j, k, 5,
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
                            temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[edge_list[i][3][1]][1][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][0][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][0][2] *
                                                 pre_weight_list4a[edge_list[i][3][1]][0][2],
                                                 cryptoContext)
                            temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[edge_list[i][3][1]][2][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][3][0]), 0)][
                                                     max(edge_list[i][3][0], 0)][1][2] *
                                                 pre_weight_list3a[edge_list[i][3][1]][1][2] *
                                                 pre_weight_list4a[edge_list[i][3][1]][1][2],
                                                 cryptoContext)
                        if edge_list[i][7][0] != 0:
                            temp_list[8] = conv2(i+group_num, j, k, 8,
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
                            temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][7][1]][0][0],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][7][0]), 0)][
                                                     max(edge_list[i][7][0], 0)][2][2] *
                                                 pre_weight_list3a[edge_list[i][7][1]][2][2] *
                                                 pre_weight_list5a[edge_list[i][7][1]][2][2],
                                                 cryptoContext)
                        if edge_list[i][1][0] != 0:
                            temp_list[6] = conv2(i+group_num, j, k, 6,
                                                 input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                 rotate_weight(pre_weight_lista[
                                                                   max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                       0)][
                                                                   max(edge_list[i][1][0], 0)][2][0] *
                                                               pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                                               pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                               edge_list[i][1][0]),
                                                 cryptoContext)
                            temp_list[7] = conv2(i+group_num, j, k, 7,
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
                            temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][1],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][0] *
                                                 pre_weight_list2a[edge_list[i][1][1]][2][0] *
                                                 pre_weight_list5a[edge_list[i][1][1]][2][0],
                                                 cryptoContext)
                            temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][2],
                                                 pre_weight_lista[
                                                     max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                     max(edge_list[i][1][0], 0)][2][1] *
                                                 pre_weight_list2a[edge_list[i][1][1]][2][1] *
                                                 pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                 cryptoContext)
                        temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][1][2],
                                             pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                             cryptoContext)
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][1][1],
                                             pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                             cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][2][1],
                                             pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                             cryptoContext)
                        temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][2][2],
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
                                output_cipher[i+group_num, j, k] = output_giant_[g]
                            else:
                                
                                
                                output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                      fhe.homo_rotate(output_giant_[g],
                                                                                      -giant * g * temp,
                                                                                      cryptoContext),
                                                                      cryptoContext)
                if k == 0:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[7] = conv2(i+group_num, j, k, 7,
                                                input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                    0)][
                                                                max(edge_list[i][1][0], 0)][2][1] *
                                                            pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                            edge_list[i][1][0]),
                                                cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8,
                                                input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                    0)][
                                                                max(edge_list[i][1][0], 0)][2][2] *
                                                            pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                            edge_list[i][1][0]),
                                                cryptoContext)
                    else:
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][0],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                    max(edge_list[i][1][0], 0)][2][1] *
                                                pre_weight_list5a[edge_list[i][1][1]][2][1],
                                                cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][1],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                    max(edge_list[i][1][0], 0)][2][2] *
                                                pre_weight_list5a[edge_list[i][1][1]][2][2],
                                                cryptoContext)
                    if edge_list[i][6][0] != 0:
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                                input_rotate_list2[edge_list[i][6][1]][0][2][edge_list[i][6][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][6][0]),
                                                                    0)][
                                                                max(edge_list[i][6][0], 0)][2][0] *
                                                            pre_weight_list5a[edge_list[i][6][1]][2][0],
                                                            edge_list[i][6][0]),
                                                cryptoContext)
                    else:
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][6][1]][0][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][6][0]), 0)][
                                                    max(edge_list[i][6][0], 0)][2][0] *
                                                pre_weight_list5a[edge_list[i][6][1]][2][0],
                                                cryptoContext)
                    if edge_list[i][2][0] != 0:
                        temp_list[0] = conv2(i+group_num, j, k, 0,
                                                input_rotate_list2[edge_list[i][2][1]][1][2][edge_list[i][2][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                    0)][
                                                                max(edge_list[i][2][0], 0)][0][0] *
                                                            pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                            edge_list[i][2][0]),
                                                cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3,
                                                input_rotate_list2[edge_list[i][2][1]][2][2][edge_list[i][2][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][2][0]),
                                                                    0)][
                                                                max(edge_list[i][2][0], 0)][1][0] *
                                                            pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                            edge_list[i][2][0]),
                                                cryptoContext)
                    else:
                        temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[edge_list[i][2][1]][1][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                    max(edge_list[i][2][0], 0)][0][0] *
                                                pre_weight_list4a[edge_list[i][2][1]][0][0],
                                                cryptoContext)
                        temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[edge_list[i][2][1]][2][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][2][0]), 0)][
                                                    max(edge_list[i][2][0], 0)][1][0] *
                                                pre_weight_list4a[edge_list[i][2][1]][1][0],
                                                cryptoContext)
                    temp_list[1] = conv2(int(i != 0)+group_num, j, k, 1, input_rotate_list[i][1][0],
                                            pre_weight_lista[0][0][0][1] * pre_weight_list4a[i][0][1],
                                            cryptoContext)
                    temp_list[2] = conv2(int(i != 0)+group_num, j, k, 2, input_rotate_list[i][1][1],
                                            pre_weight_lista[0][0][0][2] * pre_weight_list4a[i][0][2],
                                            cryptoContext)
                    temp_list[4] = conv2(int(i != 0)+group_num, j, k, 4, input_rotate_list[i][2][0],
                                            pre_weight_lista[0][0][1][1] * pre_weight_list4a[i][1][1],
                                            cryptoContext)
                    temp_list[5] = conv2(int(i != 0)+group_num, j, k, 5, input_rotate_list[i][2][1],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
                                                                    fhe.homo_rotate(output_giant_[g],
                                                                                    -giant * g * temp,
                                                                                    cryptoContext),
                                                                    cryptoContext)
                if k == 1:
                    temp_list = np.empty((9), dtype=object)
                    temp_list1 = np.empty((9), dtype=object)
                    if edge_list[i][1][0] != 0:
                        temp_list[6] = conv2(i+group_num, j, k, 6,
                                                input_rotate_list2[edge_list[i][1][1]][0][0][edge_list[i][1][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                    0)][
                                                                max(edge_list[i][1][0], 0)][2][0] *
                                                            pre_weight_list5a[edge_list[i][1][1]][2][0]*
                                                pre_weight_list2a[edge_list[i][1][1]][2][0],
                                                            edge_list[i][1][0]),
                                                cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7,
                                                input_rotate_list2[edge_list[i][1][1]][0][1][edge_list[i][1][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                    0)][
                                                                max(edge_list[i][1][0], 0)][2][1] *
                                                            pre_weight_list5a[edge_list[i][1][1]][2][1]*
                                                pre_weight_list2a[edge_list[i][1][1]][2][1],
                                                            edge_list[i][1][0]),
                                                cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8,
                                                input_rotate_list2[edge_list[i][1][1]][0][2][edge_list[i][1][0]],
                                                rotate_weight(pre_weight_lista[
                                                                max(int(zero_num_temp[i] - edge_list[i][1][0]),
                                                                    0)][
                                                                max(edge_list[i][1][0], 0)][2][2] *
                                                            pre_weight_list5a[edge_list[i][1][1]][2][2]*
                                                pre_weight_list2a[edge_list[i][1][1]][2][2],
                                                            edge_list[i][1][0]),
                                                cryptoContext)
                    else:
                        temp_list[6] = conv2(i+group_num, j, k, 6, input_rotate_list[edge_list[i][1][1]][0][0],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                    max(edge_list[i][1][0], 0)][2][0] *
                                                pre_weight_list5a[edge_list[i][1][1]][2][0]*
                                                pre_weight_list2a[edge_list[i][1][1]][2][0],
                                                cryptoContext)
                        temp_list[7] = conv2(i+group_num, j, k, 7, input_rotate_list[edge_list[i][1][1]][0][1],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                    max(edge_list[i][1][0], 0)][2][1] *
                                                pre_weight_list5a[edge_list[i][1][1]][2][1]*
                                                pre_weight_list2a[edge_list[i][1][1]][2][1],
                                                cryptoContext)
                        temp_list[8] = conv2(i+group_num, j, k, 8, input_rotate_list[edge_list[i][1][1]][0][2],
                                                pre_weight_lista[
                                                    max(int(zero_num_temp[i] - edge_list[i][1][0]), 0)][
                                                    max(edge_list[i][1][0], 0)][2][2] *
                                                pre_weight_list5a[edge_list[i][1][1]][2][2]*
                                                pre_weight_list2a[edge_list[i][1][1]][2][2],
                                                cryptoContext)
                    temp_list[0] = conv2(i+group_num, j, k, 0, input_rotate_list[i][1][0],
                                            pre_weight_list_edgea[i][0][0] * pre_weight_list4a[i][0][0],
                                            cryptoContext)
                    temp_list[1] = conv2(i+group_num, j, k, 1, input_rotate_list[i][1][1],
                                            pre_weight_list_edgea[i][0][1] * pre_weight_list4a[i][0][1],
                                            cryptoContext)
                    temp_list[2] = conv2(i+group_num, j, k, 2, input_rotate_list[i][1][2],
                                            pre_weight_list_edgea[i][0][2] * pre_weight_list4a[i][0][2],
                                            cryptoContext)
                    temp_list[3] = conv2(i+group_num, j, k, 3, input_rotate_list[i][2][0],
                                            pre_weight_list_edgea[i][1][0] * pre_weight_list4a[i][1][0],
                                            cryptoContext)
                    temp_list[4] = conv2(i+group_num, j, k, 4, input_rotate_list[i][2][1],
                                            pre_weight_list_edgea[i][1][1] * pre_weight_list4a[i][1][1],
                                            cryptoContext)
                    temp_list[5] = conv2(i+group_num, j, k, 5, input_rotate_list[i][2][2],
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
                            output_cipher[i+group_num, j, k] = output_giant_[g]
                        else:
                            
                            
                            output_cipher[i+group_num, j, k] = fhe.homo_add(output_cipher[i+group_num, j, k],
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
                output_cipher[i+group_num][j][k] = fhe.homo_rescale(output_cipher[i+group_num][j][k], 1, cryptoContext)
    else:
        for i in range(group_num):
            for j in range(wi):
                for k in range(wo):
                    output_cipher[i][j][k] = fhe.homo_rescale(output_cipher[i][j][k], 1, cryptoContext)
                    output_cipher[i][j][k] = fhe.homo_add(output_cipher[i][j][k], res_initial[i][j][k], cryptoContext)
    return output_cipher
def mask_top_or_bottom(pre_weight_list,b, wi, wo, height, width, batch_size, in_channel,
                       output_channel,layer):  
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    if layer=="downsample0":
        giant,baby = perfect_square_split(in_channel)  
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
def ceil_power_of_2(x):
    if x <= 0:
        return 1
    return 1 << math.ceil(math.log2(x))
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
def pre_mask2(wi, wo, height, width, batch_size, in_channel, cryptoContext,index):
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
def downsampling(input, pre_mask, wi, wo, height, width, batch_size, in_channel, output_channel, plan, cryptoContext,openfhe_context):
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
        if output_channel==32:
            r = old_idx // 12
            c = old_idx % 12
            if c == 11:
                return None  
            return int(r * 11 + c)
        if output_channel==64:
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
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
                                    temp_output2 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][0][2],
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
                                    temp_output3 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][2][0],
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
                                    temp_output4 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][2][2],
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
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
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
                                    temp_output4 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][2][1],
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
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
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
                                    temp_output4 = fhe.homo_rotate(fhe.homo_mul_pt(input[int(temp1[k][1])][1][2],
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
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
                                                                   pre_mask[int(temp1[k][0])], cryptoContext),batch_size * (int(temp1[k][0]) - k),cryptoContext)
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
                output[i][j][k]=fhe.homo_rescale(output[i][j][k],1,cryptoContext)
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
        output_location(locate_temp - int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[0] - \
        output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 5, 0] = \
        output_location(locate_temp - int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[0] - \
        output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 6, 0] = \
        output_location(locate_temp + int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[0] - \
        output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 7, 0] = \
        output_location(locate_temp + int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[0] - \
        output_location(locate_temp, wi, wo, height, width, batch_size, output_channel)[0]
        locate_list[i, 4, 1] = \
        output_location(locate_temp - int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 5, 1] = \
        output_location(locate_temp - int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 6, 1] = \
        output_location(locate_temp + int(height_pad / wi) - 1, wi, wo, height, width, batch_size, output_channel)[1]
        locate_list[i, 7, 1] = \
        output_location(locate_temp + int(height_pad / wi) + 1, wi, wo, height, width, batch_size, output_channel)[1]
    return locate_list
def read_values_from_file(i,j,k,val_name, level, slots, cryptoContext, mask,scale=1.0):
    if val_name=="layer4-conv1bn1-n1" or val_name=="layer4-conv1bn1-n2":
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
            values=values*mask
            name = f"{val_name}{i}{j}{k}"
            encoded = fhe.encode(values, name, level, slots, False, cryptoContext)
            return encoded
    if val_name=="layer7-conv1bn1-n1" or val_name=="layer7-conv1bn1-n2":
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
            values=values*mask
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
            values=values*mask
            name = f"{val_name}{i}{j}{k}"
            encoded = fhe.encode(values, name, level, slots, False, cryptoContext)
            return encoded
def homo_Aespa_perfect_square(i,j,k,x, filename, cryptoContext,left_mask,bottom_mask,left_=0,bottom_=0):
    if left_==0 and bottom_==0:
        if x.noise_deg >1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        temp=np.ones(32768)
        n1 = read_values_from_file(i,j,k,n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext, left_mask,scale)
        temp=np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i,j,k,n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots, cryptoContext,left_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
    elif left_==1 and bottom_==0:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        n1 = read_values_from_file(i,j,k,n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext,left_mask, scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i,j,k,n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots, cryptoContext,left_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
    elif left_==0 and bottom_==1:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        n1 = read_values_from_file(i,j,k,n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext,bottom_mask, scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i,j,k,n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots, cryptoContext,bottom_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
    elif left_==1 and bottom_==1:
        if x.noise_deg > 1:
            x = fhe.homo_rescale(x, 1, cryptoContext)
        n1_filename = filename + '-n1'
        n2_filename = filename + '-n2'
        slots = x.slots
        scale = 1
        n1 = read_values_from_file(i,j,k,n1_filename, cryptoContext.L - x.cur_limbs, slots, cryptoContext,left_mask*bottom_mask, scale)
        temp = np.ones(slots)
        temp1 = fhe.homo_add_pt(x, n1, cryptoContext)
        perfect_squre = fhe.homo_square(temp1, cryptoContext)
        perfect_squre = fhe.homo_rescale(perfect_squre, 1, cryptoContext)
        n2 = read_values_from_file(i,j,k,n2_filename, cryptoContext.L - perfect_squre.cur_limbs, slots, cryptoContext,left_mask*bottom_mask, scale)
        res = fhe.homo_add_pt(perfect_squre, n2, cryptoContext)
        return res
@fhe.utils.profile_python_function
def batch_homo_relu(input, filename, cryptoContext,left_mask,bottom_mask):
    temp=np.ones(32768)
    for i in range(input.shape[0]):
        if i==0:
            for j in range(3):
                for k in range(3):
                    if j==0 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp,temp)
                    if j==0 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp,temp)
                    if j==0 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,0)
                    if j==1 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp,temp)
                    if j==1 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp,temp)
                    if j==1 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,0)
                    if j==2 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],0,1)
                    if j==2 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],0,1)
                    if j==2 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,1)
        else:
            temp1=np.ones(32768)
            block = 16 * 128  
            start = 15 * 128  
            end = 16 * 128  
            for base in range(0, len(temp1), block):
                temp1[base + start: base + end] = 0
            for j in range(3):
                for k in range(3):
                    if j==0 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                    if j==0 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                    if j==0 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0]*temp1,1,0)
                    if j==1 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                    if j==1 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                    if j==1 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0],1,0)
                    if j==2 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0]*temp1,0,1)
                    if j==2 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0]*temp1,0,1)
                    if j==2 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0],1,1)
    return input
@fhe.utils.profile_python_function
def batch_homo_relu3(input, filename, cryptoContext,left_mask,bottom_mask):
    temp=np.ones(32768)
    temp1=np.ones(32768)
    start=3*128
    end=4*128
    block=4*128
    for i in range(0,len(temp1),block):
        temp1[i+start:i+end]=0
    for i in range(input.shape[0]):
        for j in range(3):
            for k in range(3):
                if j==0 and k==0:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                if j==0 and k==1:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                if j==0 and k==2:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0]*temp1,1,0)
                if j==1 and k==0:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                if j==1 and k==1:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                if j==1 and k==2:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0]*temp1,1,0)
                if j==2 and k==0:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0]*temp1,0,1)
                if j==2 and k==1:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0]*temp1,0,1)
                if j==2 and k==2:
                    input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0]*temp1,1,1)
    return input
@fhe.utils.profile_python_function
def batch_homo_relu2(input, filename, cryptoContext,left_mask,bottom_mask):
    temp1 = np.ones(32768)
    block = 8 * 128  
    start = 7 * 128  
    end = 8 * 128  
    temp=np.ones(32768)
    for base in range(0, len(temp1), block):
        temp1[base + start: base + end] = 0
    for i in range(input.shape[0]):
        if i==0:
            for j in range(3):
                for k in range(3):
                    if j==0 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp,temp)
                    if j==0 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,0)
                    if j==0 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,0)
                    if j==1 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],0,1)
                    if j==1 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,1)
                    if j==1 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,1)
                    if j==2 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],0,1)
                    if j==2 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,1)
                    if j==2 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0],1,1)
        else:
            for j in range(3):
                for k in range(3):
                    if j==0 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,temp1,temp1)
                    if j==0 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0],1,0)
                    if j==0 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0],1,0)
                    if j==1 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0]*temp1,0,1)
                    if j==1 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0],1,1)
                    if j==1 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0]*temp1,bottom_mask[i,j,k][0][0],1,1)
                    if j==2 and k==0:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0]*temp1,0,1)
                    if j==2 and k==1:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0]*temp1,1,1)
                    if j==2 and k==2:
                        input[i][j][k] = homo_Aespa_perfect_square(i,j,k,input[i][j][k], filename, cryptoContext,left_mask[i,j,k][0][0],bottom_mask[i,j,k][0][0]*temp1,1,1)
    return input
@fhe.utils.profile_python_function
def batch_homo_bs(input, index, logBsSlots_list, levelBudget_list, cryptoContext):
    output=np.empty((input.shape[0],input.shape[1],input.shape[2]),dtype=object)
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            for k in range(input.shape[2]):
                output[i][j][k] = fhe.homo_bootstrap(input[i][j][k], cryptoContext.L-index, logBsSlots_list[0], levelBudget_list[0], cryptoContext)
    return output
def final_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                 right_edge, bottom_edge, up_edge, index,cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer=="downsample0"or layer=="downsample1":
        a,b=pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
    else:
        a = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
        b=np.zeros((group_num,2**15))
    pre_weight_list = mask_top_or_bottom(a,b,  wi, wo, height, width, batch_size, in_channel,
                                         output_channel,layer)
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo,layer)  
    _, _, mask_weight_list2, mask_bias_list2 = pre_weight_edge2(right_edge, a, b, batch_size, in_channel,
                                                                output_channel, height, width, wi,
                                                                wo,layer)  
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo,layer)  
    _, _, mask_weight_list4, mask_bias_list4 = pre_weight_edge2(up_edge, a, b, batch_size, in_channel, output_channel,
                                                                height, width, wi,
                                                                wo,layer)  
    final_bias = pre_bias_list1 * mask_bias_list3
    return pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias,a
def final_weight2(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                 right_edge, bottom_edge, up_edge, index,cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    a, a2 = pre_weight2(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
    b=np.zeros((group_num,2**15))
    b2=np.zeros((group_num,2**15))
    pre_weight_list = mask_top_or_bottom(a, b, wi, wo, height, width, batch_size, in_channel,
                                         output_channel,layer)  
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
                                         output_channel,layer)  
    pre_weight_list1a, pre_bias_list1a, mask_weight_list1a, mask_bias_list1a = pre_weight_edge2(left_edge, a2, b2, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo, layer)  
    _, _, mask_weight_list2a, mask_bias_list2a = pre_weight_edge2(right_edge, a2, b2, batch_size, in_channel,
                                                                output_channel, height, width, wi,
                                                                wo, layer)  
    _, pre_bias_list2a, mask_weight_list3a, mask_bias_list3a = pre_weight_edge2(bottom_edge, a2, b2, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo, layer)  
    _, _, mask_weight_list4a, mask_bias_list4a = pre_weight_edge2(up_edge, a2, b2, batch_size, in_channel, output_channel,
                                                                height, width, wi,
                                                                wo, layer)  
    final_biasa = pre_bias_list1a * mask_bias_list3a
    return pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias,pre_weight_lista, pre_weight_list1a, mask_weight_list1a, mask_weight_list2a, mask_weight_list3a, mask_weight_list4a, b2, pre_bias_list1a, pre_bias_list2a, final_biasa
def final_weight3(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                 right_edge, bottom_edge, up_edge, index,cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer=="downsample0"or layer=="downsample1":
        a,b=pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
    else:
        a = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
        b=np.zeros((group_num,2**15))
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo,layer)  
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo,layer)  
    return mask_weight_list1, mask_weight_list3
# @fhe.utils.profile_python_function
def final_weight4(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, left_edge,
                 right_edge, bottom_edge, up_edge, index,cryptoContext):
    N = 65536
    slots = int(N / 2)
    num_in_cipher = int(slots / (output_channel * batch_size))
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    if layer=="downsample0"or layer=="downsample1":
        a,b=pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
    else:
        a = pre_weight(batch_size, in_channel, state_dict, output_channel, height, width, wi, wo, layer, scale, index,cryptoContext)
        b=np.zeros((group_num,2**15))
    pre_weight_list1, pre_bias_list1, mask_weight_list1, mask_bias_list1 = pre_weight_edge2(left_edge, a, b, batch_size,
                                                                                            in_channel, output_channel,
                                                                                            height, width, wi,
                                                                                            wo,layer)
    _, _, mask_weight_list2, mask_bias_list2 = pre_weight_edge2(right_edge, a, b, batch_size, in_channel,
                                                                output_channel, height, width, wi,
                                                                wo,layer)
    _, pre_bias_list2, mask_weight_list3, mask_bias_list3 = pre_weight_edge2(bottom_edge, a, b, batch_size, in_channel,
                                                                             output_channel, height, width, wi,
                                                                             wo,layer)
    final_bias = pre_bias_list1 * mask_bias_list3
    return b, pre_bias_list1, pre_bias_list2, final_bias,a
@fhe.utils.profile_python_function
def averagepool(input,batch_size,height,width,wi,wo,in_channel,output_channel,cryptoContext,openfhe_context):
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
                if i==0 and j==0 and k==0:
                    output=input[i][j][k]
                else:
                    output=fhe.homo_add(output,input[i][j][k],cryptoContext)
    for i in range(1,num_in_cipher):
        output=fhe.homo_add(output,fhe.homo_rotate(output.deep_copy(),i*batch_size,cryptoContext),cryptoContext)
    return output
@fhe.utils.profile_python_function
def fc(input_cipher,batch_size,height,width,wi,wo,in_channel,output_channel,cryptoContext,openfhe_context):
    model = get_Aespa_MutalChannel_PAF_resnet20()
    N = 65536
    slots = int(N / 2)
    output_channel=16
    repeat = int(slots / output_channel)  
    num_in_cipher = int(slots / (in_channel * batch_size))  
    pad = min_padding_to_next_multiple_of_k(height, 3)
    height_pad = height + pad
    width_pad = width + pad
    block_num = int(height_pad * width_pad / (wi * wo))  
    group_num = int(np.ceil(height_pad * width_pad / (num_in_cipher * wi * wo)))
    fc_weight=model.fc.weight.detach().numpy()
    fc_weight=fc_weight*(1/64)
    fc_bias=model.fc.bias.reshape(-1).detach().cpu().numpy()
    weight_pad = np.pad(fc_weight, ((0, 6), (0, 0)), mode='constant', constant_values=0)
    fc_pad=np.repeat(fc_bias,512)
    baby=4
    giant=4
    temp=batch_size*num_in_cipher
    templist=np.zeros((baby,giant,slots))
    templist_encode=np.zeros((baby,giant),dtype=object)
    output_giant = np.empty(baby, dtype=object)
    for b in range(baby):  
        for g in range(giant):  
            for r in range(int(4)):
                for i in range(output_channel):
                    input = (r*output_channel+g + i) % in_channel
                    output = (b * giant + i) % output_channel
                    templist[b][g][
                    i * temp + r * output_channel * temp:(i + 1) * temp + r * output_channel * temp] = \
                    weight_pad[output][input]
    for i in range(baby):
        for j in range(giant):
            templist_encode[i][j]=fhe.encode(templist[i][j],"",1,slots,False,cryptoContext)
    for g in range(giant):
            if g!=0:
                input_temp=fhe.homo_rotate(input_cipher.deep_copy(),g*temp,cryptoContext)
            else:
                input_temp=input_cipher.deep_copy()
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
            output = fhe.homo_add(output, fhe.homo_rotate(output_giant[b].deep_copy(), -giant * b * temp, cryptoContext),
                                    cryptoContext)
    output=fhe.homo_rescale(output,1,cryptoContext)
    temp_cipher1=output.deep_copy()
    temp_cipher2=output.deep_copy()
    temp_cipher3=output.deep_copy()
    output=fhe.homo_add(output,fhe.homo_rotate(temp_cipher1,16*temp,cryptoContext),cryptoContext)
    output=fhe.homo_add(output,fhe.homo_rotate(temp_cipher2,16*2*temp,cryptoContext),cryptoContext)
    output=fhe.homo_add(output,fhe.homo_rotate(temp_cipher3,16*3*temp,cryptoContext),cryptoContext)
    output=fhe.homo_add_pt(output,fhe.encode(fc_pad,"",1,slots,False,cryptoContext),cryptoContext)
    return output
def batch_CNN(loadorsave,pkl_dir):
    DATA_DIR = os.environ["DATA_DIR"]
    rotate_index_list = [128, 128 * 2, 128 * 3, 128 * 4, 128 * 5, 128 * 6, 128 * 7, -128, -128 * 2, -128 * 3, -128 * 4,
                         -128 * 5, -128 * 6, -128 * 7, 2048,-2048, 2048 * 2, 2048 * -2, 2048 * 3, 2048 * -4, 2048 * -8,
                         2048 * -12, 1024, -1024,3072,-2048*3,-4096,-2*4096,-3*4096,-4*4096,-5*4096,-6*4096,-7*4096,512*1,512*3,512*5,512*7,16*512,32*512,48*512]
    maxLevelsRemaining = 14
    logBsSlots_list = [15]
    logN = 16
    dnum = 3
    dcrtBits = 52
    firstMod = 55
    levelBudget_list = [[4,4]]
    rescaleTech = "FIXEDMANUAL"
    relu_degree = 180
    slots = 2 **15
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
    cryptoContext.weight_path = os.path.join(project_root, "ICML_2026_PackCNN", "data", "weights_aespa_20/")
    temp_finename=os.path.join(project_root, "ICML_2026_PackCNN", "data", "cifar10_resnet20-4118986f.pt")
    ckpt = torch.load(temp_finename, map_location="cpu")
    pre_encode_type = "middle"
    cryptoContext.pre_encode_type = pre_encode_type
    pkl_path = DATA_DIR + "/encode_20260212_150521.pkl"
    pkl_path = DATA_DIR + pkl_dir
    start_load = time.time()
    print("Performing preprocessing or preloading of weight values for subsequent computation...")
    if loadorsave==0:
        pass
    else:
        load_weight(pkl_path, cryptoContext)
    end_load = time.time()
    print("load weight time:", f" {end_load-start_load:.4f} seconds.")
    pre_mask = pre_mask2(3, 3, 32, 32, 128, 16, cryptoContext,1)
    pre_mask22 = pre_mask2(3, 3, 16, 16, 128, 32, cryptoContext,2)
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
    if loadorsave==0:
        input, batch_label = batch_input("save", 128, openfhe_context,cryptoContext)
    else:
        input, batch_label = batch_input("load", 128, openfhe_context,cryptoContext)
    scale=1
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
            b = np.zeros((8,32768))
            pre_bias_list1 = np.zeros((8,32768))
            pre_bias_list2 = np.zeros((8,32768))
            final_bias = np.zeros((8,32768))

            # NOTE: If the .pkl files have already been generated, the following code can be commented out.
            if loadorsave==0:
                pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,_=final_weight(128,4,state_dict,16,32,32,3,3,"",scale,left_edge,right_edge,bottom_edge,up_edge,1,cryptoContext)
            # NOTE: If the .pkl files have already been generated, the above code can be commented out.

            output = conv_batch2(input, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list, mask_weight_list1,
                                 mask_weight_list2, mask_weight_list3,
                                 mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 4, 16, 32, 32,
                                 3,
                                 3, cryptoContext, openfhe_context, True, None, f"0")
            output=batch_homo_relu(output,"conv1bn1", cryptoContext,mask_weight_list1,mask_weight_list3)
        return output
    def layer1(output, block_num):
        for i in range(block_num):
            templist = output.copy()
            conv2_mode = "save"
            if conv2_mode == "save":
                scale = 1
                pre_weight_list=np.zeros((16, 16, 3, 3, 4, 4, 32768))
                pre_weight_list1=np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list1=np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list2=np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list3=np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list4=np.zeros((8, 3, 3, 4, 4, 32768))
                b=np.zeros((8,32768))
                pre_bias_list1=np.zeros((8,32768))
                pre_bias_list2=np.zeros((8,32768))
                final_bias=np.zeros((8,32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave==0:
                    pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,_=final_weight(128,16,state_dict,16,32,32,3,3,f"layer1[{i}]",scale,left_edge,right_edge,bottom_edge,up_edge,1,cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2(output, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 16, 16, 32,
                                     32, 3, 3,
                                     cryptoContext, openfhe_context, True, None, f"1_{i * 2}")
                output=batch_homo_relu(output,f"layer{i+1}-conv{1}bn{1}", cryptoContext,mask_weight_list1,mask_weight_list3)
            if conv2_mode == "save":
                scale = 1
                pre_weight_list = np.zeros((16, 16, 3, 3, 4, 4, 32768))
                pre_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list1 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list2 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list3 = np.zeros((8, 3, 3, 4, 4, 32768))
                mask_weight_list4 = np.zeros((8, 3, 3, 4, 4, 32768))
                b = np.zeros((8,32768))
                pre_bias_list1 = np.zeros((8,32768))
                pre_bias_list2 = np.zeros((8,32768))
                final_bias = np.zeros((8,32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave==0:
                    pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,_=final_weight(128,16,state_dict,16,32,32,3,3,f"layer1[{i}]",scale,left_edge,right_edge,bottom_edge,up_edge,2,cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2(output, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 16, 16, 32,
                                     32, 3, 3,
                                     cryptoContext, openfhe_context, False, templist, f"1_{i * 2 + 1}")
                output=batch_homo_relu(output,f"layer{i+1}-conv{2}bn{2}", cryptoContext,mask_weight_list1,mask_weight_list3)
        return output
    def layer2(output, block_num):
        for i in range(block_num):
            templist = output.copy()
            conv2_mode = "save"
            if i==0:
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
                    if loadorsave==0:
                        pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,pre_weight_lista,pre_weight_list1a,mask_weight_list1a,mask_weight_list2a,mask_weight_list3a,mask_weight_list4a,b2,pre_bias_list1a,pre_bias_list2a,final_biasa=final_weight2(128,16,state_dict,16,32,32,3,3,f"layer2[{i}]",scale,left_edge,right_edge,bottom_edge,up_edge,1,cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv2_batch_extend(output, zero_num_temp, pre_weight_list, pre_weight_list1, edge_list,
                                         mask_weight_list1,
                                         mask_weight_list2, mask_weight_list3,
                                         mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 16, 32, 32,
                                         32, 3, 3,
                                         cryptoContext, openfhe_context, True, None, f"2_{i * 2}",down_need1,pre_weight_lista,pre_weight_list1a,mask_weight_list1a,mask_weight_list2a,mask_weight_list3a,mask_weight_list4a,b2,pre_bias_list1a,pre_bias_list2a,final_biasa)
                output = downsampling(output, pre_mask, 3, 3, 32, 32, 128, 16, 32, 2, cryptoContext, openfhe_context)
                mask_weight_list1 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list3 = np.zeros((5, 3, 3, 8, 4, 32768))
                mask_weight_list1, mask_weight_list3 = final_weight3(
                    128, 32, state_dict, 32, 16, 16, 3, 3, f"layer2[1]", scale, left_edge2, right_edge2, bottom_edge2,
                    up_edge2, 1, cryptoContext)
                output=batch_homo_relu2(output,f"layer{i+4}-conv{1}bn{1}", cryptoContext, mask_weight_list1, mask_weight_list3)
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
                    if loadorsave==0:
                        pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias,_ = final_weight(
                        128, 32, state_dict, 32, 16, 16, 3, 3, f"layer2[{i}]", scale, left_edge2, right_edge2, bottom_edge2,
                        up_edge2, 1,cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv_batch2_mod2(output, zero_num_temp2, pre_weight_list, pre_weight_list1, edge_list2,
                                                mask_weight_list1,
                                                mask_weight_list2, mask_weight_list3,
                                                mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128,
                                                32, 32, 16,
                                                16, 3, 3,
                                                cryptoContext, openfhe_context, True, None, f"2_{i * 2}",
                                                )
                output=batch_homo_relu2(output,f"layer{i+4}-conv{1}bn{1}", cryptoContext, mask_weight_list1, mask_weight_list3)
            if conv2_mode == "save":
                if i==0:
                    weight_down = np.zeros((1,1))

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave==0:
                        bias_down,pre_bias_list2,pre_bias_list3,final_bias,weight_down=final_weight4(128,32,state_dict,32,16,16,3,3,f"downsample0",scale,left_edge2,right_edge2,bottom_edge2,up_edge2,2,cryptoContext)
                        temp_finename = os.path.join(project_root, "ICML_2026_PackCNN", "data", "params1.npz")
                        np.savez(temp_finename,
                                bias_down=bias_down,
                                pre_bias_list2=pre_bias_list2,
                                pre_bias_list3=pre_bias_list3,
                                final_bias=final_bias)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.
                    temp_finename = os.path.join(project_root, "ICML_2026_PackCNN", "data", "params1.npz")
                    data = np.load(temp_finename)
                    bias_down = data["bias_down"]
                    pre_bias_list2 = data["pre_bias_list2"]
                    pre_bias_list3 = data["pre_bias_list3"]
                    final_bias = data["final_bias"]
                    conv2_test = np.empty((templist.shape[0] * 2, *templist.shape[1:]), dtype=object)
                    conv2_test[:templist.shape[0]] = templist  
                    conv2_test[templist.shape[0]:] = templist  
                    templist=downsampling(conv2_test, pre_mask, 3, 3, 32, 32, 128, 16, 32, 2, cryptoContext, openfhe_context)
                    templist=conv_downsampling(f"layer2",templist,weight_down,bias_down,pre_bias_list2,pre_bias_list3,final_bias,cryptoContext,openfhe_context,16,16,3,3,128,32,32)
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
                if loadorsave==0:
                    pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,_=final_weight(128,32,state_dict,32,16,16,3,3,f"layer2[{i}]",scale,left_edge2,right_edge2,bottom_edge2,up_edge2,2,cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2_mod2(output, zero_num_temp2, pre_weight_list, pre_weight_list1, edge_list2,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 32, 32, 16,
                                     16, 3, 3,
                                     cryptoContext, openfhe_context, False, templist, f"2_{i * 2 + 1}")
            if i==0 :
                index=0
                output = batch_homo_bs(output, index, logBsSlots_list, levelBudget_list, cryptoContext)
            output=batch_homo_relu2(output,f"layer{i+4}-conv{2}bn{2}", cryptoContext, mask_weight_list1, mask_weight_list3)
        return output
    def layer3(output, block_num):
        for i in range(block_num):
            templist = output.copy()
            conv2_mode = "save"
            if i==0:
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
                    if loadorsave==0:
                        pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,pre_weight_lista,pre_weight_list1a,mask_weight_list1a,mask_weight_list2a,mask_weight_list3a,mask_weight_list4a,b2,pre_bias_list1a,pre_bias_list2a,final_biasa=final_weight2(128,32,state_dict,32,16,16,3,3,f"layer3[{i}]",scale,left_edge2,right_edge2,bottom_edge2,up_edge2,1,cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv2_batch_extend2(output, zero_num_temp2, pre_weight_list, pre_weight_list1, edge_list2,
                                         mask_weight_list1,
                                         mask_weight_list2, mask_weight_list3,
                                         mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 32,64 , 16,
                                         16, 3, 3,
                                         cryptoContext, openfhe_context, True, None, f"3_{i * 2}",down_need1,pre_weight_lista,pre_weight_list1a,mask_weight_list1a,mask_weight_list2a,mask_weight_list3a,mask_weight_list4a,b2,pre_bias_list1a,pre_bias_list2a,final_biasa)
                output = downsampling(output, pre_mask22, 3, 3, 16, 16, 128, 32, 64, 2, cryptoContext, openfhe_context)
                mask_weight_list1 = np.zeros((3, 3, 3, 8, 8, 32768))
                mask_weight_list3 = np.zeros((3, 3, 3, 8, 8, 32768))

                # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                if loadorsave==0:
                    mask_weight_list1, mask_weight_list3 = final_weight3(
                    128, 64, state_dict, 64, 8, 8, 3, 3, f"layer3[1]", scale, left_edge3, right_edge3, bottom_edge3,
                    up_edge3, 1, cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output=batch_homo_relu3(output,f"layer{i+7}-conv{1}bn{1}", cryptoContext, mask_weight_list1, mask_weight_list3)
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
                    if loadorsave==0:
                        pre_weight_list, pre_weight_list1, mask_weight_list1, mask_weight_list2, mask_weight_list3, mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias,_ = final_weight(
                        128, 64, state_dict, 64, 8, 8, 3, 3, f"layer3[{i}]", scale, left_edge3, right_edge3, bottom_edge3,
                        up_edge3, 1,cryptoContext)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                    output = conv_batch2(output, zero_num_temp3, pre_weight_list, pre_weight_list1, edge_list3,
                                                mask_weight_list1,
                                                mask_weight_list2, mask_weight_list3,
                                                mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128,
                                                64, 64, 8,
                                                8, 3, 3,
                                                cryptoContext, openfhe_context, True, None, f"3_{i * 2}",
                                                )
                output=batch_homo_relu3(output,f"layer{i+7}-conv{1}bn{1}", cryptoContext, mask_weight_list1, mask_weight_list3)
            if conv2_mode == "save":
                if i==0:
                    weight_down = np.zeros((1, 1))

                    # NOTE: If the .pkl files have already been generated, the following code can be commented out.
                    if loadorsave==0:
                        bias_down,pre_bias_list2,pre_bias_list3,final_bias,weight_down=final_weight4(128,64,state_dict,64,8,8,3,3,f"downsample1",scale,left_edge3,right_edge3,bottom_edge3,up_edge3,2,cryptoContext)
                        temp_finename = os.path.join(project_root, "ICML_2026_PackCNN", "data", "params2.npz")
                        np.savez(temp_finename,
                             bias_down=bias_down,
                             pre_bias_list2=pre_bias_list2,
                             pre_bias_list3=pre_bias_list3,
                             final_bias=final_bias)
                    # NOTE: If the .pkl files have already been generated, the above code can be commented out.
                    temp_finename = os.path.join(project_root, "ICML_2026_PackCNN", "data", "params2.npz")
                    data = np.load(temp_finename)
                    bias_down = data["bias_down"]
                    pre_bias_list2 = data["pre_bias_list2"]
                    pre_bias_list3 = data["pre_bias_list3"]
                    final_bias = data["final_bias"]
                    conv2_test = np.empty((templist.shape[0] * 2, *templist.shape[1:]), dtype=object)
                    conv2_test[:templist.shape[0]] = templist  
                    conv2_test[templist.shape[0]:] = templist  
                    templist=downsampling(conv2_test, pre_mask22, 3, 3, 16, 16, 128, 32, 64, 2, cryptoContext, openfhe_context)
                    templist=conv_downsampling(f"layer3",templist,weight_down,bias_down,pre_bias_list2,pre_bias_list3,final_bias,cryptoContext,openfhe_context,8,8,3,3,128,64,64)
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
                if loadorsave==0:
                    pre_weight_list,pre_weight_list1,mask_weight_list1,mask_weight_list2,mask_weight_list3,mask_weight_list4,b,pre_bias_list1,pre_bias_list2,final_bias,_=final_weight(128,64,state_dict,64,8,8,3,3,f"layer3[{i}]",scale,left_edge3,right_edge3,bottom_edge3,up_edge3,2,cryptoContext)
                # NOTE: If the .pkl files have already been generated, the above code can be commented out.

                output = conv_batch2(output, zero_num_temp3, pre_weight_list, pre_weight_list1, edge_list3,
                                     mask_weight_list1,
                                     mask_weight_list2, mask_weight_list3,
                                     mask_weight_list4, b, pre_bias_list1, pre_bias_list2, final_bias, 128, 64, 64, 8,
                                     8, 3, 3,
                                     cryptoContext, openfhe_context, False, templist, f"3_{i * 2 + 1}")
            if i==0:
                index=0
                output = batch_homo_bs(output, index, logBsSlots_list, levelBudget_list, cryptoContext)
            output=batch_homo_relu3(output,f"layer{i+7}-conv{2}bn{2}", cryptoContext, mask_weight_list1, mask_weight_list3)
        return output
    time_start=time.time()
    print("Starting privacy-preserving inference...")
    layer0 = initial_layer0()
    print("layer0 finished")
    output = layer1(layer0, 3)
    print("layer1 finished")
    layer2 = layer2(output, 3)
    print("layer2 finished")
    layer3 = layer3(layer2, 3)
    print("layer3 finished")
    average=averagepool(layer3,128,8,8,3,3,32,64,cryptoContext,openfhe_context)
    fc1=fc(average,128,8,8,3,3,64,10,cryptoContext,openfhe_context)
    time_end=time.time()-time_start
    print(f"Ciphertext inference finished. Total time (including encoding): {time_end:.4f} s "
          f"(encoding time is excluded in our experiments for both our method and the baseline).")
    aaa=openfhe_context.decrypt(fc1).cpu().numpy().reshape(-1)
    mat = aaa.reshape(64, 512).T
    sub = mat[:, :10]
    batch_size=128
    correct=0
    FHE_result = np.argmax(sub, axis=1)
    for i in range(batch_size):
        print(f"For image {i}, the inference result is {FHE_result[i]}, and the correct label is {int(batch_label[i])}.")
        if FHE_result[i]==batch_label[i]:
            correct+=1
    print(f"Correct: {correct}/{batch_size}, Accuracy: {(correct / batch_size) * 100:.2f}%")
    # print(f"\n\ncorrect/total: {correct}/{batch_size}")
    return 0
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
if __name__ == "__main__":
    loadorsave = int(sys.argv[-2])
    pkl_dir = sys.argv[-1]
    batch_CNN(loadorsave,pkl_dir)
