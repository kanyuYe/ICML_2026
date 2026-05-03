import math
import numpy as np


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


def perfect_square_split(n):
    if n <= 0:
        raise ValueError("n must be integater")
    root = int(math.sqrt(n))
    if root * root == n:
        return root, root
    for i in range(root, 0, -1):
        if n % i == 0:
            return n // i, i


def min_padding_to_next_multiple_of_k(n, k):
    r = n % k
    if r == 0:
        return 0
    else:
        return k - r


def ceil_power_of_2(x):
    if x <= 0:
        return 1
    return 1 << math.ceil(math.log2(x))
