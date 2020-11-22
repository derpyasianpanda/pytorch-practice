# Utility Functions
from math import log2
from random import randint


def int_to_bits(number, bit_len=0):
    bits = [int(bit) for bit in list(bin(number))[2:]]
    if bit_len:
        bits = ([0] * (bit_len - len(bits))) + bits
    return bits


def generate_evens(max, amount):
    labels = [0] * amount
    data = [
        int_to_bits(number, int(log2(max)))
        for number
        in [randint(0, max + 1) for _ in range(amount)]
    ]
    return labels, data