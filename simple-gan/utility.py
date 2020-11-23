# Utility Functions
from random import randint

def bits_to_int(bits):
    return int(sum([2 ** (len(bits) - i - 1) * bit for (i, bit) in enumerate(bits)]))

def int_to_bits(number, bit_len=0):
    bits = [int(bit) for bit in list(bin(number))[2:]]
    if bit_len:
        bits = ([0] * (bit_len - len(bits))) + bits
    return bits


def generate_evens(bit_len, amount):
    labels = [1] * amount
    data = [
        int_to_bits(number, bit_len)
        for number
        in [randint(0, ((2 ** bit_len) - 1) // 2) * 2 for _ in range(amount)]
    ]
    return labels, data