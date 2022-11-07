import numpy as np

def binary(integer):
    """
    Returns the binary representation of a positive integer as a string. E.g. 10 is returned  as "1010"

    :param integer: positive integer
    :return: binary representation of integer as string
    """
    return np.binary_repr(integer)

def count_binary(integer):
    """
    Counts the amount of 0s and 1s in the binary representation of given positive integer.

    :param integer: positive integer
    :return: 2 tuple, consisting of amount of zeroes in the first entry, and amount of 1s in the second entry.
    """
    b = binary(integer)
    return (b.count("0"), b.count("1"))



def bitflip(integer, index):
    """
    Flips the bit of the integers binary representation at given index.
    Index counts from the  right, i.e. the 0 in the binary of 11, ergo "1101", is at index 1.

    Done by  bitshifting to the left index times.

    :param integer: positive integer
    :param index: index to flip the bit at
    :return: integer of the binary number with the flipped bit
    """
    x = integer
    x_flip = (x ^ (1 << index))
    return x_flip




