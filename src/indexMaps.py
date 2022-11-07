from binaryOps import *
import operators

def create_index_to_state(systemSize, totalSpin = None):
    d = {}
    index = 0
    for i in range(2**systemSize):
        if (totalSpin is  None) or (operators.total_spin(i, systemSize) == totalSpin):
            d[index] = i
            index += 1
    return d

def create_state_to_index(index_to_state):
    inv_map = {elem: key for key, elem in index_to_state.items()}
    return inv_map