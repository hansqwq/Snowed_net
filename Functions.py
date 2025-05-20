import numpy as np

def zeroSupplement(bitArray):
    if len(bitArray) == 3:
        return bitArray
    elif len(bitArray) < 3:
        for i in range(0, 3 - len(bitArray)):
            bitArray = np.insert(bitArray, 0, False)  # Supply 0 at top
        return bitArray
    else:
        raise IndexError("Please check the dimension of your koch network.")

def inputTransform(addr):
    """
    将地址列表转换为字符串形式的键，不截断任何对子标签，
    直接用空格连接所有元素，并用中括号括起来。
    例如：["1", "0", "2", "1"] -> "[1 0 2 1]"
    """
    return "[" + " ".join(addr) + "]"