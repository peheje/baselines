import math


def f1(std):
    return -0.5 * (math.log(2 * math.pi * std ** 2) + 1)


def f2(std):
    return -(math.log(std) + 0.5 * math.log(2.0 * math.pi * math.e))


def f3(std):
    return -math.log(std*math.sqrt(2*math.pi*math.e))


for i in range(1, 10):
    print("f1", f1(i))
    print("f2", f2(i))
    print("f3", f3(i))
