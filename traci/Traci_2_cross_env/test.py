def ternary(n):
    """ https://stackoverflow.com/questions/34559663/convert-decimal-to-ternarybase3-in-python """
    if n == 0:
        return [0, 0, 0, 0]
    nums = []
    while n:
        n, r = divmod(n, 3)
        nums.append(r)
    for i in range(4-len(nums)):
        nums.append(0)
    return list(reversed(nums))


for i in range(3**4):
    print(ternary(i))