import os

newest = sorted(os.listdir("."), key=os.path.getctime)[-1]
print(newest)