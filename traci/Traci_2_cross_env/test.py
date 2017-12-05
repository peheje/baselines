import os

folder = "/Users/phj/Dropbox/Code/banditstone"
newest1 = max(os.listdir(folder), key=lambda f: os.path.getctime("{}/{}".format(folder, f)))
print(newest1)