import os
import numpy as np
import cv2

files_dir = '/home/sun/facades_datasets/1.CMP/CMP_base/base'
files = os.listdir(files_dir)

R = 0.
G = 0.
B = 0.
N = 0

for file in files:
    if os.path.splitext(file)[-1] == ".jpg":
        img = cv2.imread(os.path.join(files_dir,file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        h, w, c = img.shape
        N += h * w

        R_t = img[:, :, 0]
        R += np.sum(R_t)

        G_t = img[:, :, 1]
        G += np.sum(G_t)

        B_t = img[:, :, 2]
        B += np.sum(B_t)

R_mean = R/N
G_mean = G/N
B_mean = B/N

print("R_mean: %f, G_mean: %f, B_mean: %f" % (R_mean, G_mean, B_mean))


R = 0.
G = 0.
B = 0.
N = 0

for file in files:
    if os.path.splitext(file)[-1] == ".jpg":
        img = cv2.imread(os.path.join(files_dir,file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(img)
        h, w, c = img.shape
        N += h * w

        R_t = img[:, :, 0]
        R += np.sum(np.square(R_t-R_mean))

        G_t = img[:, :, 1]
        G += np.sum(np.square(G_t-G_mean))

        B_t = img[:, :, 2]
        B += np.sum(np.square(B_t-B_mean))

R_std=np.sqrt(R/N)
G_std=np.sqrt(G/N)
B_std=np.sqrt(B/N)

print("R_std: %f, G_std: %f, B_std: %f" % (R_std, G_std, B_std))