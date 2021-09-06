import cv2
import glob
import os
import numpy as np


data_path = r"C:\Users\17286\Desktop\dice"
pred_path = glob.glob(os.path.join(data_path, "test03/*.png"))
tar_path = glob.glob(os.path.join(data_path, "test04/*.png"))

for index in range(0, 90):
    predict_path = pred_path[index]
    target_path = tar_path[index]

    predict = cv2.imread(predict_path)
    target = cv2.imread(target_path)

    predict = cv2.cvtColor(predict, cv2.COLOR_BGR2GRAY)
    target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

    row = predict.shape[0]
    col = predict.shape[1]

    d = []
    s = []
    for r in range(row):
        for c in range(col):
            if target[r][c] == predict[r][c]:
                s.append(target[r][c])

    m1 = np.linalg.norm(s)   #默认2范数
    m2 = np.linalg.norm(target.flatten()) + np.linalg.norm(predict.flatten())
    d.append(2 * m1 / m2)
    msg = "这是第{}张图的dice系数".format(index) + str(2 * m1 / m2)
    print(msg)





