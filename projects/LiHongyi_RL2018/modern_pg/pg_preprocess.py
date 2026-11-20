import cv2
import numpy as np

#数据预处理，解决图像问题
def preprocess_frame(observation):

    # 1. RGB 彩色图像 → 灰度图
    gray = cv2.cvtColor(
        observation,
        cv2.COLOR_RGB2GRAY
    )

    # 2. 缩放图像到 80×80
    resized = cv2.resize(
        gray,
        (80, 80)
    )

    # 3. 转为 float32，并归一化到 [0,1]
    processed = resized.astype(np.float32) / 255.0

    return processed

#计算帧差，返回Numpy数组
def compute_frame_difference(current_frame, previous_frame):

    if previous_frame is None:
        return np.zeros_like(current_frame)

    return current_frame - previous_frame

