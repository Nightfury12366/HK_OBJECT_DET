
"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""

import argparse
import cv2
import os


def get_args():
    parser = argparse.ArgumentParser('SkyLake HK_Object_detection Video Test')
    # 默认检测国旗国徽
    parser.add_argument('--is_dibiao', type=bool, default=False, help='检测国旗还是地标')
    args = parser.parse_args()
    return args


# 主函数也改好了
if __name__ == '__main__':
    opt = get_args()
    if opt.is_dibiao is False:  # 判断加载地标模型还是国旗国徽模型
        data_path = 'video/guoqi'
    else:
        data_path = 'video/dibiao'
    # Video's path
    video_src = data_path  # set int to use webcam, set str to read from a video file
    path_list = os.listdir(video_src)
    for video in path_list:  # 进度条
        # Video capture
        cap = cv2.VideoCapture(os.path.join(video_src, video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sum_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print("Name: ", video.ljust(15), "  FPS: ", int(fps))
        cap.release()
    cv2.destroyAllWindows()


