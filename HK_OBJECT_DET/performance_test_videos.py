## Core Author: Zylo117
# Script's Author: winter2897

"""
Simple Inference Script of EfficientDet-Pytorch for detecting objects on webcam
"""
import base64

import numpy as np
import argparse
import time
import torch
import cv2
# import numpy as np
from torch.backends import cudnn
from backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, preprocess_video
import json
import yaml

import os
from tqdm import tqdm
import sys
import string
import re


def get_args():
    parser = argparse.ArgumentParser('SkyLake HK_Object_detection Video Test')
    # 输入视频的路径
    parser.add_argument('--data_path', type=str, default='video/guoqi',
                        help='the root folder of dataset')  # 默认是国旗视频文件夹
    # 默认检测国旗国徽
    parser.add_argument('--is_dibiao', type=bool, default=False, help='检测国旗还是地标')
    args = parser.parse_args()
    return args


# 主函数也改好了
if __name__ == '__main__':
    compound_coef = 0
    force_input_size = None  # set None to use default size

    threshold = 0.85
    iou_threshold = 0.05

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    opt = get_args()

    if opt.is_dibiao is False:  # 判断加载地标模型还是国旗国徽模型
        params = yaml.safe_load(open(f'projects/display_GuoQi.yml'))
        net_path = f'weights/efficientdet-d0_guoqi.pth'
    else:
        params = yaml.safe_load(open(f'projects/display_DiBiao.yml'))
        net_path = f'weights/efficientdet-d0_dibiao.pth'

    obj_list = params['obj_list']

    landmark_id = params['landmark_id']
    landmark_dict = params['landmark_dict']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(net_path))
    model.requires_grad_(False)
    model.eval()

    Object_Dict = {}  # 大字典

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video's path
    video_src = opt.data_path  # set int to use webcam, set str to read from a video file
    path_list = os.listdir(video_src)
    progress_bar = tqdm(path_list)
    for video in path_list:  # 进度条
        # Video capture
        # print("######################### "+video+" #########################")
        cap = cv2.VideoCapture(os.path.join(video_src, video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        sum_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # print(fps)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        video_name = video

        Object_Dict[video_name] = {}  # 小字典

        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1  # 帧号
            # frame preprocessing
            ori_imgs, framed_imgs, framed_metas = preprocess_video(frame, max_size=input_size)

            if use_cuda:
                x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
            else:
                x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

            x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

            # model predict
            with torch.no_grad():
                features, regression, classification, anchors = model(x)

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)

            # 添加带有时间戳的对象
            progress_bar.set_description('Video: {}. Progress: {}/{}'.format(video, count, sum_count))
            # print('处理进度：', count, '/', sum_count)
            remove_index = []  # 要在显示中剔除的目标框
            if len(out[0]['class_ids']) > 0:

                for index in range(len(out[0]['class_ids'])):
                    i = out[0]['class_ids'][index]
                    # 调整各个目标之间的阈值
                    if (obj_list[int(i)] == 'jzjgc' and out[0]['scores'][index] < 0.9) or (
                            obj_list[int(i)] == 'xghzzx' and out[0]['scores'][index] < 0.95) or (
                            obj_list[int(i)] == 'lbf' and out[0]['scores'][index] <= 0.915):
                        remove_index.append(index)
                        continue
                    if obj_list[int(i)] not in Object_Dict[video_name].keys():  # 这个目标不在字典之中，是新目标
                        Object_Dict[video_name][obj_list[int(i)]] = []  # 物体列表
                        Object_Dict[video_name][obj_list[int(i)]].append([count, '-', count])  # 加上本身到本身
                    else:
                        time_last = Object_Dict[video_name][obj_list[int(i)]][-1][-1]
                        if abs(count - time_last) < fps / 2:
                            Object_Dict[video_name][obj_list[int(i)]][-1][-1] = count
                        else:
                            Object_Dict[video_name][obj_list[int(i)]].append(
                                [count, '-', count])

        # 去重复
        for key in list(Object_Dict[video_name].keys()):
            for a_time_list in Object_Dict[video_name][key][:]:
                if a_time_list[2] - a_time_list[0] <= fps:  # 小于1秒的物体删除
                    if len(Object_Dict[video_name][key]) == 1:
                        del Object_Dict[video_name][key]
                        break
                    else:
                        Object_Dict[video_name][key].remove(a_time_list)

        cap.release()
        progress_bar.update()

    data = json.dumps(Object_Dict, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    cv2.destroyAllWindows()

    fileObject = open('performance_test_guoqi.json', 'w')
    fileObject.write(data)
    fileObject.close()
