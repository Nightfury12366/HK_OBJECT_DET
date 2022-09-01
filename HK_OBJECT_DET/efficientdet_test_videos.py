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
import sys
import string
import re


def get_args():
    parser = argparse.ArgumentParser('SkyLake HK_Object_detection Video Test')
    # 输入视频的路径
    parser.add_argument('--data_path', type=str, default='video/4.mp4', help='the root folder of dataset')
    # 默认检测国旗国徽
    parser.add_argument('--is_dibiao', type=bool, default=False, help='检测国旗还是地标')
    # 是否是模型升级后的模型配置文件
    parser.add_argument('--is_update', type=bool, default=False, help='是否使用更新之后的模型')
    args = parser.parse_args()
    return args


# function for display， 改好了
def display(preds, imgs, remove_list):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            if j in remove_list:
                continue
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        return imgs[i]


# 改好了
def display_2(preds, imgs, obj_list, remove_list):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:
            return imgs[i]

        for j in range(len(preds[i]['rois'])):
            if j in remove_list:
                continue
            (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
            cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (255, 255, 0), 2)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            cv2.putText(imgs[i], '{}, {:.3f}'.format(obj, score),
                        (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)

        return imgs[i]


# 视频地标检测，改好了
def run(filename, savename):
    compound_coef = 0
    force_input_size = None  # set None to use default size

    threshold = 0.9
    iou_threshold = 0.05

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    params = yaml.safe_load(open(f'projects/display_DiBiao.yml'))

    obj_list = params['obj_list']
    landmark_id = params['landmark_id']
    landmark_dict = params['landmark_dict']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    net_path = f'weights/efficientdet-d0_dibiao.pth'

    # video_src = opt.data_path  # set int to use webcam, set str to read from a video file
    video_src = str('video/' + filename)
    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(net_path))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sum_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter(f'results/{savename}.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    Object_Dict = {"objects": {}}

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

        remove_index = []  # 要在显示中剔除的目标框
        # 添加带有时间戳的对象
        print('处理进度：', count, '/', sum_count)
        if len(out[0]['class_ids']) > 0:

            for index in range(len(out[0]['class_ids'])):
                i = out[0]['class_ids'][index]
                # 调整各个目标之间的阈值
                if (obj_list[int(i)] == 'jzjgc' and out[0]['scores'][index] < 0.9) or (
                        obj_list[int(i)] == 'xghzzx' and out[0]['scores'][index] < 0.95):
                    remove_index.append(index)
                    continue

                # print(obj_list[int(i)], count, out[0]['scores'][index])
                if obj_list[int(i)] not in Object_Dict['objects'].keys():  # 这个目标不在字典之中，是新目标

                    iobject = {"id": landmark_id[obj_list[int(i)]], "description": landmark_dict[obj_list[int(i)]],
                               "time": [], "image": ""}
                    iobject["time"].append([count / fps * 1.0, '-', count / fps * 1.0])

                    with open("demo_images/" + obj_list[int(i)] + ".jpg", 'rb') as f:
                        image_byte = base64.b64encode(f.read())
                    image_str = image_byte.decode('ascii')
                    iobject["image"] = image_str

                    Object_Dict['objects'][obj_list[int(i)]] = iobject
                else:
                    time_last = Object_Dict['objects'][obj_list[int(i)]]["time"][-1][-1]
                    if abs(count / fps * 1.0 - time_last) < 0.5:
                        Object_Dict['objects'][obj_list[int(i)]]["time"][-1][-1] = count / fps * 1.0

                    else:
                        Object_Dict['objects'][obj_list[int(i)]]["time"].append(
                            [count / fps * 1.0, '-', count / fps * 1.0])

                # print(out)
            # print(out[0]['class_ids'])

        # result
        out = invert_affine(framed_metas, out)
        img_show = display_2(out, ori_imgs, obj_list, remove_index)

        # show frame by frame # 显示图片

        # cv2.imshow('frame', img_show)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        videoWriter.write(img_show)
    # 去重复
    for key in list(Object_Dict['objects'].keys()):
        for a_time_list in Object_Dict['objects'][key]['time'][:]:
            if a_time_list[2] - a_time_list[0] <= 1.0:
                if len(Object_Dict['objects'][key]['time']) == 1:
                    del Object_Dict['objects'][key]
                    break
                else:
                    Object_Dict['objects'][key]['time'].remove(a_time_list)

    data = json.dumps(Object_Dict, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    cap.release()
    cv2.destroyAllWindows()

    fileObject = open('test_video_object_data.json', 'w')
    fileObject.write(data)
    fileObject.close()
    return Object_Dict


# 视频国旗国徽检测，改好了
def run2(filename, savename):
    compound_coef = 0
    force_input_size = None  # set None to use default size

    threshold = 0.85
    iou_threshold = 0.05

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    params = yaml.safe_load(open(f'projects/display_GuoQi.yml'))

    obj_list = params['obj_list']
    landmark_id = params['landmark_id']
    landmark_dict = params['landmark_dict']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    opt = get_args()
    # Video's path

    net_path = f'weights/efficientdet-d0_guoqi.pth'

    # video_src = opt.data_path  # set int to use webcam, set str to read from a video file
    video_src = str('video/' + filename)
    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(net_path))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sum_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    videoWriter = cv2.VideoWriter(f'results/{savename}.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    Object_Dict = {"objects": {}}

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

        print('处理进度：', count, '/', sum_count)
        # 添加带有时间戳的对象
        remove_index = []  # 要在显示中剔除的目标框
        if len(out[0]['class_ids']) > 0:

            for index in range(len(out[0]['class_ids'])):
                i = out[0]['class_ids'][index]
                if obj_list[int(i)] == 'fyb':  # 剔除反样本
                    remove_index.append(index)
                    continue
                # print(obj_list[int(i)], count, out[0]['scores'][index])
                if obj_list[int(i)] not in Object_Dict['objects'].keys():  # 这个目标不在字典之中，是新目标

                    iobject = {"id": landmark_id[obj_list[int(i)]], "description": landmark_dict[obj_list[int(i)]],
                               "time": [], "image": ""}
                    iobject["time"].append([count / fps * 1.0, '-', count / fps * 1.0])

                    with open("demo_images/" + obj_list[int(i)] + ".jpg", 'rb') as f:
                        image_byte = base64.b64encode(f.read())
                    image_str = image_byte.decode('ascii')
                    iobject["image"] = image_str

                    Object_Dict['objects'][obj_list[int(i)]] = iobject
                else:
                    time_last = Object_Dict['objects'][obj_list[int(i)]]["time"][-1][-1]
                    if abs(count / fps * 1.0 - time_last) < 0.5:
                        Object_Dict['objects'][obj_list[int(i)]]["time"][-1][-1] = count / fps * 1.0

                    else:
                        Object_Dict['objects'][obj_list[int(i)]]["time"].append(
                            [count / fps * 1.0, '-', count / fps * 1.0])

        # result
        out = invert_affine(framed_metas, out)
        img_show = display_2(out, ori_imgs, obj_list, remove_index)

        # show frame by frame # 显示图片

        # cv2.imshow('frame', img_show)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        videoWriter.write(img_show)
    # 去重复
    for key in list(Object_Dict['objects'].keys()):
        for a_time_list in Object_Dict['objects'][key]['time'][:]:
            if a_time_list[2] - a_time_list[0] <= 1.0:
                if len(Object_Dict['objects'][key]['time']) == 1:
                    del Object_Dict['objects'][key]
                    break
                else:
                    Object_Dict['objects'][key]['time'].remove(a_time_list)

    data = json.dumps(Object_Dict, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    cap.release()
    cv2.destroyAllWindows()

    fileObject = open('test_video_object_data.json', 'w')
    fileObject.write(data)
    fileObject.close()
    return Object_Dict


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

    # Video's path
    video_src = opt.data_path  # set int to use webcam, set str to read from a video file

    # load model
    model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list))
    model.load_state_dict(torch.load(net_path))
    model.requires_grad_(False)
    model.eval()

    if use_cuda:
        model = model.cuda()
    if use_float16:
        model = model.half()

    # Box
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    # Video capture
    cap = cv2.VideoCapture(video_src)
    fps = cap.get(cv2.CAP_PROP_FPS)
    sum_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(fps)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    haha = video_src.split('/')[-1].split('.')[0]  # 把.mp4 去掉
    videoWriter = cv2.VideoWriter(f'results/out_{haha}.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, size)
    Object_Dict = {"objects": {}}

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
        print('处理进度：', count, '/', sum_count)
        remove_index = []  # 要在显示中剔除的目标框
        if len(out[0]['class_ids']) > 0:

            for index in range(len(out[0]['class_ids'])):
                i = out[0]['class_ids'][index]
                # 调整各个目标之间的阈值
                if (obj_list[int(i)] == 'jzjgc' and out[0]['scores'][index] < 0.9) or (
                        obj_list[int(i)] == 'xghzzx' and out[0]['scores'][index] < 0.95)\
                        or (obj_list[int(i)] == 'lbf' and out[0]['scores'][index] <= 0.915):
                    remove_index.append(index)
                    continue
                # print(obj_list[int(i)], count)
                if obj_list[int(i)] not in Object_Dict['objects'].keys():  # 这个目标不在字典之中，是新目标

                    iobject = {"id": landmark_id[obj_list[int(i)]], "description": landmark_dict[obj_list[int(i)]],
                               "time": []}
                    iobject["time"].append([count / fps * 1.0, '-', count / fps * 1.0])

                    with open("demo_images/" + obj_list[int(i)] + ".jpg", 'rb') as f:
                        image_byte = base64.b64encode(f.read())
                    image_str = image_byte.decode('ascii')
                    iobject["image"] = image_str

                    Object_Dict['objects'][obj_list[int(i)]] = iobject
                else:
                    time_last = Object_Dict['objects'][obj_list[int(i)]]["time"][-1][-1]
                    if abs(count / fps * 1.0 - time_last) < 0.5:
                        Object_Dict['objects'][obj_list[int(i)]]["time"][-1][-1] = count / fps * 1.0
                    else:
                        Object_Dict['objects'][obj_list[int(i)]]["time"].append(
                            [count / fps * 1.0, '-', count / fps * 1.0])

        # result
        out = invert_affine(framed_metas, out)
        img_show = display(out, ori_imgs, remove_index)

        # show frame by frame # 显示图片

        # cv2.imshow('frame', img_show)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        videoWriter.write(img_show)
    # 去重复
    for key in list(Object_Dict['objects'].keys()):
        for a_time_list in Object_Dict['objects'][key]['time'][:]:
            if a_time_list[2] - a_time_list[0] <= 1.0:
                if len(Object_Dict['objects'][key]['time']) == 1:
                    del Object_Dict['objects'][key]
                    break
                else:
                    Object_Dict['objects'][key]['time'].remove(a_time_list)

    data = json.dumps(Object_Dict, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    cap.release()
    cv2.destroyAllWindows()

    fileObject = open('test_video_object_data.json', 'w')
    fileObject.write(data)
    fileObject.close()
