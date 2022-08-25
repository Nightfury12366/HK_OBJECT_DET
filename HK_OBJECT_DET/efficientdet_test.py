# Author: Zylo117

"""
Simple Inference Script of EfficientDet-Pytorch
"""
import numpy as np
import argparse
import time
import torch
from torch.backends import cudnn
from matplotlib import colors

from backbone import EfficientDetBackbone
import cv2

import os
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, \
    plot_one_box
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
from PIL import Image, ImageDraw, ImageFont
import json
import yaml

color_list = standard_to_bgr(STANDARD_COLORS)


def get_args():
    parser = argparse.ArgumentParser('SkyLake HK_Object_detection Image Test')
    parser.add_argument('--data_path', type=str, default='input/1.jpg', help='输入图片路径')
    parser.add_argument('--is_dibiao', type=bool, default=False, help='检测国旗还是地标')  # 默认检测国旗国徽
    args = parser.parse_args()
    return args


# 显示模块
def display(preds, imgs, imshow=False, imwrite=True, savename=None, obj_list=None, remove_list=None):
    for i in range(len(imgs)):
        if len(preds[i]['rois']) == 0:  # 没有要检测的物体，保存原图然后跳过
            cv2.imwrite(f'results/{savename}.jpg', imgs[i])
            continue

        imgs[i] = imgs[i].copy()

        for j in range(len(preds[i]['rois'])):
            if j in remove_list:  # 如果是不需要的物体，就跳过它
                continue
            x1, y1, x2, y2 = preds[i]['rois'][j].astype(np.int)
            obj = obj_list[preds[i]['class_ids'][j]]
            score = float(preds[i]['scores'][j])

            plot_one_box(imgs[i], [x1, y1, x2, y2], label=obj, score=score,
                         color=color_list[get_index_label(obj, obj_list)])
        # imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)

        if imshow:
            cv2.imshow('img', imgs[i])
            cv2.waitKey(0)

        if imwrite:  # 保存文件
            cv2.imwrite(f'results/{savename}.jpg', imgs[i])


# 地标图片检测run函数，改好了
def run(filename, savename):
    compound_coef = 0
    force_input_size = None  # set None to use default size
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.85  # 0.85以上才认为是
    iou_threshold = 0.05

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    params = yaml.safe_load(open(f'projects/display_DiBiao.yml'))  # 加载地标配置文件

    obj_list = params['obj_list']

    landmark_id = params['landmark_id']
    landmark_dict = params['landmark_dict']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # Image's path
    img_path = str('input/' + filename)
    net_path = f'weights/efficientdet-d0_dibiao.pth'

    if os.path.isdir(img_path):
        path_list = os.listdir(img_path)
        for i in range(len(path_list)):
            path_list[i] = os.path.join(img_path, path_list[i])
        path_list = [path_list[0]]
        print(path_list)
    else:
        path_list = [img_path]

    Object_Dict = {"objects": {}}

    for img in path_list:
        ori_imgs, framed_imgs, framed_metas = preprocess(img, mean=[0.4883, 0.4422, 0.4551], std=[0.2602, 0.2623, 0.2685], max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)

        model.load_state_dict(torch.load(net_path, map_location='cpu'))
        # model.load_state_dict(torch.load(f'weights/efficientdet-d5_HK.pth', map_location='cpu'))

        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

            # 添加带有时间戳的对象
            remove_index = []
            if len(out[0]['class_ids']) > 0:
                for index in range(len(out[0]['class_ids'])):
                    i = out[0]['class_ids'][index]
                    if obj_list[int(i)] == 'fyb':  # 剔除负样本
                        remove_index.append(index)
                        continue
                    x1, y1, x2, y2 = out[0]['rois'][index]
                    if obj_list[int(i)] not in Object_Dict['objects'].keys():  # 这个目标不在字典之中，是新目标

                        iobject = {"id": landmark_id[obj_list[int(i)]], "description": landmark_dict[obj_list[int(i)]],
                                   "locate": []}
                        iobject["locate"].append([int(x1), int(y1), int(x2), int(y2)])
                        Object_Dict['objects'][obj_list[int(i)]] = iobject
                    else:
                        Object_Dict['objects'][obj_list[int(i)]]["locate"].append([int(x1), int(y1), int(x2), int(y2)])

        out = invert_affine(framed_metas, out)

        display(out, ori_imgs, imshow=False, imwrite=True, savename=savename, obj_list=obj_list,
                remove_list=remove_index)

    data = json.dumps(Object_Dict, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    fileObject = open('test_image_object_data.json', 'w')
    fileObject.write(data)
    fileObject.close()
    return Object_Dict


# 国旗国徽检测run2函数，改好了
def run2(filename, savename):
    compound_coef = 0
    force_input_size = None  # set None to use default size
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.8
    iou_threshold = 0.05

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True

    params = yaml.safe_load(open(f'projects/display_GuoQi.yml'))  # 加载国旗配置文件

    obj_list = params['obj_list']

    landmark_id = params['landmark_id']
    landmark_dict = params['landmark_dict']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # Image's path
    img_path = str('input/' + filename)
    net_path = f'weights/efficientdet-d0_guoqi.pth'

    if os.path.isdir(img_path):
        path_list = os.listdir(img_path)
        for i in range(len(path_list)):
            path_list[i] = os.path.join(img_path, path_list[i])
        path_list = [path_list[0]]
        print(path_list)
    else:
        path_list = [img_path]

    Object_Dict = {"objects": {}}

    for img in path_list:
        ori_imgs, framed_imgs, framed_metas = preprocess(img, mean=[0.4883, 0.4422, 0.4551], std=[0.2602, 0.2623, 0.2685], max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)

        model.load_state_dict(torch.load(net_path, map_location='cpu'))

        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

            # 添加带有时间戳的对象
            remove_index = []  # 要在显示中剔除的目标框
            if len(out[0]['class_ids']) > 0:
                for index in range(len(out[0]['class_ids'])):
                    i = out[0]['class_ids'][index]
                    if obj_list[int(i)] == 'fyb':  # 剔除负样本
                        remove_index.append(index)
                        continue
                    x1, y1, x2, y2 = out[0]['rois'][index]
                    if obj_list[int(i)] not in Object_Dict['objects'].keys():  # 这个目标不在字典之中，是新目标

                        iobject = {"id": landmark_id[obj_list[int(i)]], "description": landmark_dict[obj_list[int(i)]],
                                   "locate": []}
                        iobject["locate"].append([int(x1), int(y1), int(x2), int(y2)])
                        Object_Dict['objects'][obj_list[int(i)]] = iobject
                    else:
                        Object_Dict['objects'][obj_list[int(i)]]["locate"].append([int(x1), int(y1), int(x2), int(y2)])

        out = invert_affine(framed_metas, out)

        display(out, ori_imgs, imshow=False, imwrite=True, savename=savename, obj_list=obj_list,
                remove_list=remove_index)

    data = json.dumps(Object_Dict, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    fileObject = open('test_image_object_data.json', 'w')
    fileObject.write(data)
    fileObject.close()
    return Object_Dict


#  主函数，改好了
if __name__ == '__main__':

    compound_coef = 0
    force_input_size = None  # set None to use default size
    # replace this part with your project's anchor config
    anchor_ratios = [(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]
    anchor_scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

    threshold = 0.8
    iou_threshold = 0.05

    use_cuda = True
    use_float16 = False
    cudnn.fastest = True
    cudnn.benchmark = True
    opt = get_args()

    if opt.is_dibiao is False:  # 判断加载地标模型还是国旗国徽模型
        params = yaml.safe_load(open(f'projects/display_GuoQi.yml'))
        net_path = f'weights/efficientdet-d0_guoqi.pth'
        # compound_coef = 0
    else:
        params = yaml.safe_load(open(f'projects/display_DiBiao.yml'))
        net_path = f'weights/efficientdet-d0_dibiao.pth'

    obj_list = params['obj_list']
    landmark_id = params['landmark_id']
    landmark_dict = params['landmark_dict']

    # tf bilinear interpolation is different from any other's, just make do
    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]
    input_size = input_sizes[compound_coef] if force_input_size is None else force_input_size

    # Image's path
    img_path = opt.data_path  # set int to use webcam, set str to read from a video file

    if os.path.isdir(img_path):
        path_list = os.listdir(img_path)
        for i in range(len(path_list)):
            path_list[i] = os.path.join(img_path, path_list[i])
        print(path_list)
    else:
        path_list = [img_path]

    Object_Dict_All = {}

    for img in path_list:
        ori_imgs, framed_imgs, framed_metas = preprocess(img, mean=[0.4883, 0.4422, 0.4551], std=[0.2602, 0.2623, 0.2685], max_size=input_size)
        if use_cuda:
            x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
        else:
            x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

        x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)

        model.load_state_dict(torch.load(net_path, map_location='cpu'))

        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with torch.no_grad():
            features, regression, classification, anchors = model(x)
            regressBoxes = BBoxTransform()
            clipBoxes = ClipBoxes()

            out = postprocess(x,
                              anchors, regression, classification,
                              regressBoxes, clipBoxes,
                              threshold, iou_threshold)

            # 添加带有时间戳的对象
            Object_Dict = {"objects": {}}
            remove_index = []  # 要在显示中剔除的目标框
            if len(out[0]['class_ids']) > 0:
                for index in range(len(out[0]['class_ids'])):
                    i = out[0]['class_ids'][index]
                    if obj_list[int(i)] == 'fyb':  # 剔除负样本
                        remove_index.append(index)
                        continue

                    x1, y1, x2, y2 = out[0]['rois'][index]

                    if obj_list[int(i)] not in Object_Dict['objects'].keys():  # 这个目标不在字典之中，是新目标

                        iobject = {"id": landmark_id[obj_list[int(i)]], "description": landmark_dict[obj_list[int(i)]],
                                   "locate": []}
                        iobject["locate"].append([int(x1), int(y1), int(x2), int(y2)])
                        Object_Dict['objects'][obj_list[int(i)]] = iobject
                    else:
                        Object_Dict['objects'][obj_list[int(i)]]["locate"].append([int(x1), int(y1), int(x2), int(y2)])

            Object_Dict_All[img.split('/')[-1]] = Object_Dict

        out = invert_affine(framed_metas, out)
        display(out, ori_imgs, imshow=False, imwrite=True, savename=img.split('/')[-1].split('.')[0], obj_list=obj_list,
                remove_list=remove_index)

    data = json.dumps(Object_Dict_All, sort_keys=True, indent=4, ensure_ascii=False, separators=(',', ': '))
    print(data)

    fileObject = open('test_image_object_data.json', 'w')
    fileObject.write(data)
    fileObject.close()
