# coding=utf-8
import xml.etree.ElementTree as ET
import os
import json
import argparse
import urllib
import yaml

ap = argparse.ArgumentParser()
ap.add_argument('-p', '--project', type=str, default='DiBiao', help='project file that contains parameters')
ap.add_argument('-is_train', '--is_train', type=bool, default=False, help='处理训练集还是验证集，默认先处理验证集')
args = ap.parse_args()

project_name = args.project
is_train = args.is_train

params = yaml.safe_load(open(f'projects/{project_name}.yml'))
voc_clses = params['obj_list']  # 加载类别列表

categories = []

all_result = {}
for iind, cat in enumerate(voc_clses):  # 生成索引
    all_result[cat] = 0  # 最初始全部全0
    cate = {}
    cate['supercategory'] = cat
    cate['name'] = cat
    cate['id'] = iind + 1
    categories.append(cate)


def add_object(xmlname, id):
    sig_xml_box = []
    tree = ET.parse(xmlname)  # 输入是指定的xml文件，包括路径！
    root = tree.getroot()
    images = {}
    for i in root:  # 遍历一级节点
        if i.tag == 'filename':  # 图片名
            file_name = i.text  # 0001.jpg
            # print('image name: ', file_name)
            images['file_name'] = file_name
        if i.tag == 'size':
            for j in i:
                if j.tag == 'width':  # 图片大小
                    width = j.text
                    images['width'] = width
                if j.tag == 'height':
                    height = j.text
                    images['height'] = height
        if i.tag == 'object':  # 目标
            for j in i:
                if j.tag == 'name':
                    cls_name = j.text
                    if cls_name in voc_clses:
                        all_result[cls_name] += 1


def txt2list(txtfile):
    f = open(txtfile)
    l = []
    for line in f:
        l.append(line[:-1])
    print(l)
    f.close()
    return l


def file_name(is_train, path):
    if is_train is False:
        fileout = open('XML_val_name.txt', 'wt')
    else:
        fileout = open('XML_train_name.txt', 'wt')
    F = []
    count = 0
    for root, dirs, files in os.walk(path):
        # print root
        # print dirs
        for file in files:
            count += 1
            # print file.decode('gbk')    #文件名中有中文字符时转码
            if os.path.splitext(file)[1] == '.xml':
                t = os.path.splitext(file)[0]
                print(t)  # 打印所有xml格式的文件名
                fileout.write(t)
                fileout.write('\n')
                F.append(t)  # 将所有的文件名添加到L列表中
    fileout.close()
    print(count)
    return F  # 返回L列表


if __name__ == '__main__':

    train_txt = 'XML_train_name.txt'
    val_txt = 'XML_val_name.txt'

    if is_train is False:
        news2020xmls = f'./datasets/HK_project/XML_files/anns_val'  # 测试xml标注的路径地址
        file_name(is_train, news2020xmls)
        xml_names = txt2list(val_txt)  # 每次切换数据集这里都要改
        json_name = './datasets/HK_project/annotations/instances_val2020.json'
    else:
        news2020xmls = f'./datasets/HK_project/XML_files/anns_train'  # 训练xml标注的路径地址
        file_name(is_train, news2020xmls)
        xml_names = txt2list(train_txt)
        json_name = './datasets/HK_project/annotations/instances_train2020.json'

    xmls = []  # 所有XML文件路径名

    '''这一步到时候要改'''
    for ind, xml_name in enumerate(xml_names):
        xmls.append(os.path.join(news2020xmls, xml_name + '.xml'))
    '''这一步到时候要改'''

    for i_index, xml_file in enumerate(xmls):
        add_object(xml_file, i_index)

    print(all_result)
    pSum = 0
    for key, value in all_result.items():
        pSum += value
    print(pSum)

