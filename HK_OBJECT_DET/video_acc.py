import json
import os.path


def callAcc(pre, tag):
    # [a-b]
    sorted(tag, key=lambda x: x[0])  # 根据起始时间排序
    sorted(pre, key=lambda x: x[0])
    ppos = tpos = 0
    totU = 0
    totI = 0
    # temp = 0
    while ppos < len(pre) and tpos < len(tag):
        a = pre[ppos]
        b = tag[tpos]
        temp = IoU2D(a, b)
        if temp == 0:
            ppos += 1
        elif temp == -1:
            tpos += 1
            totU += (b[2] - b[0] + 1)
        else:
            totI += temp
            if a[2] <= b[2]:
                ppos += 1
            if a[2] >= b[2]:
                tpos += 1
                totU += (b[2] - b[0] + 1)
    return totI / totU, totI, totU


def keyFrameAcc(pre_file, tag_file, out_file=None):
    with open(tag_file, 'r', encoding='utf-8') as fp:
        tag_map = json.load(fp)
    with open(pre_file, 'r', encoding='utf-8') as fp:
        pre_map = json.load(fp)
    result = {}
    totol_I = 0
    totol_U = 0
    for video_k, video_v in pre_map.items():
        video_map = {}
        if video_k in tag_map:  # 寻找对应视频
            tag_video = tag_map[video_k]
            for character in video_v:
                if character in tag_video:
                    acc, v_I, v_U = callAcc(video_v[character], tag_video[character])
                    # print(video_k,character,acc)
                    totol_I += v_I
                    totol_U += v_U
                    video_map[character] = acc
        result[video_k] = video_map
    print(result)
    print("Average Accuracy: {:.4%}".format(totol_I / totol_U))
    if out_file:
        try:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4, ensure_ascii=False)
            print("Json file saved: {:s}".format(out_file))
        except:
            print("Fail to write file: {:s} !".format(out_file))
    else:
        print("No outfile!")

    # print(type(mp))
    # print(mp)


def IoU2D(a, b):
    if a[2] < b[0]:  # [a][b]
        return 0
    elif a[0] > b[2]:  # [b][a]
        return -1
    else:
        return min(a[2], b[2]) - max(a[0], b[0]) + 1


keyFrameAcc(tag_file="performance_test_guoqi.json", pre_file="performance_test_guoqi.json", out_file="result.json")
# print(IoU2D([1,3],[2,4]))
