import numpy as np
import base64
import json

import os
import requests
import subprocess

from flask import Flask, jsonify,Response, render_template
from flask_restful import Api, Resource, reqparse
from flask_cors import CORS
from werkzeug.datastructures import FileStorage

import efficientdet_test_videos as vivi
import efficientdet_test as pipi

# if __name__ == "__main__":
#     # pipi.run2('080.jpg', '080')
#     vivi.run2('4.mp4', '4')
#     # vivi.run2('6.mp4', '6')


# run函数是地标检测
# run2函数是国徽国旗检测

def create_app():
    app = Flask(__name__)
    api = Api(app)

    # 添加接口路由
    api.add_resource(DiBiao_Video, "/dibiao_video", endpoint="dibiao_video")
    api.add_resource(DiBiao_Picture, "/dibiao_photo", endpoint="dibiao_photo")
    api.add_resource(GuoQi_Video, "/guoqi_video", endpoint="guoqi_video")
    api.add_resource(GuoQi_Picture, "/guoqi_photo", endpoint="guoqi_photo")

    return app


class DiBiao_Video(Resource):  # 视频地标检测
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("video", type=FileStorage, required=True, location="files")
    @classmethod
    def main_func(cls, filename):
        name = filename[:filename.find(".")]
        res = vivi.run(filename, name)  # 地标检测的run函数
        return res

    def post(self):
        args = self.parser.parse_args()
        print('haha')
        video = args["video"]
        video.save(f"video/{video.filename}")
        res = self.main_func(video.filename)

        return jsonify(res)


class GuoQi_Video(Resource):   # 视频国旗国徽检测
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("video", type=FileStorage, required=True, location="files")
    @classmethod
    def main_func(cls, filename):
        name = filename[:filename.find(".")]
        res = vivi.run2(filename, name)  # 国旗国徽检测的run2函数
        return res

    def post(self):
        args = self.parser.parse_args()
        print('xixi')
        video = args["video"]
        video.save(f"video/{video.filename}")
        res = self.main_func(video.filename)

        return jsonify(res)


class DiBiao_Picture(Resource):  # 图片地标检测
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("picture", type=FileStorage, required=True, location="files")
    @classmethod
    def main_func(cls, filename):
        res = pipi.run(filename, filename)  # 地标检测的run函数
        return res

    def post(self):
        args = self.parser.parse_args()
        picture = args["picture"]
        picture.save(f"input/{picture.filename}")
        res = self.main_func(picture.filename)

        return jsonify(res)


class GuoQi_Picture(Resource):  # 图片国旗国徽检测
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument("picture", type=FileStorage, required=True, location="files")
    @classmethod
    def main_func(cls, filename):
        res = pipi.run2(filename, filename)  # 国旗国徽检测的run2函数
        return res

    def post(self):
        args = self.parser.parse_args()
        picture = args["picture"]
        picture.save(f"input/{picture.filename}")
        res = self.main_func(picture.filename)

        return jsonify(res)


if __name__ == "__main__":
    app = create_app()
    CORS(app)
    app.run(host="0.0.0.0", port=4002, debug=False)
