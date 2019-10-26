# 这是一个非常简单的使用Web服务上传图片运行人脸识别的案例，后端服务器会识别这张图片是不是奥巴马，并把识别结果以json键值对输出
# 比如：运行以下代码
# $ curl -XPOST -F "file=@obama2.jpg" http://127.0.0.1:5001
# 会返回：
# {
#  "face_found_in_image": true,
#  "is_picture_of_obama": true
# }
#
# 本项目基于Flask框架的案例 http://flask.pocoo.org/docs/0.12/patterns/fileuploads/

# 提示：运行本案例需要安装Flask，你可以用下面的代码安装Flask
# $ pip3 install flask

import face_recognition
from flask import Flask, jsonify, request, redirect
import os
import pickle

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
face_hibernate_file = 'face.pt'
face_picture_path = 'face/pic/'

# if os.path.exists(face_hibernate_file):
#     os.remove(face_hibernate_file)


def get_all_know_face():
    # 0. torch读取缓存文件
    # 1. 遍历文件夹
    # 2. img = face_recognition.load_image_file(file_stream)
    if os.path.exists(face_hibernate_file):
        face = pickle.load(open(face_hibernate_file, "rb"))
        return face['face_know'], face['id_nums']
    face_know = []
    id_nums = []
    for root, dirs, fs in os.walk(face_picture_path):
        for f in fs:
            face_know.append(
                face_recognition.face_encodings(face_recognition.load_image_file(os.path.join(root, f)))[0])
            id_nums.append(f)
    pickle.dump({'face_know': face_know, 'id_nums': id_nums}, open(face_hibernate_file, "wb"))
    return face_know, id_nums


face_encoding_tmp = get_all_know_face()

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/tinker')
def entry():
    return jsonify({'tinker': 'tinker'})


@app.route('/recognition', methods=['POST'])
def recognition():
    # 检测图片是否上传成功
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # 图片上传成功，检测图片中的人脸
            return detect_faces_in_image(file)

    # 图片上传失败，输出以下html代码
    return jsonify({'code': 1, 'msg': 'upload error'})


# 注册人脸，更新人脸缓存
def register_face(file_stream, filename):
    img = face_recognition.load_image_file(file_stream)
    regist_face_encodings = face_recognition.face_encodings(img)
    if len(regist_face_encodings) < 1:
        return jsonify({'code': 1, 'msg': 'can not recognize any faces !'})

    global face_encoding_tmp
    face_know, id_nums = face_encoding_tmp
    face_know.append(regist_face_encodings[0])
    id_nums.append(filename)
    face_dump = {'face_know': face_know, 'id_nums': id_nums}
    pickle.dump(face_dump, open(face_hibernate_file, "wb"))
    face_encoding_tmp = get_all_know_face()
    # 保存图片
    result = {
        'code': 0,
        'id_num': filename,
        'msg': 'success',
    }
    return jsonify(result)


@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # 图片上传成功，检测图片中的人脸
            return register_face(file, file.filename)

    # 图片上传失败，输出以下html代码
    return jsonify({'code': 1, 'msg': 'upload error'})


# 返回列表中为 TRUE 的index
def get_face_rec(match_results):
    for i in range(len(match_results)):
        if match_results[i]:
            return i
    return -1


def detect_faces_in_image(file_stream):
    known_face_encoding_list, id_nums = face_encoding_tmp

    # 载入用户上传的图片
    img = face_recognition.load_image_file(file_stream)
    # 为用户上传的图片中的人脸编码
    unknown_face_encodings = face_recognition.face_encodings(img)

    if len(unknown_face_encodings) < 1:
        return jsonify({'code': 1, 'msg': 'can not recognize any faces !'})
    match_results = face_recognition.compare_faces(known_face_encoding_list, unknown_face_encodings[0])

    face_index = get_face_rec(match_results)
    if face_index == -1:
        return jsonify({'code': 1, 'msg': 'Match nothing !'})

    # 讲识别结果以json键值对的数据结构输出
    result = {
        'code': 0,
        'id_num': id_nums[face_index],
        'msg': 'success',
    }
    return jsonify(result)


def main():
    app.run(host='0.0.0.0', port=5001, debug=True)


if __name__ == "__main__":
    main()
    # face_test_encoding = face_recognition.face_encodings(face_recognition.load_image_file('WechatIMG743.jpeg'))[0]
    # face_know, id_nums = get_all_know_face()
    # result = face_recognition.compare_faces(face_know, face_test_encoding)
    # index = get_face_rec(result)
    # print("身份证号：", id_nums[index])
