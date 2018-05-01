# -*- coding: utf-8 -*-
import os
from flask import Flask, request
from flask_uploads import UploadSet, configure_uploads, IMAGES, \
    patch_request_class

from qiniu import Auth, put_file, etag, urlsafe_base64_encode
import qiniu.config
import threading
import time

import requests



# 需要填写你的 Access Key 和 Secret Key
access_key = '*'
secret_key = '*'
# 构建鉴权对象
q = Auth(access_key, secret_key)
# 要上传的空间
bucket_name = 'nobuges'

app = Flask(__name__)
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd()  # 文件储存地址

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)  # 文件大小限制，默认为16MB

html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         邮箱:
         <input type="text" name="email"><br>
         <input type=file name=photo>
         <input type=submit value=上传>
    </form>
    '''


def uploads_qiniu(file, filename, email):
    print(1, email)
    # 要上传文件的本地路径
    key = str(int(time.time())) + filename
    # localfile = str(int(time.time())) + str(request.files['photo'].filename)
    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key, 3600)
    ret, info = put_file(token, key, file)
    print(r'http://ovjldou7m.bkt.clouddn.com/' + key)
    post_url(email, 'http://ovjldou7m.bkt.clouddn.com/' + key)
    print(2)


def post_url(email_addr, file_url):
    postdata = {'email': email_addr, 'fileurl': file_url}
    r = requests.post("http://119.29.22.158/savefile", data=postdata)
    print(r.text)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        email_addr = request.form['email']
        file_url = photos.url(filename)
        threading.Thread(target=uploads_qiniu, args=(filename, request.files['photo'].filename, email_addr)).start()

        return html + '<br><img src=' + file_url + '>'
    return html


if __name__ == '__main__':
    app.run()
