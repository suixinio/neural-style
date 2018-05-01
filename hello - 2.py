from flask import Flask
from flask import request
import threading
import sqlite3
import urllib.request
import os
import time
from qiniu import Auth, put_file, etag, urlsafe_base64_encode
import smtplib

from email.mime.text import MIMEText

import nu

app = Flask(__name__)

# 需要填写你的 Access Key 和 Secret Key
access_key = '*'
secret_key = '*'
# 构建鉴权对象
q = Auth(access_key, secret_key)
# 要上传的空间
bucket_name = 'nobuges'


def open_url(url):
    response = urllib.request.urlopen(url)
    html = response.read()
    return html


def write_read_sqlite(sql_info):
    conn = sqlite3.connect('test.db')
    c = conn.cursor()
    cursor = c.execute(sql_info)
    tmp = []
    for item in cursor:
        tmp.append(item)
    conn.commit()
    conn.close()
    return tmp


@app.route('/savefile', methods=['POST', 'GET'])
def login():
    print(request.method)
    if request.method == 'POST':
        email_addr = request.form['email']
        file_url = request.form['fileurl']
        print(email_addr, type(email_addr), file_url)
        # 写入数据库
        sql_str = "INSERT INTO pictureinfo(email, w_p_url, state) VALUES ('" \
                  + email_addr + \
                  "','" + str(file_url) + "',0)"
        write_read_sqlite(sql_str)
        print(sql_str)

        # c.execute("INSERT INTO pictureinfo (email,w_p_url,state) \
        #       VALUES (" + email_addr + ",+" + file_url + ",0)");

    # the code below is executed if the request method
    # was GET or the credentials were invalid
    return '{OK}'


def find_need_deal():
    sql_str = "SELECT email,w_p_url,y_p_url,state FROM pictureinfo"
    cursor = write_read_sqlite(sql_str)

    for row in cursor:
        # print('row', row)
        if row[3] == 0:
            yield row


def uploads_qiniu(file_path, filename):
    # 要上传文件的本地路径
    key = str(int(time.time())) + filename
    # 生成上传 Token，可以指定过期时间等
    token = q.upload_token(bucket_name, key, 3600)
    ret, info = put_file(token, key, file_path)
    return 'http://ovjldou7m.bkt.clouddn.com/' + key


def senf_email(item, qiniuurl):
    html = """
    <html>
    <img src = "%s">
    <h1>你好</h1>
    </html>
    """ % qiniuurl

    msg = MIMEText(html, 'html', 'utf-8')
    msg['Subject'] = "Hello"
    msg['From'] = "init16@163.com"
    msg['To'] = item[0]

    s = smtplib.SMTP('smtp.163.com', 25)

    s.login('init16@163.com', 'fu2823787')
    s.sendmail(msg['From'], msg['To'], msg.as_string())
    s.quit()


def update_picture():
    while True:
        for item in find_need_deal():
            print('item', item)
            url = item[1]
            print(url)

            # 获得文件
            img_file = open_url(url)
            with open(url[33:], 'wb') as f:
                f.write(img_file)
            img_path = os.path.join(os.getcwd(), url[33:])
            # 图片处理
            img_deal_file_name = nu.p_main(img_path, os.path.join(os.getcwd(), '1-style.jpg'), 'deal')
            # 上传七牛
            qiniu_url = uploads_qiniu(os.path.join(os.getcwd(), img_deal_file_name), img_deal_file_name)
            # 更新数据库
            sql_str = "UPDATE pictureinfo SET y_p_url = '" + qiniu_url + "' ,state='1' WHERE w_p_url = '" + url + "'"
            write_read_sqlite(sql_str)
            # 发送邮件
            senf_email(item, qiniu_url)


if __name__ == '__main__':
    threading.Thread(target=update_picture, args=()).start()
    app.run(
        host='127.0.0.1',
        port=5001,
    )
