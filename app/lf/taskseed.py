import json, os, glob
from flask import Flask,request,jsonify
import pymysql.cursors

#  mysql command
#  CREATE TABLE task(id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,title varchar(355),detail varchar(12383),useProcessId varchar(255),status varchar(20));

UPLOAD_IMAGE_FOLDER = "static/images"
UPLOAD_MOVIE_FOLDER = "static/movie"
ALLOWED_EXTENSIONS = set(["png", "jpg", "gif"])

def allwed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def connect_sql():
    return pymysql.connect(host="tomasdb",
                             user="tomas",
                             password="pass",
                             database="leveler",
                             cursorclass=pymysql.cursors.DictCursor)

def seedAction():
    for index in range(40):
        _index = index + 1
        con = connect_sql()
        _id = f"process0{_index}"
        connectId = f"read00{_index}"
        title = f"目的までのプロセス解析00{_index}"
        detail = f"意味の詳細と情報化{_index}"
        # processdata = list(dict(
        #         title=f"アルゴリズムと変数定義{_index-2}",
        #         detail="title",
        #         methodId=f"methodId001",
        #         executionId=f"executionId00{_index-2}",
        #         makeProcess=[]
        #     ),
        #     dict(
        #         title=f"変数定義と個別情報量の補助情報定義{_index-2}",
        #         detail="title",
        #         methodId=f"methodId001",
        #         executionId=f"executionId00{_index-2}",
        #         makeProcess=[]
        #     )
        # )
        processdata = [{
                "title": f"アルゴリズムと変数定義{_index}",
                "detail": "detail",
                "methodId":f"methodId001",
                "executionId":f"executionId00{_index}",
                "makeProcess":[]
            },
            {
                "title":f"変数定義と個別情報量の補助情報定義{_index}",
                "detail":"detail",
                "methodId":f"methodId001",
                "executionId":f"executionId00{_index}",
                "makeProcess":[]
            }
        ]
        print(title)
        try:
            with con as db:
                cursor = db.cursor()
                set_sql = "INSERT INTO process (id,title,detail,processdata,connectId) VALUES (%s, %s, %s, %s, %s)"
                cursor.execute(set_sql,(_id,title,detail,json.dumps(processdata),connectId))
                tasktest = db.commit()
                resultText = "ok"

        finally:
            if cursor:
                cursor.close()

seedAction()