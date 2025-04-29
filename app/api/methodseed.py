import json, os, glob
from flask import Flask,request,jsonify
import pymysql.cursors

#  mysql command
#  CREATE TABLE method (id varchar(255),methodId varchar(255),title varchar(355),detail text,tagger varchar(255),structure varchar(7383));

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
        _index = index + 6
        con = connect_sql()
        _id = _index
        methodId = f"計画ポイント解析00{_index-2}"
        title = f"計画ポイント解析00{_index-2}"
        detail = f"意味の詳細と情報化{_index-2}"
        structure = f"フェーズとカテゴライズ化{_index-2}"
        tagger = "解析,分析情報化"
        adjustmentNumbers = [0.3]
        print(title)
        try:
            with con as db:
                cursor = db.cursor()
                set_sql = "INSERT INTO method (id, methodId, title, detail, structure,tagger,adjustmentNumbers) VALUES (%s, %s, %s, %s, %s, %s, %s)"
                cursor.execute(set_sql,(_id,methodId,title,detail,structure,tagger,adjustmentNumbers))
                tasktest = db.commit()
                resultText = "ok"
        finally:
            if cursor:
              cursor.close()

seedAction()