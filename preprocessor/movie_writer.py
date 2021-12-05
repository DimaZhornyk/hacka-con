import cv2
import requests
import os
from processing import ImageEmbedder, TFVectorizer
import sys
from db import coll

backend_url = "http://app_worker:9000/write"
no_image = "https://scontent.fiev6-1.fna.fbcdn.net/v/t1.18169-9/1525596_639252369449625_2033844048_n.png?_nc_cat=101&ccb=1-5&_nc_sid=09cbfe&_nc_ohc=tG-P7WthWYwAX_6czxw&_nc_ht=scontent.fiev6-1.fna&oh=212ccdcefbfe41eddd042419af7d2024&oe=61D0784B"


def getFileLength(path):
    with open(path, "r") as file:
        nonempty_lines = [line.strip("\n") for line in file if line != "\n"]

    return len(nonempty_lines)


def pushWrite(embedding, tfs, identifier):
    data = {
        "vector": embedding.tolist(),
        "mediaId": identifier,
        "segment": tfs
    }
    r = requests.post(backend_url, json=data)
    if r.status_code != 204:
        print(r.text)


def extractImages(path_in):
    _, filename = os.path.split(path_in)
    cnt = coll.count_documents({}) + 1
    coll.insert_one({"_id": cnt, "name": filename, "url": no_image})
    count = 0
    vidcap = cv2.VideoCapture(path_in)
    success = True
    total_time_processed = 0
    embedder = ImageEmbedder()
    tfer = TFVectorizer()
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC, (count * 500))

        success, image = vidcap.read()
        if not success:
            break

        embedding = embedder.get_vec([image])[0]
        tf = tfer.get_tf_string(embedding)

        pushWrite(embedding, tf, cnt)

        total_time_processed += 0.5
        if total_time_processed % 10 == 0:
            print(total_time_processed)
        count += 1


extractImages(sys.argv[1])
