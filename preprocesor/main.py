from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import numpy as np
from processing import ImageEmbedder, TFVectorizer, neighbours
from db import coll
from PIL import Image
from io import BytesIO

app = Flask(__name__)
cors = CORS(app)
embedder = ImageEmbedder()
tfer = TFVectorizer()

backend_url = "http://app_worker:9000/find"
RESULTS_QUANTITY = 10

def query_img(img):
    vec = embedder.get_vec([img])[0]
    tf_str = tfer.get_tf_string(vec)
    nb = neighbours(tf_str)

    data = {
        "vector": vec.tolist(),
        "neighbours": nb,
        "quantity": RESULTS_QUANTITY
    }

    r = requests.post(backend_url, json=data)

    vectors = r.json()["vectors"]
    ids = list(set([x["media_id"] for x in vectors]))

    media = {}
    for m in coll.find({"_id": {"$in": ids}}):
        media[m["_id"]] = m

    for v in vectors:
        v["name"] = media[v["media_id"]]["name"]
        v["url"] = media[v["media_id"]]["url"]

    return vectors


@app.route('/find', methods=['POST'])
def find():
    f = request.files['file']
    if f:
        f = f.read()
    else:
        return jsonify(isError=True, message="bad request", statusCode=400)
    npimg = np.array(Image.open(BytesIO(f)))
    vectors = query_img(npimg)
    return jsonify(isError=False, message=vectors, statusCode=200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
