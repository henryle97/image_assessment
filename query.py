from random import choice
from flask import Flask, jsonify, request
import time, requests
from PIL import Image
from io import BytesIO
import numpy as np
from Thread import MyThread
from image_assessment import ImageAssessment
import copy
model_aesthetic = ImageAssessment(weights_file="models/MobileNet/weights_mobilenet_aesthetic_0.07.hdf5")
model_technical = ImageAssessment(weights_file="models/MobileNet/weights_mobilenet_technical_0.11.hdf5")

model = [[model_aesthetic, "aesthetic"], [model_technical, "technical"]]

desktop_agents = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/602.2.14 (KHTML, like Gecko) Version/10.0.1 Safari/602.2.14',
    'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.98 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.71 Safari/537.36',
    'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0']


def random_headers():
    return {'User-Agent': choice(desktop_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'}


def download_image(image_url):
    header = random_headers()

    response = requests.get(image_url, headers=header, stream=True, verify=False, timeout=5)

    image = Image.open(BytesIO(response.content)).convert('RGB')

    return image


def jsonify_str(output_list):
    with app.app_context():
        with app.test_request_context():
            result = jsonify(output_list)
    return result


app = Flask(__name__)


def create_query_result(input_url, results, error=None):
    if error is not None:
        results = 'Error: ' + str(error)
    query_result = {
        'results': results
    }
    return query_result

@app.route("/query", methods=['GET', 'POST'])
def query():
    try:
        image_url = request.args.get('url', default='', type=str)
        img = download_image(image_url)
        img = img.convert('RGB')
    except Exception as ex:
        return jsonify_str(create_query_result("", "", ex))

    img_arr = np.array(img.resize((224,224)))

    t1 = time.time()
    threads = [MyThread(model[i][0], copy.deepcopy(img_arr), model[i][1]) for i in range(len(model))]
    for thread in threads:
        thread.start()
    results = []
    for thread in threads:
        rs = thread.join()
        results.append(rs)

    result_json = {results[0][1]: results[0][0], results[1][1]: results[1][0]}

    print(f"time={time.time() - t1}")
    return jsonify_str(result_json)

@app.route("/query", methods=['GET', 'POST'])
def query():
    try:
        image_url = request.args.get('url', default='', type=str)
        img = download_image(image_url)
        img = img.convert('RGB')
    except Exception as ex:
        return jsonify_str(create_query_result("", "", ex))

    img_arr = np.array(img.resize((224,224)))

    t1 = time.time()
    threads = [MyThread(model[i][0], copy.deepcopy(img_arr), model[i][1]) for i in range(len(model))]
    for thread in threads:
        thread.start()
    results = []
    for thread in threads:
        rs = thread.join()
        results.append(rs)

    result_json = {results[0][1]: results[0][0], results[1][1]: results[1][0]}

    print(f"time={time.time() - t1}")
    return jsonify_str(result_json)


app.run("localhost", 1904, threaded=True, debug=True)