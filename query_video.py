from random import choice
from flask import Flask, jsonify, request, send_file
import time, requests
from PIL import Image
from io import BytesIO
import numpy as np
from Thread_video import MyThread
from image_assessment import ImageAssessment
import copy
import utils_video

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


def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')

@app.route("/query", methods=['GET', 'POST'])
def query():
    try:
        video_url = request.args.get('url', default='', type=str)
        frames = utils_video.process(video_url)
    except Exception as ex:
        return jsonify_str(create_query_result("", "", ex))

    t1 = time.time()
    threads = [MyThread(model[i][0], copy.deepcopy(frames), model[i][1]) for i in range(len(model))]
    for thread in threads:
        thread.start()
    results = {}
    for thread in threads:
        rs = thread.join()
        results[rs[1]] = rs[0]
    print(results)
    total_scores = (results["aesthetic"]*6 + results["technical"]*4)/10.0
    print(total_scores)
    index_max = np.argmax(total_scores)
    print(index_max)
    img_pil = Image.fromarray(frames[index_max])
    print(f"Total time: {time.time() - t1}")
    return serve_pil_image(img_pil)


app.run("localhost", 1905, threaded=True, debug=True)
