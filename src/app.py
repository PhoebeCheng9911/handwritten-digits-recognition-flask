# coding:utf-8
import numpy as np
from flask import Flask, render_template, request
import json

from datetime import timedelta


from mnist_predict import mnist_digit_rec

DEFAULT_TOKEN = "THISISAFUCKINGTOKEN"

app = Flask(__name__,
            template_folder='templates',
            static_folder='static',
            static_url_path='/static')


app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/')
def hello():
    return render_template('canva.html')


@app.route('/digit_rec',methods=["POST"])
def digit_rec():

    return_dict = {'code': '200', 'message': '处理成功', 'result': False}

    print(request.get_json())


    if request.get_data() is None:
        return_dict['code'] = '5002'
        return_dict['message'] = '请求参数为空'
        return json.dumps(return_dict, ensure_ascii=False)


    get_data = request.get_json()

    token = get_data.get("token")
    if token != DEFAULT_TOKEN:
        return_dict['code'] = '5001'
        return_dict['message'] = 'TOKEN错误'
        return json.dumps(return_dict, ensure_ascii=False)

    img_base64 = get_data.get("img")


    result_dict = mnist_digit_rec(img_base64)
    print(result_dict)


    json_encode_result = json.dumps(result_dict, cls=NpEncoder)
    print(json_encode_result)

    return_dict['result'] = json_encode_result

    return_str = json.dumps(return_dict, ensure_ascii=False)
    print(return_str)

    return return_str

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            print(obj.tolist())
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

if __name__ == '__main__':
    # app.debug = True
    app.run(debug=True,host='0.0.0.0')