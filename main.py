from flask import Flask, Response, render_template, send_from_directory
import os 

app = Flask(__name__, template_folder=os.path.abspath(""), static_folder=os.path.abspath("static"))

@app.route('/', methods=['GET'])
def home(): 
    return render_template('home.html')

@app.route('/tiny_face_detector_model-weights_manifest.json')
def static_json(): 
    return send_from_directory('static', 'tiny_face_detector_model-weights_manifest.json')

@app.route('/tiny_face_detector_model-shard1')
def static_1():
    return send_from_directory('static', 'tiny_face_detector_model-shard1')

@app.route('/ssd_mobilenetv1_model-weights_manifest.json')
def static_json1(): 
    return send_from_directory('static', 'ssd_mobilenetv1_model-weights_manifest.json')

@app.route('/ssd_mobilenetv1_model-shard1')
def static_2():
    return send_from_directory('static', 'ssd_mobilenetv1_model-shard1')

@app.route('/ssd_mobilenetv1_model-shard2')
def static_3():
    return send_from_directory('static', 'ssd_mobilenetv1_model-shard2')

@app.route('/static/<the_path>')
def send_static(the_path):
    return send_from_directory('static', the_path)

if __name__ == "__main__": 
    app.run(host='127.0.0.1', port=8800, debug=True)

