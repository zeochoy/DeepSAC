import os
#import glob
from flask import Flask
from flask import jsonify, Response, request, render_template

from sacapp import app
from model.utils import *
from model.deepsacmodel import *

#valid_mimetypes = ['text/csv', 'text/txt']
cell_lines = read_list(app.config['CELL_LINES_FILE'], as_int=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Run Prediction on the model
        cell_line = request.form['sacinput']
        if cell_line not in cell_lines:
            return jsonify({'error': 'Cell line cannot be found.'})
        res = get_deepsac_prediction(cell_line)
        return jsonify(res)

@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(cell_lines), mimetype='application/json')
