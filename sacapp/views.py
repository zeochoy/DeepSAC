import os
#import glob
from flask import Flask
from flask import jsonify, Response, request, render_template

from sacapp import app
from model.utils import *
from model.deepsacmodel import *

#valid_mimetypes = ['text/csv', 'text/txt']
cell_lines = read_list(app.config['CELL_LINES_FILE'], as_int=False)
pert_drugids = read_list(app.config['PERT_DRUGS_FILE'], as_int=True)
dr_drugids = read_list(app.config['IC50_DRUGS_FILE'], as_int=True)
drug2id = read_dict(app.config['DRUG2ID_FILE'], as_int=False)
drug2id = {k:int(v) for k,v in drug2id.items()}
id2drug = {v:k for k,v in drug2id.items()}
pert_drugs = [id2drug[i] for i in pert_drugids]
dr_drugs = [id2drug[i] for i in dr_drugids]

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

@app.route('/drexplorer', methods=['POST'])
def drexplorer():
    if request.method == 'POST':
        # Run Prediction on the model
        drug = request.form['drinput']
        if drug not in dr_drugs:
            return jsonify({'error': 'Drug cannot be found.'})
        res = get_dr_result(drug)
        return jsonify(res)

@app.route('/pertpredict', methods=['POST'])
def pertpredict():
    if request.method == 'POST':
        # Run Prediction on the model
        cl = request.form['pertclinput']
        pert = request.form['pertinput']
        drug = request.form['pertdruginput']
        if cl not in cell_lines:
            return jsonify({'error': 'Cell line cannot be found.'})
        if pert not in pert_drugs:
            return jsonify({'error': 'Potentiator cannot be found.'})
        if drug not in dr_drugs:
            return jsonify({'error': 'Drug cannot be found.'})
        if drug is None:
            return jsonify({'error': 'Drug must not be empty.'})
        res = get_pert_prediction(cl, drug2id[pert], drug2id[drug])
        return jsonify(res)

@app.route('/_autocomplete', methods=['GET'])
def autocomplete():
    return Response(json.dumps(cell_lines), mimetype='application/json')

@app.route('/_pert-cl-autocomplete', methods=['GET'])
def pert_cl_autocomplete():
    return Response(json.dumps(cell_lines), mimetype='application/json')

@app.route('/_pert-autocomplete', methods=['GET'])
def pert_autocomplete():
    return Response(json.dumps(pert_drugs), mimetype='application/json')

@app.route('/_pert-drug-autocomplete', methods=['GET'])
def pert_drug_autocomplete():
    return Response(json.dumps(dr_drugs), mimetype='application/json')

@app.route('/_dr-autocomplete', methods=['GET'])
def dr_autocomplete():
    return Response(json.dumps(dr_drugs), mimetype='application/json')
