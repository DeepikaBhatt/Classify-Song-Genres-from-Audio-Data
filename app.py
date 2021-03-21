#!/usr/bin/env python
# coding: utf-8

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

Filename = 'model.pkl'
with open(Filename, 'rb') as file:  
    model = pickle.load(file)

@app.route('/')
def index_page():
    print(model)
    return render_template('index.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict_logic():
    
    if request.method == 'POST':
        acousticness = float(request.form.get('acousticness'))
        danceability = float(request.form.get('danceability'))
        energy = float(request.form.get('energy'))
        instrumentalness = float(request.form.get('instrumentalness'))
        liveness = float(request.form.get('liveness'))
        speechiness = float(request.form.get('speechiness'))                  
        tempo = float(request.form.get('tempo'))
        valence = float(request.form.get('valence'))

    predicted_name = model.predict([[acousticness,danceability,energy,instrumentalness,liveness,speechiness,tempo,valence]]).tolist()[0]

    return render_template('index.html', predicted_name=predicted_name)


if __name__ == "__main__":
    app.run()







