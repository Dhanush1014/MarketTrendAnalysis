import base64
import flask
from flask import Flask,render_template,request
from sklearn.externals import joblib
from scipy import misc
from flask_bootstrap import Bootstrap
import numpy as np
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
app=Flask(__name__)
Bootstrap(app)
@app.route("/")
def index():
	return flask.render_template('index.html')
@app.route('/about')
def about():
	return flask.render_template('about.html')

@app.route('/predict',methods=['POST','GET'])
def make_prediction():
	if request.method=='POST':
		value=request.form['price']
		prediction=model1.predict(int(value))
		if(prediction<0):
			prediction=int(value)
		prediction=int(prediction)
		
		inflation=request.form['inflation']
		option=request.form.get('option1')
		array=[0]*4
		if(option=='Formal'):
			array[0]=1
		elif(option=='Casual'):
			array[1]=1
		elif(option=='Innerwear'):
			array[2]=1
		else:			
			array[3]=1
		a=[array[0],array[1],array[2],value,prediction,inflation]
		inflation1=float(inflation)
		val=int(value)
		value1=int(model2.predict([a]))
		value2=int(model3.predict([[array[0],array[1],array[2],val,prediction,inflation1]]))
		array=[0]*4
	return flask.render_template('result.html',retail=value,result=prediction,result1=value1,result2=value2)
if __name__=='__main__':
	model1=joblib.load('linmodel.pkl')
	model2=joblib.load('predmodel.pkl')
	model3=joblib.load('premodel1.pkl')
	app.run(host='0.0.0.0',port=8000,debug=True)
