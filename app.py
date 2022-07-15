# Import essential libraries
from email import message
from fileinput import filename
from django.shortcuts import render
from flask import Flask, appcontext_popped,render_template,request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object 
filename='movie-genre-mnb-model.pkl'
classifier=pickle.load(open(filename,'rb'))
cv=pickle.load(open('cv-transformer.pkl','rb'))

app=Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        my_prediction=classifier.predict(vect)
        return render_template('result.html',prediction=my_prediction)

if __name__=='__main__':
    app.run(debug=True)