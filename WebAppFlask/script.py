"""
Developed by Aindriya Barua in April, 2019
Code for paper https://www.researchgate.net/publication/349190662_Analysis_of_Contextual_and_Non-contextual_Word_Embedding_Models_for_Hindi_NER_with_Web_Application_for_Data_Collection
This project builds a web application where the user can enter Hindi tect and gets the NER tag predictions
The user is then requested for feedback, where they can annotate the words again and give feedback if any predictions were wrog, this is used for further data collection.

If you use any part of the resources provided in this repo, kindly cite:
Barua, A., Thara, S., Premjith, B. and Soman, K.P., 2020, December. Analysis of Contextual and Non-contextual Word Embedding Models for Hindi NER with Web Application for Data Collection. In International Advanced Computing Conference (pp. 183-202). Springer, Singapore.
"""

import os
import flask
import pickle
import re
from flask import Flask, render_template, request
words = []
#creating instance of the class
app=Flask(__name__)
#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
#@app.route('/result')

def index():
    return flask.render_template('index.html')

#prediction function
def get_result(input):

   try:
    
    loaded_wv_model = pickle.load(open("wvmodel.pkl","rb"))
    word_vec = loaded_wv_model.wv[input]
    loaded_clf_model = pickle.load(open("model.pkl","rb"))
    pred_label = loaded_clf_model.predict(word_vec.reshape(1, -1))
    labels = ['datenum', 'event', 'location', 'name', 'number', 'occupation',
       'organization', 'other', 'things']
    result = labels[pred_label[0]]
    return (result)
   except:
    return ("other")


@app.route('/result',methods = ['POST'])
def result():
  if request.method == 'POST':
    para = request.form['text']
    para = re.sub(' +', ' ', para)
    global words
    words = para.split(' ')
    print(">>>>>>>>>>>>>>>>>>>>")
    print(type(words))
    result=[]
    
    for i in words:
       res = get_result(i)
       result.append(res)
    
    print(result)
    labels=['datenum', 'event', 'location', 'name', 'number', 'occupation',
       'organization', 'other', 'things']
    return render_template("result.html", words = words, result = result, labels = labels, len = len(result), datalist = [])


@app.route('/feedback', methods = ['GET', 'POST'])
def feedback():
    # POST request
    if request.method == 'POST':
        res = request.get_json()  
        print("======================================================================")
        print(str(res))
        file1 = open("datacollection.txt","a", encoding='utf-8')
        count = 0
        for i in res:
          global words
          file1.write(str(words[count]))
          file1.write(" : ")
          file1.write(str(i))
          file1.write(",")
          count = count+1
        file1.write("\n")
        file1.close()

        return 'OK', 200

    # GET request
    else:
        message = {'greeting':'Get request'}
        return jsonify(message)  # serialize and use JSON headers

if __name__ == '__main__':
   app.run(debug = True)
