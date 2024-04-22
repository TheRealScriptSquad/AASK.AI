from flask import Flask, render_template, request
from werkzeug.datastructures import ImmutableMultiDict
from aaskAILoader import aaskAI
import pandas as pd
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/web', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Access form data
        data = request.form

        # Perform calculations or processing using the form data
        ai= aaskAI('aaskAI.pkl','aaskAIEncodes.pkl')
        cols=['Age', 'Sex', 'Marital Status', 'Scholarship', 'GPA Last', 'COURSE ID', 'Reading Sci', 'Reading NonSci', 'Study Hours', 'Midterm Prep - 2', 'Notes Taken', 'Listening', "Mother's Occupation", "Father's Occupation", 'Income', "Mother's Education", "Father's Education", 'Parental Status', 'Project Impact', 'Discussion Interest']

        data_dis=dict(data)
        lis=list(data_dis.values())
        lis[5]=int(lis[5])
        df=pd.DataFrame([lis],columns=cols)

        df=ai.preprocess(df)
        result = ai.predict(df)

        print(result)

        return str(np.round(result[0],2))
        # print(data)
        # res='yes'
        # return res

    return render_template('web.html')

if __name__ == '__main__':
    app.run(debug=True,port=8080)