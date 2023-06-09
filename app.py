from flask import Flask, render_template, request
import pickle
import os
from flask import send_from_directory

# python3 -m flask run
app = Flask(__name__)  # name for the Flask app (refer to output)
filename = 'extra/heart_disease_prediction_mdl.pkl'

# running the server
#model = pickle.load(open(filename, 'rb'))
@app.route("/")
def index():
    # returning string
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                          'favicon.ico',mimetype='image/vnd.microsoft.icon')

@app.route("/predict", methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        age = request.form["age"]
        sex = request.form["sex"]
        trestbps = request.form["trestbps"]
        chol = request.form["chol"]
        oldpeak = request.form["oldpeak"]
        thalach = request.form["thalach"]
        fbs = request.form["fbs"]
        exang = request.form["exang"]
        slope = request.form["slope"]
        cp = request.form["cp"]
        thal = request.form["thal"]
        ca = request.form["ca"]
        restecg = request.form["restecg"]

        pred_args = [age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]

        ml_model = pickle.load(open('Model/hd_prediction_mdl.pkl', 'rb'))
        model_prediction = ml_model.predict([pred_args])
        if model_prediction == 0:
            res_val = "NO HEART PROBLEM"
        else:
            res_val = "HEART PROBLEM"
        return render_template('index.html', prediction_text='PATIENT HAS {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)
