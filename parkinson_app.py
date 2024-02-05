from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open("parkinson_rf.pkl", "rb"))


@app.route("/")
def home():
    return render_template("parkinsson_home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # MDVP-Fo
        mdvpfo = int(request.form['mdvpfo'])
        # Spread
        spread = request.form['spread']
        # MDVP-Flo
        mdvpflo = request.form['mdvpflo']
        # MDVP-jitter
        mdvpjitter = int(request.form['mdvpjitter'])
        # D2
        d2 = int(request.form['d2'])
        # DFA
        dfa=int(request.form['dfa'])
        #MDVP-Fhi
        mdvpfhi=request.form['mdvpfhi']
        #RPDE
        rpde=int(request.form['rpde'])



        prediction=model.predict([[mdvpfo,spread,mdvpflo,mdvpjitter,d2,dfa,mdvpfhi,rpde]])

        return render_template('Parkinsson_home.html',prediction=prediction)

if __name__=="__main__":
    app.run(debug=True)