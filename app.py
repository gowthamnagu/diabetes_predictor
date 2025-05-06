from flask import Flask,request, render_template
import numpy as np
import joblib

app=Flask(__name__)
model=joblib.load("diabetes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    try:
        features=[float(request.form[f"feature{i}"]) for i in range (1,11)] 
        prediction=model.predict([features])[0]
        return render_template("index.html",prediction=round(prediction,2))
    except Exception as e:
        return render_template("index.html",error=str(e))                                 




if __name__=="__main__":
    app.run(debug=True)