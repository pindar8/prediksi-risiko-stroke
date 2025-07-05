
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model dan encoder
model = joblib.load("model_stroke.pkl")
encoders = joblib.load("encoders.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            # Ambil data dari form
            gender = encoders['gender'].transform([request.form['gender']])[0]
            age = float(request.form['age'])
            hypertension = int(request.form['hypertension'])
            heart_disease = int(request.form['heart_disease'])
            ever_married = encoders['ever_married'].transform([request.form['ever_married']])[0]
            work_type = encoders['work_type'].transform([request.form['work_type']])[0]
            residence_type = encoders['Residence_type'].transform([request.form['Residence_type']])[0]
            avg_glucose = float(request.form['avg_glucose_level'])
            bmi = float(request.form['bmi'])
            smoking_status = encoders['smoking_status'].transform([request.form['smoking_status']])[0]

            # Bentuk array input
            data = np.array([[gender, age, hypertension, heart_disease,
                              ever_married, work_type, residence_type,
                              avg_glucose, bmi, smoking_status]])

            # Prediksi
            result = model.predict(data)[0]
            prediction = "Berisiko Stroke" if result == 1 else "Tidak Berisiko Stroke"
        except:
            prediction = "Input tidak valid atau error dalam prediksi."
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
