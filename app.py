
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model", "ann_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)


FEATURES = list(scaler.feature_names_in_)


CLASS_NAMES = {
    0: "Malignant",
    1: "Benign"
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "POST":

        try:
            input_values = []

            for feature in FEATURES:
                value = request.form.get(feature)

                if value is None or value.strip() == "":
                    raise ValueError(
                        f"Please enter a value for {feature}"
                    )

                number = float(value)

                if not np.isfinite(number):
                    raise ValueError(
                        f"Invalid value for {feature}"
                    )

                input_values.append(number)

            
            input_data = pd.DataFrame(
                [input_values],
                columns=FEATURES
            )

            
            scaled_data = scaler.transform(input_data)

        
            prediction = model.predict(
                scaled_data,
                verbose=0
            )

            if prediction.shape != (1, 2):
                raise ValueError(
                    "Unexpected ANN output shape: "
                    + str(prediction.shape)
                )

            
            predicted_class = int(np.argmax(prediction[0]))

            result = CLASS_NAMES[predicted_class]

   
            scores = prediction[0].tolist()

            return render_template(
                "result.html",
                result=result,
                scores=scores,
                class_names=CLASS_NAMES
            )

        except ValueError as e:

            return render_template(
                "predict.html",
                features=FEATURES,
                error=str(e)
            )

        except Exception as e:

            return render_template(
                "predict.html",
                features=FEATURES,
                error="Prediction failed: " + str(e)
            )

    return render_template(
        "predict.html",
        features=FEATURES
    )


if __name__ == "__main__":
    app.run(debug=True)
