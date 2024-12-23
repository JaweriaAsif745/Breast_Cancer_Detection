from flask import Flask, request, render_template
import numpy as np
import pickle

# Load the model
# model = pickle.load(open("breast_cancer_model.pkl", 'rb'))

# Load the pipeline
pipeline = pickle.load(open("breast_cancer_pipeline.pkl", 'rb'))


# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    features = request.form['features']
    features_lst = features.split(',')
    np_features = np.asarray(features_lst, dtype=np.float32)
    pred = pipeline.predict(np_features.reshape(1, -1))

    output = ["cancerous" if pred[0] == 1 else "not cancerous"]

    return render_template('index.html', message=output)

# Run the app
if __name__ == "__main__":
    app.run(debug=True)