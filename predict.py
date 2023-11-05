import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask
from flask import request
from flask_cors import CORS

# read model file
with open("model.bin", 'rb') as f_in:
    model = pickle.load(f_in)
    
app = Flask("Heart")
CORS(app)

@app.route("/", methods=['GET'])

def index():
    return '<h1>Home page</h1>'

@app.route("/predict", methods=['POST'])

def predict():
    data = request.get_json()
    customer = data
    customer_df = pd.DataFrame([customer])
    cont_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    st_scaler = StandardScaler()
    X_norm=customer_df
    X_norm[cont_cols]=pd.DataFrame(st_scaler.fit_transform(X_norm[cont_cols]), columns=[cont_cols])
    
    pred = model.predict(X_norm)[0]
    outputLabels = {
    0: "no_disease",
    1: "disease"}
    
    return outputLabels[pred]
    
if __name__ == "__main__":
    app.run(debug = True, host='0.0.0.0',port="8080")
