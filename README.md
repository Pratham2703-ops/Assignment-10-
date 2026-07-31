Heart Disease Prediction - ML Model Deployment
End-to-end ML project predicting heart disease risk. Deployed with Flask and Render.
Live URL
https://your-app-name.onrender.com

Dataset
Kaggle - Heart Disease Dataset

Setup
bash
pip install -r requirements.txt
python train_model.py
python app.py

API
POST /predict

JSON
{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}

Response:
JSON
{"prediction":"Heart Disease Detected","prediction_label":1,"confidence":"85.42%"}

Files
train_model.py - Model training
app.py - Flask API
model.pkl - Trained model
scaler.pkl - Feature scaler
requirements.txt - Dependencies
templates/index.html - Web UI

Conclusion
The Random Forest classifier achieved strong accuracy on the test dataset. Key challenges during deployment included ensuring proper model serialization and cloud environment compatibility. This project demonstrates the importance of MLOps practices in making ML models production-ready for healthcare applications. 
