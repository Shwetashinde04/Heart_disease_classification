import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Load the dataset
df = pd.read_csv(r"C:\Users\ADMIN\Downloads\Heart.CSV\heart.csv")


# Prepare data
X = df.drop(columns=['output'])
y = df['output']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=143, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


svc = SVC(C=0.1, kernel='linear', gamma='scale')
svc.fit(X_train_scaled, y_train)

# Create Streamlit app
st.title('Heart Disease Prediction')

# Add image by providing the local file path
st.image(r"C:\Users\ADMIN\Downloads\heart-rate-7504343_1280.png", caption='Your Image Caption', use_column_width=True)


# Sidebar for user input
st.sidebar.header('Enter Patient Details')
age = st.sidebar.slider('Age', min_value=0, max_value=100, value=50)
sex = st.sidebar.radio('Sex', ['Male', 'Female'])
cp = st.sidebar.selectbox('Chest Pain Type', [0, 1, 2, 3])
trtbps = st.sidebar.slider('Resting Blood Pressure', min_value=80, max_value=200, value=120)
chol = st.sidebar.slider('Cholesterol', min_value=100, max_value=600, value=200)
fbs = st.sidebar.radio('Fasting Blood Sugar', [0, 1])
restecg = st.sidebar.selectbox('Resting Electrocardiographic Results', [0, 1, 2])
thalachh = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exng = st.sidebar.radio('Exercise Induced Angina', [0, 1])
oldpeak = st.sidebar.slider('ST Depression Induced by Exercise Relative to Rest', min_value=0.0, max_value=6.2, value=2.0)
slp = st.sidebar.selectbox('Slope of the Peak Exercise ST Segment', [0, 1, 2])
caa = st.sidebar.selectbox('Number of Major Vessels Colored by Flourosopy', [0, 1, 2, 3, 4])
thall = st.sidebar.selectbox('Thal', [0, 1, 2, 3])

# Function to make predictions
def predict(model, data):
    scaled_data = scaler.transform(data)
    return model.predict(scaled_data)

# Function to map numeric predictions to labels
def map_prediction(prediction):
    if prediction == 0:
        return 'No'
    elif prediction == 1:
        return 'Yes'
    else:
        return 'Unknown'

# Button to trigger prediction
if st.sidebar.button('Predict'):
    query_point = np.array([[age, 1 if sex == 'Male' else 0, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])

    svc_pred = predict(svc, query_point)
    
    # Map numeric prediction to labels
    svc_pred_label = map_prediction(svc_pred[0])
    
    st.write('### SVC Prediction:', svc_pred_label)

