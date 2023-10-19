import tensorflow as tf
import numpy as np 
import pickle as pkl
import os
import pandas as pd
import streamlit as st
import io 
import base64

st.title('Bankruptcy Prediction')

feature_cols = ['industrial_risk', 'management_risk', 'financial_flexibility',
       'credibility', 'competitiveness', 'operating_risk']

@st.cache_resource
def load_model():
    try:
        models = {
        'CatBoost': pkl.load(open('saved_model\CatBoostClassifier.pkl', 'rb')),
        'CategoricalNB': pkl.load(open('saved_model\CategoricalNB.pkl', 'rb')),
        'DecisionTree Entropy': pkl.load(open('saved_model\DecisionTreeClassifier_entropy.pkl', 'rb')),
        'DecisionTree Gini': pkl.load(open('saved_model\DecisionTreeClassifier_gini.pkl', 'rb')),
        'KNeighbors': pkl.load(open('saved_model\KNeighborsClassifier.pkl', 'rb')),
        'LinearSVC': pkl.load(open('saved_model\LinearSVC.pkl', 'rb')),
        'Logistic Regressor': pkl.load(open('saved_model\LogisticRegressor.pkl', 'rb')),
        'MultinomialNB': pkl.load(open('saved_model\MultinomialNB.pkl', 'rb')),
        'RandomForest': pkl.load(open('saved_model\RandomForestClassifier.pkl', 'rb')),
        'NN': tf.keras.models.load_model('saved_model/NN.h5'),
        }
        scales = {'Low':0,
                  'Medium':0.5,
                  'High':1}
        return models,scales
    except Exception as e:
        st.error(f'Error in loading model \n {e}', icon="🚨")

def download_csv(df):
    # Create a BytesIO object
    buffer = io.StringIO()

    # Save the DataFrame to the BytesIO buffer as a CSV file
    df.to_csv(buffer, index=False)

    # Create the download button
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="output_prediction.csv">Download Output CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def predict(dataframe,model,modelname):
    with st.spinner(f'Predicting Using {modelname}.....'):
        pred = model.predict(dataframe)
        if modelname == 'NN':
            pred = tf.cast((tf.squeeze(pred) > 0.5),tf.int8)
        dataframe['class'] = pred
        dataframe['class'] = dataframe['class'].apply(lambda x: 'non-bankruptcy' if x else 'bankruptcy')
        st.success(f'Prediction Completed Using {modelname}', icon="🎉")
        st.dataframe(dataframe.head())
        download_csv(dataframe)
        

def predict_ind(data,model,modelname):
    with st.spinner(f'Predicting Using {modelname}.....'):
        pred = model.predict([data])
        if modelname == 'NN':
            pred = np.expand_dims(tf.cast((tf.squeeze(pred) > 0.5),tf.int8),axis=0)
        if pred[0]:
            st.success(f"Prediction: The Company will not Face Bankruptcy", icon="🎉")
        else:
            st.error(f"Prediction: The Company will Face Bankruptcy", icon="🚨")
        

def main():
    with st.spinner('Loading Model.....'):
        models,scales = load_model()
    if not models:
        return st.error('Error in loading model', icon="🚨")
    
    toggle = st.checkbox('Upload Dataframe')

    if toggle:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        model_name = st.selectbox('Select Model', list(models.keys()))
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            try:
                df=df[feature_cols]
                if st.button('Predict'):
                    predict(df,models[model_name],model_name)
            except Exception as e:
                st.error(f'Error in loading dataframe \n {e}', icon="🚨")
    
    else:
        col1, col2 = st.columns(2)
        industrial_risk = col1.selectbox('industrial_risk', ['Low','Medium','High'])
        management_risk = col2.selectbox('management_risk', ['Low','Medium','High'])

        col3, col4 = st.columns(2)
        financial_flexibility = col3.selectbox('financial_flexibility', ['Low','Medium','High'])
        credibility = col4.selectbox('credibility', ['Low','Medium','High'])
        
        col5, col6 = st.columns(2)
        competitiveness = col5.selectbox('competitiveness', ['Low','Medium','High'])
        operating_risk = col6.selectbox('operating_risk', ['Low','Medium','High'])
        
        col7 = st.columns(1)
        model_name = col7[0].selectbox('Select Model', list(models.keys()))

        if st.button('Predict'):
            predict_ind([scales[industrial_risk],scales[management_risk],
                         scales[financial_flexibility],scales[credibility],
                         scales[competitiveness],scales[operating_risk]],models[model_name],model_name)

if __name__ == '__main__':
    main()