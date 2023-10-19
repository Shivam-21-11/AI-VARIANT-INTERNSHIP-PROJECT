import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.metrics.r_square import RSquare
import streamlit as st
import pickle as pkl
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

st.title('PMSM Motor Speed Prediction')

important_feature = ['i_d', 'i_q', 'u_q', 'u_d', 'coolant', 'pm', 'torque']

@st.cache_resource
def load_model():
    try:
        model_dict = {'Catboost' : pkl.load(open('model/catboost.pkl', 'rb')),
        'Bayesian Ridge' : pkl.load(open('model/Bayesian.pkl', 'rb')),
        'KNN' : pkl.load(open('model/knn.pkl', 'rb')),
        'Dense Network' : tf.keras.models.load_model('./model/neuralnet.h5'),
        'Ridge' : pkl.load(open('model/ridge.pkl', 'rb')),
        'SGD' : pkl.load(open('model/sgd.pkl', 'rb'))}
        return model_dict
    except Exception as e:
        st.error('Error in loading model', icon="🚨")



def download_csv(df):
    # Create a BytesIO object
    buffer = io.StringIO()

    # Save the DataFrame to the BytesIO buffer as a CSV file
    df.to_csv(buffer, index=False)

    # Create the download button
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)



def predict_2000(data,model,model_name):
    with st.spinner(f'Predicting Using {model_name}.....'):
        try:
            pred = model.predict(data)
            if len(pred.shape) == 1:
                fig , ax = plt.subplots()
                ax.plot(range(len(data) - 2000, len(data)), pred[len(data) - 2000:], label='Prediction',color='red')
                plt.xlabel('Index')
                plt.ylabel('Motor Speed')
                plt.legend()
                st.pyplot(fig)
            else:
                pred = pred.flatten()
                fig, ax = plt.subplots()
                ax.plot(range(len(data) - 2000, len(data)), pred[len(data) - 2000:], label='Prediction',color='red')
                plt.xlabel('Index')
                plt.ylabel('Motor Speed')
                plt.legend()
                st.pyplot(fig)
            download_csv(pd.DataFrame({'motor_speed': pred}))
        except Exception as e:
            st.error('Error Performing Prediction',icon="🚨")




def predict(data,model,model_name):
    with st.spinner(f'Predicting Using {model_name}.....'):
        try:
            pred = model.predict(data)
            if len(pred.shape) == 1:
                fig, ax = plt.subplots()
                ax.plot(range(len(data)), pred, label='Prediction', color='red')
                plt.xlabel('Index')
                plt.ylabel('Motor Speed')
                plt.legend()
                st.pyplot(fig)
            else:
                pred = pred.flatten()
                fig, ax = plt.subplots()
                ax.plot(range(len(data)), pred, label='Prediction', color='red')
                plt.xlabel('Index')
                plt.ylabel('Motor Speed')
                plt.legend()
                st.pyplot(fig)
            download_csv(pd.DataFrame({'motor_speed': pred}))
        except Exception as e:
            st.error('Error Performing Prediction', icon="🚨")




def predict_plain(data,model,model_name):
    with st.spinner(f'Predicting Using {model_name}....'):
        pred = model.predict([data])
        if model_name != 'Dense Network':
            st.success(f"The Predicted Motor Speed is {round(pred[0],7)}")
        else:
            st.success(f"The Predicted Motor Speed is {round(pred[0][0],7)}")




def main():
    with st.spinner('Loading Model.....'):
        models = load_model()

    toggle = st.checkbox("Upload DataFrame")
    if toggle:
        file = st.file_uploader('Upload a CSV file', type='csv')
        model_name = st.selectbox('Model', ('Catboost', 'Bayesian Ridge', 'KNN', 'Dense Network', 'Ridge', 'SGD'))
        if st.button('Predict'):
            if file is None:
                st.error('Invalid File',icon="🚨")
            else:
                csv = pd.read_csv(file)
                try:
                    csv = csv[important_feature]
                    if csv.shape[0] > 2000:
                        st.warning('Num Rows in CSV is > 2000 Plotting only last 2000 Datapoints')
                        predict_2000(csv.iloc[:-2000,:], models[model_name],model_name)
                    else:
                        predict(csv,models[model_name],model_name)
                except Exception as e:
                    st.error(f'DataFrame Error , Make sure CSV has following columns {important_feature}')
    else:
        col1, col2 = st.columns(2)
        i_d = col1.number_input("i_d", value=0.0,step=0.1,format="%.7f")
        i_q = col2.number_input("i_q", value=0.0,step=0.1,format="%.7f")
        col3, col4 = st.columns(2)
        u_q = col3.number_input("u_q", value=0.0,step=0.1,format="%.7f")
        u_d = col4.number_input("u_d", value=0.0,step=0.1,format="%.7f")
        col5, col6 = st.columns(2)
        coolant = col5.number_input("coolant", value=0.0,step=0.1,format="%.7f")
        pm = col6.number_input("pm", value=0.0,step=0.1,format="%.7f")
        col7,col8 = st.columns(2)
        torque = col7.number_input("torque", value=0.0,step=0.1,format="%.7f")
        model_name = col8.selectbox('Model', ('Catboost', 'Bayesian Ridge', 'KNN', 'Dense Network', 'Ridge', 'SGD'))
        if st.button('Predict'):
            predict_plain([i_d, i_q, u_q, u_d, coolant, pm, torque],models[model_name],model_name)




if __name__ == '__main__':
    main()