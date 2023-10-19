import streamlit as st
import pandas as pd
import numpy as np
import catboost as cb
import io 
import base64
import pickle as pkl

st.title('GDP Classification')

@st.cache_resource
def load_model():
    try:
        model = pkl.load(open('./catboost.h5', 'rb'))
        
        return model
    except Exception as e:
        st.error(f'Error in loading model {e}',icon="🚨")

important_features = ['Birth Rate', 'Business Tax Rate', 'CO2 Emissions',
       'Days to Start Business', 'Ease of Business', 'Energy Usage', 'GDP',
       'Health Exp % GDP', 'Health Exp/Capita', 'Hours to do Tax',
       'Infant Mortality Rate', 'Internet Usage', 'Lending Interest',
       'Life Expectancy Female', 'Life Expectancy Male', 'Mobile Phone Usage',
       'Number of Records', 'Population 0-14', 'Population 15-64',
       'Population 65+', 'Population Total', 'Population Urban',
       'Tourism Inbound', 'Tourism Outbound']


def download_csv(df):
    # Create a BytesIO object
    buffer = io.StringIO()

    # Save the DataFrame to the BytesIO buffer as a CSV file
    df.to_csv(buffer, index=False)

    # Create the download button
    b64 = base64.b64encode(buffer.getvalue().encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="output_prediction.csv">Download Output CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def main():
    with st.spinner('Loading Model....'):
        model = load_model()
    toggle = st.checkbox('Have Dataset')
    if toggle:
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            try:
                df=df[important_features]
                df.dropna(inplace=True)
                if st.button('Predict'):
                    pred = model.predict(df)
                    df['Pred'] = pred
                    st.dataframe(df.head())
                    download_csv(df)
            except Exception as e:
                st.error(f'Error in loading dataframe \n {e}', icon="🚨")
    else:
        t1 = st.checkbox('Check on test')
        
        col1, col2 = st.columns(2)
        br = col1.number_input('Birth Rate', min_value=0.0, value=0.0)
        btr = col2.number_input('Business Tax Rate', min_value=0.0, value=0.0)

        col3, col4 = st.columns(2)
        co2_emission = col3.number_input('Co2 Emission', min_value=0.0, value=0.0)
        dtsb = col4.number_input('Days to Start Business', min_value=0.0, value=0.0)

        col5, col6 = st.columns(2)
        esb = col5.number_input('Ease of business', min_value=0.0, value=0.0)
        eg = col6.number_input('Energy Usage', min_value=0.0, value=0.0)

        col7, col8 = st.columns(2)
        gdp = col7.number_input('GDP', min_value=0.0, value=0.0)
        hpgdp = col8.number_input('Health % GDP', min_value=0.0, value=0.0)

        col9, col10 = st.columns(2)
        hcap = col9.number_input('Health Capital', min_value=0.0, value=0.0)
        htdt = col10.number_input('Hours to do Tax', min_value=0.0, value=0.0)

        col11, col12 = st.columns(2)
        imr = col11.number_input('Infant Mortality Rate', min_value=0.0, max_value=100.0, value=0.0)
        iu = col12.number_input('Internet Usage', min_value=0.0, value=0.0)

        col13, col14 = st.columns(2)
        li = col13.number_input('Lending Interest', min_value=0.0, value=0.0)
        lef = col14.number_input('Life Expectancy Female', min_value=0.0, value=0.0)

        col15, col16 = st.columns(2)
        lem = col15.number_input('Life Expectancy Male', min_value=0.0, value=0.0)
        mpu = col16.number_input('Mobile Phone Usage', min_value=0.0, value=0.0)

        col17, col18 = st.columns(2)
        p1 = col17.number_input('Population 0-14', min_value=0.0, value=0.0)
        p2 = col18.number_input('Population 15-64', min_value=0.0, value=0.0)

        col19, col20 = st.columns(2)
        p3 = col19.number_input('Population 65+', min_value=0.0, value=0.0)
        pt = col20.number_input('Population Total', min_value=0.0, value=0.0)

        col21, col22 = st.columns(2)
        pu = col21.number_input('Population Urban', min_value=0.0, value=0.0)
        tib = col22.number_input('Tourism Inbound', min_value=0.0, value=0.0)

        col23,col24 = st.columns(2)
        tob = col23.number_input('Tourism Outbound', min_value=0.0, value=0.0)

        if st.button('Predict'):
            if t1:
                t = pd.read_csv('./Cleaned.csv')
                t = t[important_features]
                p = model.predict(t.iloc[0])
                st.write(t.iloc[0])
                st.success(f"Predicted Class {p[0]}", icon="🎉")
                return
            dt = pd.DataFrame({'Birth Rate':br, 'Business Tax Rate':btr, 'CO2 Emissions':co2_emission,
                           'Days to Start Business':dtsb, 'Ease of Business':esb, 'Energy Usage':eg, 'GDP':gdp,
                           'Health Exp % GDP':hpgdp, 'Health Exp/Capita':hcap, 'Hours to do Tax':htdt,
                            'infant Mortality Rate':imr, 'Internet Usage':iu, 'Lending Interest':li,
                            'Life Expectancy Female':lef,'Life Expectancy Male':lem,    'Mobile Phone Usage':mpu,'Number of Records':1,
                            'Population 0-14':p1, 'Population 15-64':p2, 'Population 65+':p3, 'Population Total':pt,
                            'Population Urban':pu, 'Tourism Inbound':tib, 'Tourism Outbound':tob},index=[0],columns=important_features)
            st.dataframe(dt)
            try:
                pred = model.predict(dt)
                st.success(f'Predicted Class {pred[0][0]}', icon="🎉")
            except Exception as e:
                st.error(f'Error in prediction {e}',icon="🚨")
            
if __name__ == '__main__':
    main()