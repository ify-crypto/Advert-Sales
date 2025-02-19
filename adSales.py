import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
#import plotly .express as px


# st.title('Advert and Sales')
# st.subheader('Built by Ifeyinwa')

advert = pd.read_csv('AdvertAndSales.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>ADVERT SALES PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by IFEYINWA</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)


st.image('pngwing.com (9).png')
st.divider()
st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown("Businesses often struggle with inaccurate sales forecasts and ineffective advertising strategies, leading to wasted marketing budgets and missed revenue opportunities. Our Advert and Sales Prediction App leverages AI-driven analytics to predict sales trends, optimize ad performance, and provide actionable insights, helping businesses maximize their ROI and make advert-driven marketing decisions.")

st.divider()


st.dataframe(advert,use_container_width= True)

st.sidebar.image('pngwing.com.png',caption = "Welcome User")

tv = st.sidebar.number_input('Television advert exp', min_value=0.0, max_value=10000.0, value=advert.TV.median())
radio = st.sidebar.number_input('Radio advert exp', min_value=0.0, max_value=10000.0, value=advert.Radio.median())
socials = st.sidebar.number_input('Social media exp', min_value= 0.0, max_value = 10000.0, value=advert['Social Media'].median())
infl = st.sidebar.selectbox('Type of Influencer', advert.Influencer.unique(), index=1)



# user input,we want to recognise the original name given in the dataset and link it

inputs = {
    'TV' : [tv],    
    'Radio' : [radio],
    'Social Media' : [socials],
    'Influencer' : [infl]
}


# if we want the input  to appear under the  dataset

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)

# transform the user inputs,import the transformers(scalers)

tv_scaler = joblib.load('TV_scaler.pkl')
radio_scaler = joblib.load('Radio_scaler.pkl')
social_scaler = joblib.load('Social Media_scaler.pkl')
influencer_encoder = joblib.load('Influencer_encoder.pkl')

# link the scalers to the user inputs

inputVar['TV'] = tv_scaler.transform(inputVar[['TV']])
inputVar['Radio'] = radio_scaler.transform(inputVar[['Radio']])
inputVar['Social Media'] = social_scaler.transform(inputVar[['Social Media']])
inputVar['Influencer'] = influencer_encoder.transform(inputVar[['Influencer']])


#Bringing in the model
model = joblib.load('advertmodel.pkl')

# we create a button to use for the prediction

predictbutton = st.button('Push to Predict the Sales')

if predictbutton: 
    predicted = model.predict(inputVar)
    st.success(f'the predicted Sales value is : {predicted}')
