import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from PIL import Image

# load Data
gold = pd.read_csv('gld_price_data.csv')


# Splitting the Features and Target

X = gold.drop(['Date', 'GLD'], axis=1)
y = gold['GLD']

# Training The Model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = RandomForestRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
score = r2_score(y_test, pred)




# Website
st.title("Gold Price Prediction Model")
img = Image.open('gold_img.jpg')
st.image(img, width=100, use_container_width =True)

st.subheader('Using Random Forest Regressor')
st.write(gold)
st.subheader('Model Performance')
st.write(score)