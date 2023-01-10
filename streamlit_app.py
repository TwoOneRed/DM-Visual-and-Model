import streamlit as st
import pandas as pd

df = pd.read_csv('Data_Cleaned.csv')

st.dataframe(df)
#st.write(df['Date'])
