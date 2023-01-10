import streamlit as st
import pandas as pd

df = pd.read_csv('Data_Cleaned.csv')

st.write(df[0])
