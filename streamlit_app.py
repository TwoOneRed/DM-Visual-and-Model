import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Data_Cleaned.csv')

st.title('Data Mining Project')
st.text('1191100280  Alvin Fong Weng Yew')
st.text('1191100281  Tan Sin Zhung')
st.text('1191100292  Leong Yi Hong')

st.header('Cleaned Datasets')
st.dataframe(df)

#QUESTION 1
st.header('Question 1')
st.subheader('Is there a significant difference between Race (Malay, Indian, Chinese, Foreigner) and the sales of drink in the laundry store?')

q1 = df[['Race','buyDrink']]

q1 = q1.groupby('Race').sum('buyDrink').reset_index()

width = st.sidebar.slider("plot width", 1, 25, 3)
height = st.sidebar.slider("plot height", 1, 25, 1)

q1plt = plt.figure(figsize=(8,8))
ax = sns.barplot(x='Race', y='buyDrink', data=q1)
for i in ax.containers:
    ax.bar_label(i,)

st.pyplot(q1plt)
