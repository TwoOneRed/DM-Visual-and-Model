import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

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

plt.figure(figsize=(6,6))
ax = sns.barplot(x='Race', y='buyDrink', data=q1)
for i in ax.containers:
    ax.bar_label(i,)
         
st.pyplot(ax)
