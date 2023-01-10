import streamlit as st
import pandas as pd

import matplotlib.pyplot as plt


st.title('Data Mining Project')
st.text('1191100280  Alvin Fong Weng Yew')
st.text('1191100281  Tan Sin Zhung')
st.text('1191100292  Leong Yi Hong')

st.header('Cleaned Datasets')
df = pd.read_csv('Data_Cleaned.csv')
st.dataframe(df)
#st.write(df['Date'])

#QUESTION 1
st.header('Question 1')
st.subheader('Is there a significant difference between Race (Malay, Indian, Chinese, Foreigner) and the sales of drink in the laundry store?')
         
q1 = df[['Race','TotalSpent_RM']]
q1 = q1.dropna()
q1['TotalSpent_RM'] = q1['TotalSpent_RM'].astype(int)

malay = q1[q1['Race']=='malay']['TotalSpent_RM']
indian = q1[q1['Race']=='indian']['TotalSpent_RM']
chinese = q1[q1['Race']=='chinese']['TotalSpent_RM']

q1 = [malay, indian, chinese]
q1plt = plt.figure(figsize = (10,10))

plt.boxplot(q1,labels=['Malay','Indian','Chinese'])
plt.xlabel('Race')
plt.ylabel('Money Spent')
         
st.pyplot(q1plt)
