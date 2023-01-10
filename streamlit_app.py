import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

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

q1plt = plt.figure(figsize=(10,4))
ax = sns.barplot(x='Race', y='buyDrink', data=q1)
for i in ax.containers:
    ax.bar_label(i,)

st.pyplot(q1plt)


#QUESTION 2
st.header('Question 2')
st.subheader('Do basket size and family with kids influence the number of people buying drinks?')

q2 = df[['Basket_Size', 'With_Kids', 'buyDrink']]

# Finding mean Monthly Family Expenses for each LifeStyle and Education
q2 = q2.groupby(['Basket_Size','With_Kids']).sum('buyDrink').round(2).reset_index()

#############################################################################
#1 Chart

q2plt = plt.figure(figsize=(9,7))
ax = sns.barplot(x='Basket_Size', y='buyDrink', hue='With_Kids', data=q2)

for i in ax.containers:
    ax.bar_label(i,)

st.pyplot(q1plt)
#############################################################################

#2 Statistical Test
# Get required data
q2 = df[['Basket_Size', 'With_Kids', 'buyDrink']]

# Two Way ANOVA for statistical test
model = ols('buyDrink ~ C(Basket_Size) + C(With_Kids) + C(Basket_Size):C(With_Kids)', data=q2).fit()

st.dataframe(sm.stats.anova_lm(model, typ=2))

#width = st.sidebar.slider("plot width", 1, 25, 3)
#height = st.sidebar.slider("plot height", 1, 25, 1)
