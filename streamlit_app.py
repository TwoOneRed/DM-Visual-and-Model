import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm

df = pd.read_csv('Data_Cleaned.csv')
dataset = pd.read_csv('laundry.csv')
weather = pd.read_csv('weather.csv')

st.title('Data Mining Project')
st.header('Member')
member = pd.DataFrame({"ID":['1191100280','1191100281','1191100292'],"Name":['Alvin Fong Weng Yew','Tan Sin Zhung','Leong Yi Hong'], "Phone Number":['011-2029 5617','011-366 1060','011-7289 2995']})
st.dataframe(member)

st.header('Laundry Datasets')
st.dataframe(dataset)

st.header('Weather Datasets')
st.dataframe(weather)

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

q2plt = plt.figure(figsize=(7,5))
ax = sns.barplot(x='Basket_Size', y='buyDrink', hue='With_Kids', data=q2)

for i in ax.containers:
    ax.bar_label(i,)

st.pyplot(q2plt)
#############################################################################

#2 Statistical Test
# Get required data
q2 = df[['Basket_Size', 'With_Kids', 'buyDrink']]

# Two Way ANOVA for statistical test
model = ols('buyDrink ~ C(Basket_Size) + C(With_Kids) + C(Basket_Size):C(With_Kids)', data=q2).fit()

st.dataframe(sm.stats.anova_lm(model, typ=2))

#width = st.sidebar.slider("plot width", 1, 25, 3)
#height = st.sidebar.slider("plot height", 1, 25, 1)

#QUESTION 3

#st.header('Question 3')
#st.subheader('Any correlations between weather information and number of people buying drinks?')


#q3 = df[['temp', 'humidity', 'windspeed','cloudcover', 'visibility', 'buyDrink']].reset_index(drop=True)

#q3plt = plt.figure(figsize=(7,5))
#ax = sns.pairplot(q3, hue ='buyDrink')

#st.pyplot(q3plt)

#QUESTION 4
st.header('Question 4')
st.subheader('Are there difference in average total spent (RM) in laundry shops between each of the age groups?')

q4 = df[['Age_Range','TotalSpent_RM']]

bins = [10, 20, 30, 40, 50, 60]
labels = ['10-20', '21-30', '31-40', '41-50', '51-60']
q4['Age Group'] = pd.cut(q4['Age_Range'], bins=bins, labels=labels)

q4 = q4.groupby('Age Group').mean('TotalSpent_RM').reset_index()

q4plt = plt.figure(figsize = (10,5))
plt.title('Average Total Spent in RM for different groups')
ax = sns.barplot(x='Age Group', y='TotalSpent_RM', data=q4)
for i in ax.containers:
    ax.bar_label(i,)

ax = sns.pairplot(q3, hue ='buyDrink')
st.pyplot(q4plt)