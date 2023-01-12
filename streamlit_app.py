import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from statsmodels.formula.api import ols
import statsmodels.api as sm

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

import warnings
warnings.filterwarnings('ignore')

import webbrowser

df = pd.read_csv('Data_Cleaned.csv')
dataset = pd.read_csv('laundry.csv')
weather = pd.read_csv('weather.csv')

df_encode = df.copy()
df_encode = df_encode.apply(LabelEncoder().fit_transform)

st.title('Data Mining Project')
st.header('Member')
member = pd.DataFrame({"ID":['1191100280','1191100281','1191100292'],"Name":['Alvin Fong Weng Yew','Tan Sin Zhung','Leong Yi Hong'], "Phone Number":['011-20295617','011-3661060','011-72892995']})
st.dataframe(member)

st.header('Laundry Datasets')
st.dataframe(dataset)

st.header('Weather Datasets')
st.dataframe(weather)

st.header('Cleaned Datasets')
st.dataframe(df)

#QUESTION 1
st.header('Exploratory Data Analysis')
st.subheader('Question 1')
st.text('Is there a significant difference between Race (Malay, Indian, Chinese, Foreigner) and the sales of drink in the laundry store?')

q1 = df[['Race','buyDrink']]

q1 = q1.groupby('Race').sum('buyDrink').reset_index()

q1plt = plt.figure(figsize=(10,4))
ax = sns.barplot(x='Race', y='buyDrink', data=q1)
for i in ax.containers:
    ax.bar_label(i,)

st.pyplot(q1plt)


#QUESTION 2
st.subheader('Question 2')
st.text('Do basket size and family with kids influence the number of people buying drinks?')

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

#st.subheader('Question 3')
#st.text('Any correlations between weather information and number of people buying drinks?')


#q3 = df[['temp', 'humidity', 'windspeed','cloudcover', 'visibility', 'buyDrink']].reset_index(drop=True)

#q3plt = plt.figure(figsize=(7,5))
#ax = sns.pairplot(q3, hue ='buyDrink')

#st.pyplot(q3plt)

#QUESTION 4
st.subheader('Question 4')
st.text('Are there difference in average total spent (RM) in laundry shops between each of the age groups?')

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

st.pyplot(q4plt)

st.subheader('Question 5')
st.text('ARM')
q6 = df[['Race','Gender','Body_Size','With_Kids','Kids_Category','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Day','Time_Of_The_Day','Spectacles']]

oneh = pd.get_dummies(q6)

# Find frequent item sets using the FP-growth algorithm
frequent_item_sets = fpgrowth(oneh, min_support=0.10, use_colnames=True)

# Compute association rules
rules = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.3)

# Display the association rules
rules

st.header('PART 2. Feature Selection')
st.subheader('BORUTA Features')

rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
boruta = BorutaPy(rf, n_estimators="auto", random_state=1)

y = df_encode["buyDrink"]
X = df_encode.drop("buyDrink", axis = 1)
colnames = X.columns

# use 80-20 split, random state = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

boruta.fit(np.array(X), np.array(y))

boruta_score = ranking(list(map(float, boruta.ranking_)), colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score", ascending = False)

st.text('BORUTA Top 10 Features')
bortop10 = plt.figure(figsize=(15,5))
plt.title('TOP 10')
plt.bar(boruta_score.head(10)['Features'], boruta_score.head(10)['Score'])

st.dataframe(boruta_score.head(10))
st.pyplot(bortop10)

st.text('BORUTA Bottom 10 Features')
borbot10 = plt.figure(figsize=(15,5))
plt.title('BOTTOM 10')
plt.bar(boruta_score.tail(10)['Features'], boruta_score.tail(10)['Score'])
st.dataframe(boruta_score.tail(10))
st.pyplot(borbot10)

rf = RandomForestClassifier(n_jobs=-1,class_weight='balanced_subsample',max_depth = 5,n_estimators=20,random_state=1)
rf.fit(X,y)
rfe = RFECV(rf,min_features_to_select = 1, cv=2)
rfe.fit(X,y)

rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

st.subheader('Recursive feature elimination (RFE) Features')
st.text('Recursive feature elimination  Top 10 Features')

st.dataframe(rfe_score.head(10))
rfetop10 = plt.figure(figsize=(15,5))
plt.title('TOP 10')
plt.bar(rfe_score.head(10)['Features'], rfe_score.head(10)['Score'])
st.pyplot(rfetop10)


st.text('Recursive feature elimination Bottom 10 Features')
st.dataframe(rfe_score.tail(10))
rfebot10 = plt.figure(figsize=(15,5))
plt.title('BOTTOM 10')
plt.bar(rfe_score.tail(10)['Features'], rfe_score.tail(10)['Score'])
st.pyplot(rfebot10)


st.subheader('Feature Comparison')

feature_list = ['boruta', 'rfe']
feature_num, acc_rfe, acc_boruta =[],[],[]

for i in range(1, 20):
    feature_num.append(i)
    for feature in feature_list:
        
        # Create X and y dataset
        y = df_encode['buyDrink']
        X = df_encode.drop('buyDrink', axis = 1)
        
        clf = GaussianNB()
  
        if feature == 'boruta':
            cols = boruta_score.Features[0:i]
            X = X[cols].copy()

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = round((accuracy_score(y_test, y_pred)*100), 2)
            acc_boruta.append(acc)

        elif feature == 'rfe':
            cols = rfe_score.Features[0:i]
            X = X[cols].copy()
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            acc = round((accuracy_score(y_test, y_pred)*100), 2)
            acc_rfe.append(acc)
            
boruta_acc_result = pd.DataFrame(list(zip(feature_num,acc_boruta, acc_rfe)),columns = ["No_Of_Features","BORUTA","RFE"])
boruta_acc_result = pd.melt(boruta_acc_result, id_vars = "No_Of_Features",var_name = "Model", value_name = "Accuracy")

# Plot the line charts
feaComp = plt.figure(figsize=(11.7,8.27))
ax = sns.lineplot(x = "No_Of_Features", y = "Accuracy", hue = "Model", data = boruta_acc_result)
ax.set(ylim=(0, 100))
ax.set(title="Accuracy Trend for Different Feature Selections")
st.pyplot(feaComp)

st.text("Accuracy For Recursive feature elimination = " + boruta_acc_result[boruta_acc_result["Model"] == "RFE"]['Accuracy'].mean())
st.text("Accuracy For Naive Bayes = " + boruta_acc_result[boruta_acc_result["Model"] == "BORUTA"]['Accuracy'].mean())


st.header('PART 3 Model Construction and Comparison')
st.subheader('Classification For Naive Bayes')

st.text("Naive Bayes Top 5 Features")

st.text("Naive Bayes Top 10 Features")

st.text("Compare Naive Bayes by Features")

st.text("SMOTE comparison for Naive Bayes")


st.subheader('Classification For XGBoost')

st.text("XGBoost Top 5 Features")

st.text("XGBoost Top 10 Features")

st.text("Compare XGBoost by Features")

st.text("SMOTE comparison for XGBoost")

st.subheader('Ensemble')

st.subheader('Hyperparameter')

st.subheader('Regression for Linear Regression')

st.subheader('Regression for Logistic Regression')

if st.button('Attach files to email'):
    #webbrowser.open(f'mailto:?subject={subject}&body={message}&attach={uploaded_file}')
    st.success('Files attached to email!')