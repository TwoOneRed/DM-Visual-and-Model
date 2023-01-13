import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.cluster import KMeans
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
import webbrowser
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('Data_Cleaned.csv')
dataset = pd.read_csv('laundry.csv')
weather = pd.read_csv('weather.csv')
smote = pd.read_csv('Data_Smote.csv')

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

###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

#QUESTION 1
st.header('Exploratory Data Analysis')
st.subheader('Question 1')
st.text('Is there a significant difference between Race (Malay, Indian, Chinese, Foreigner) and the sales of drink in the laundry store?')

q1 = df[['Race','buyDrink']]

q1 = q1.groupby('Race').sum('buyDrink').reset_index()

q1plt = plt.figure(figsize=(10,4))
ax = sns.barplot(x='Race', y='buyDrink', data=q1)
plt.title('The number of people buying drinks by Race')
for i in ax.containers:
    ax.bar_label(i,)

st.pyplot(q1plt)

##########################################################################################################################################################
#QUESTION 2
st.subheader('Question 2')
st.text('Do basket size and family with kids influence the number of people buying drinks?')

q2 = df[['Basket_Size', 'With_Kids', 'buyDrink']]

# Finding mean Monthly Family Expenses for each LifeStyle and Education
q2 = q2.groupby(['Basket_Size','With_Kids']).sum('buyDrink').round(2).reset_index()

#1 Chart
q2plt = plt.figure(figsize=(7,5))
ax = sns.barplot(x='Basket_Size', y='buyDrink', hue='With_Kids', data=q2)
plt.title('Impact of Kids on Basket Size and Drink Purchases')

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


##########################################################################################################################################################
#QUESTION 3

st.subheader('Question 3')
st.text('Any correlations between weather information and number of people buying drinks?')

q3 = df[['temp', 'humidity', 'windspeed','cloudcover', 'visibility', 'buyDrink']].reset_index(drop=True)

q3plt = plt.figure(figsize=(7,5))
ax = sns.pairplot(q3, hue ='buyDrink')
ax.fig.suptitle("The relationships between weather and number of prople buying drinks")

st.pyplot(q3plt)

##########################################################################################################################################################
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

##########################################################################################################################################################
#QUESTION 5
st.subheader('Question 5')
st.text('K-means Clustering')
st.text('Does TimeSpent_minutes and Age_Range has differences in terms of buying drinks in laundry shops?')
# Load the data
q5 = df
q5cl = df[['Age_Range','TimeSpent_minutes']]

# Create the KMeans model
kmeans = KMeans(n_clusters=8,random_state=1)

# Fit the model to the data
kmeans.fit(q5cl)

# Get the cluster labels for each data point
q5['labels'] = kmeans.predict(q5cl)

q5plot = plt.figure(figsize=(7,5))
ax = sns.scatterplot(x="Age_Range", y="TimeSpent_minutes", hue="labels", data=q5,palette='rocket',legend='full')
plt.title('Age Range vs Time Spent: K-means Clustering Analysis of Laundry Shop Customers"')
st.pyplot(q5plot)

q5plot1 = plt.figure(figsize=(7,5))
q5 = q5.groupby('labels').sum('buyDrink').reset_index()
ax = sns.barplot(data=q5, x="labels", y="buyDrink",palette='rocket')
for i in ax.containers:
    ax.bar_label(i,)
st.pyplot(q5plot1)

##########################################################################################################################################################
#QUESTION 6
st.subheader('Question 5')
st.text('Association Rule Mining')
st.text('What are the most common characteristics of customers that are most likely to come to the laundry shops?')

q6 = df[['Race','Gender','Body_Size','With_Kids','Kids_Category','Basket_colour','Attire','Shirt_Colour','shirt_type','Pants_Colour','pants_type','Wash_Item','Day','Time_Of_The_Day','Spectacles']]

# one-hot encoding
oneh = pd.get_dummies(q6)

# Find frequent item sets using the FP-growth algorithm
frequent_item_sets = fpgrowth(oneh, min_support=0.10, use_colnames=True)

# Compute association rules
rules = association_rules(frequent_item_sets, metric='confidence', min_threshold=0.3)

# Display the association rules
st.dataframe(rules)

###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

st.header('PART 2. Feature Selection')
st.subheader('BORUTA Features')

# Feature selection using BORUTA
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
boruta = BorutaPy(rf, n_estimators="auto", random_state=1)

y = df_encode["buyDrink"]
X = df_encode.drop("buyDrink", axis = 1)
colnames = X.columns

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

boruta.fit(X.values, y.values.ravel())

boruta_score = ranking(list(map(float, boruta.ranking_)), colnames, order=-1)
boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features', 'Score'])
boruta_score = boruta_score.sort_values("Score", ascending = False)

st.text('Boruta Top 10 Features')
st.dataframe(boruta_score.head(10))

st.text('Boruta Bottom 10 Features')
st.dataframe(boruta_score.tail(10))

bor = plt.figure(figsize=(20,10))

plt.subplot(2,1,1)
plt.title('BORUTA TOP 10')
plt.bar(boruta_score.head(10)['Features'], boruta_score.head(10)['Score'])

plt.subplot(2,1,2)
plt.title('BORUTA BOTTOM 10')
plt.bar(boruta_score.tail(10)['Features'], boruta_score.tail(10)['Score'])

st.text('Display Top-10 and Bottom-10 Features in barchart (BORUTA)')
st.pyplot(bor)

###########################################################################################################################################################################
###########################################################################################################################################################################

st.subheader('Recursive feature elimination (RFE) Features')

rf = RandomForestClassifier(n_jobs=-1,class_weight='balanced_subsample',max_depth = 5,n_estimators=20,random_state=1)
rf.fit(X,y)
rfe = RFECV(rf,min_features_to_select = 1, cv=2)
rfe.fit(X,y)

rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
rfe_score = rfe_score.sort_values("Score", ascending = False)

st.text('RFE Top 10 Feature')
st.dataframe(rfe_score.head(10))

st.text('RFE Bottom 10 Feature')
st.dataframe(rfe_score.tail(10))

rfe = plt.figure(figsize=(20,10))

plt.subplot(2,1,1)
plt.title('RFE TOP 10')
plt.bar(rfe_score.head(10)['Features'], rfe_score.head(10)['Score'])

plt.subplot(2,1,2)
plt.title('RFE BOTTOM 10')
plt.bar(rfe_score.tail(10)['Features'], rfe_score.tail(10)['Score'])

st.text('Display Top-10 and Bottom-10 Features in barchart (RFE)')
st.pyplot(rfe)

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
comp = plt.figure(figsize=(11.7,8.27))

ax = sns.lineplot(x = "No_Of_Features", y = "Accuracy", hue = "Model", data = boruta_acc_result)
ax.set(ylim=(0, 100))
ax.set(title="Accuracy Trend for Different Feature Selections")

st.text('Accuracy Comparison With Number Of Features and The Accuracy For BORUTA and RFE')
st.pyplot(comp)



###########################################################################################################################################################################
###########################################################################################################################################################################
###########################################################################################################################################################################

st.header('PART 3 Model Construction and Comparison')
st.subheader('Classification For Naive Bayes')
st.markdown("**Naive Bayes Top 5 Features**")

top5_df = df_encode[["dew", "humidity", "windspeed", "Age_Range", "sealevelpressure", "buyDrink"]]

#create X and y dataset
y = top5_df["buyDrink"]
X = top5_df.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top5 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 20)

top5_nb = GaussianNB()

top5_nb.fit(X_train, y_train)
y_pred = top5_nb.predict(X_test)

acc_top5nb = top5_nb.score(X_test, y_test_top5)
st.text("Accuracy: {:.4f}".format(acc_top5nb))

# get the auc score
prob_top5NB = top5_nb.predict_proba(X_test)
prob_top5NB = prob_top5NB[:, 1]

auc_top5NB = roc_auc_score(y_test_top5, prob_top5NB)
st.text('AUC: %.2f' % auc_top5NB)

###########################################################################################################################################################################

st.markdown("**Naive Bayes Top 10 Features**")

top10_df = df_encode[["dew", "humidity", "windspeed", "Age_Range", "sealevelpressure", "visibility", "TimeSpent_minutes", 
                       "winddir", "feelslike", "temp", "buyDrink"]]

#create X and y dataset
y = top10_df["buyDrink"]
X = top10_df.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top10 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 20)

top10_nb = GaussianNB()

top10_nb.fit(X_train, y_train)
y_pred = top10_nb.predict(X_test)
    
acc_top10nb = top10_nb.score(X_test, y_test_top10)
st.text("Accuracy: {:.4f}".format(acc_top10nb))

# get the auc score
prob_top10NB = top10_nb.predict_proba(X_test)
prob_top10NB = prob_top10NB[:, 1]

auc_top10NB = roc_auc_score(y_test_top10, prob_top10NB)
st.text('AUC: %.2f' % auc_top10NB)

###########################################################################################################################################################################
#ROC Curve For Top 5 and Top 10 Features

# Plot ROC Curve
fpr_top5NB, tpr_top5NB, thresholds_top5NB = roc_curve(y_test_top5, prob_top5NB) 
fpr_top10NB, tpr_top10NB, thresholds_top10NB = roc_curve(y_test_top10, prob_top10NB) 

rocnb = plt.figure(figsize = (15,12))
plt.plot(fpr_top5NB, tpr_top5NB, color='orange', label='Top-5 features') 
plt.plot(fpr_top10NB, tpr_top10NB, color='blue', label='Top-10 features') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for NB')
plt.legend()

st.markdown("**ROC Curve For Naive Bayes**")
st.pyplot(rocnb)

###########################################################################################################################################################################

st.markdown("**Naive Bayes With SMOTE dataset (top 5 features)**")
top5_df_smote = smote[["dew", "humidity", "windspeed", "Age_Range", "sealevelpressure", "buyDrink"]]

#create X and y dataset
y = top5_df_smote["buyDrink"]
X = top5_df_smote.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top5 = train_test_split(X, y, test_size=0.2, random_state = 50)

top5_nb_smote = GaussianNB()

top5_nb_smote.fit(X_train, y_train)
y_pred = top5_nb_smote.predict(X_test)

acc_top5nb_smote = top5_nb_smote.score(X_test, y_test_top5)
st.text("Accuracy: {:.4f}".format(acc_top5nb_smote))

# get the auc score
prob_top5NB = top5_nb.predict_proba(X_test)
prob_top5NB = prob_top5NB[:, 1]

auc_top5NB_smote = roc_auc_score(y_test_top5, prob_top5NB)
st.text('AUC: %.2f' % auc_top5NB_smote)


###########################################################################################################################################################################

st.markdown("**Naive Bayes With SMOTE dataset (top 10 features)**")
top10_df_smote = smote[["dew", "humidity", "windspeed", "Age_Range", "sealevelpressure", "visibility", "TimeSpent_minutes", 
                       "winddir", "feelslike", "temp", "buyDrink"]] 

#create X and y dataset
y = top10_df_smote["buyDrink"]
X = top10_df_smote.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top10 = train_test_split(X, y, test_size=0.2, random_state = 50)

top10_nb_smote = GaussianNB()

top10_nb_smote.fit(X_train, y_train)
y_pred = top10_nb_smote.predict(X_test)

acc_top10nb_smote = top10_nb_smote.score(X_test, y_test_top10)
st.text("Accuracy: {:.4f}".format(acc_top10nb_smote))

# get the auc score
prob_top10NB = top10_nb.predict_proba(X_test)
prob_top10NB = prob_top10NB[:, 1]

auc_top10NB_smote = roc_auc_score(y_test_top10, prob_top10NB)
st.text('AUC: %.2f' % auc_top10NB_smote)

###########################################################################################################################################################################
# Plot ROC Curve FOR SMOTE NB
fpr_top5NB, tpr_top5NB, thresholds_top5NB = roc_curve(y_test_top5, prob_top5NB) 
fpr_top10NB, tpr_top10NB, thresholds_top10NB = roc_curve(y_test_top10, prob_top10NB) 

nbsmote = plt.figure(figsize = (15,12))
plt.plot(fpr_top5NB, tpr_top5NB, color='orange', label='Top-5 features') 
plt.plot(fpr_top10NB, tpr_top10NB, color='blue', label='Top-10 features') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for NB using SMOTE Dataset')
plt.legend()

st.markdown("**ROC Curve For Naive Bayes with SMOTE**")
st.pyplot(nbsmote)

###########################################################################################################################################################################
###########################################################################################################################################################################

st.subheader("Classification For XGBoost")
st.markdown("**XGBoost Top 5 Features**")

y = top5_df["buyDrink"]
X = top5_df.drop("buyDrink", axis = 1)

top5_xg = XGBClassifier()

#Split train-test dataset
X_train, X_test, y_train, y_test_top5 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 20)
    
top5_xg.fit(X_train, y_train)
y_pred = top5_xg.predict(X_test)

acc_top5xg = top5_xg.score(X_test, y_test_top5)
st.text("Accuracy: {:.4f}".format(acc_top5xg))

# get the auc score
prob_top5XG = top5_xg.predict_proba(X_test)
prob_top5XG = prob_top5XG[:, 1]

auc_top5XG = roc_auc_score(y_test_top5, prob_top5XG)
st.text('AUC: %.2f' % auc_top5XG)

###########################################################################################################################################################################

st.markdown("**XGBoost Top 10 Features**")

#create X and y dataset
y = top10_df["buyDrink"]
X = top10_df.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top10 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 20)
      
top10_xg = XGBClassifier()

top10_xg.fit(X_train, y_train)
y_pred = top10_xg.predict(X_test)

acc_top10xg = top10_xg.score(X_test, y_test_top10)
st.text("Accuracy: {:.4f}".format(acc_top10xg))

# get the auc score
prob_top10XG = top10_xg.predict_proba(X_test)
prob_top10XG = prob_top10XG[:, 1]

auc_top10XG = roc_auc_score(y_test_top10, prob_top10XG)
st.text('AUC: %.2f' % auc_top10XG)

###########################################################################################################################################################################
# Plot ROC Curve XGBOOST
fpr_top5XG, tpr_top5XG, thresholds_top5XG = roc_curve(y_test_top5, prob_top5XG) 
fpr_top10XG, tpr_top10XG, thresholds_top10XG = roc_curve(y_test_top10, prob_top10XG) 

xgb = plt.figure(figsize = (15,12))
plt.plot(fpr_top5XG, tpr_top5XG, color='orange', label='Top-5 features') 
plt.plot(fpr_top10XG, tpr_top10XG, color='blue', label='Top-10 features') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost')
plt.legend()

st.markdown("**ROC Curve For XGBoost**")
st.pyplot(xgb)
###########################################################################################################################################################################

st.markdown("**XGBoost With SMOTE datasets (top 5 features)**")
#create X and y dataset
y = top5_df_smote["buyDrink"]
X = top5_df_smote.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top5 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 40)
    
top5_xg.fit(X_train, y_train)
y_pred = top5_xg.predict(X_test)

acc_top5xg_smote = top5_xg.score(X_test, y_test_top5)
st.text("Accuracy: {:.4f}".format(acc_top5xg_smote))

# get the auc score
prob_top5XG = top5_xg.predict_proba(X_test)
prob_top5XG = prob_top5XG[:, 1]

auc_top5XG_smote = roc_auc_score(y_test_top5, prob_top5XG)
st.text('AUC: %.2f' % auc_top5XG_smote)

###########################################################################################################################################################################

st.markdown("**XGBoost With SMOTE datasets (top 10 features)**")
#### create X and y dataset
y = top10_df_smote["buyDrink"]
X = top10_df_smote.drop("buyDrink", axis = 1)

#Split train-test dataset
X_train, X_test, y_train, y_test_top10 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 40)
      
top10_xg = XGBClassifier()

top10_xg.fit(X_train, y_train)
y_pred = top10_xg.predict(X_test)

acc_top10xg_smote = top10_xg.score(X_test, y_test_top10)
st.text("Accuracy: {:.4f}".format(acc_top10xg_smote))

# get the auc score
prob_top10XG = top10_xg.predict_proba(X_test)
prob_top10XG = prob_top10XG[:, 1]

auc_top10XG_smote = roc_auc_score(y_test_top10, prob_top10XG)
st.text('AUC: %.2f' % auc_top10XG_smote)
###########################################################################################################################################################################
# Plot ROC Curve
fpr_top5XG, tpr_top5XG, thresholds_top5XG = roc_curve(y_test_top5, prob_top5XG) 
fpr_top10XG, tpr_top10XG, thresholds_top10XG = roc_curve(y_test_top10, prob_top10XG) 

xgbsmote = plt.figure(figsize = (15,12))
plt.plot(fpr_top5XG, tpr_top5XG, color='orange', label='Top-5 features') 
plt.plot(fpr_top10XG, tpr_top10XG, color='blue', label='Top-10 features') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for XGBoost using SMOTE dataset')
plt.legend()

st.markdown("**ROC Curve For XGBoost with SMOTE**")
st.pyplot(xgbsmote)

st.markdown('**Accuracy For Non-Smote Dataset**')
st.text("Accuracy of NB using Top 5 features: "+ str(round(acc_top5nb, 4)))
st.text("Accuracy of NB using Top 10 features: "+ str(round(acc_top10nb, 4)))
st.text("Accuracy of XGBoost using Top 5 features: "+ str(round(acc_top5xg, 4)))  
st.text("Accuracy of XGBoost using Top 10 features: "+ str(round(acc_top10xg, 4)))

st.markdown('**Accuracy For Smote Dataset**')
st.text("Accuracy of NB using Top 5 features: "+ str(round(acc_top5nb_smote, 4)))
st.text("Accuracy of NB using Top 10 features: "+ str(round(acc_top10nb_smote, 4)))
st.text("Accuracy of XGBoost using Top 5 features: "+ str(round(acc_top5xg_smote, 4)))
st.text("Accuracy of XGBoost using Top 10 features: "+ str(round(acc_top10xg_smote, 4)))


###########################################################################################################################################################################
###########################################################################################################################################################################
st.subheader('Hyperparameter')


###########################################################################################################################################################################
###########################################################################################################################################################################

st.subheader('Ensemble')

###########################################################################################################################################################################
###########################################################################################################################################################################

st.subheader('Regression for Linear Regression')

###########################################################################################################################################################################

st.subheader('Regression for Logistic Regression')

###########################################################################################################################################################################
###########################################################################################################################################################################


with open("test.html", "rb") as html_file:
    PDFbyte = html_file.read()

st.download_button(label="Export_Report",
                    data=PDFbyte,
                    file_name="test.html",
                    mime='application/octet-stream')

