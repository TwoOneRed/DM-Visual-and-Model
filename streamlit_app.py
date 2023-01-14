import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy import mean,std
from statsmodels.formula.api import ols
import statsmodels.api as sm
from sklearn.cluster import KMeans
from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split, GridSearchCV, RepeatedStratifiedKFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score

from xgboost import XGBClassifier
from mlxtend.frequent_patterns import fpgrowth, association_rules
import webbrowser
import warnings
import pickle
warnings.filterwarnings('ignore')


df = pd.read_csv('Data_Cleaned.csv')
dataset = pd.read_csv('laundry.csv')
weather = pd.read_csv('weather.csv')
smote = pd.read_csv('Data_Smote.csv')

df_encode = df.copy()
df_encode = df_encode.apply(LabelEncoder().fit_transform)

df_onehot = df.copy()

col_vars = [col for col in df_onehot.columns.tolist() if df_onehot[col].dtype.name == "object"]

for var in col_vars:
    col_list = 'var'+'_'+var
    col_list = pd.get_dummies(df_onehot[var], prefix = var)
    df_temp = df_onehot.join(col_list)
    df_onehot = df_temp

df_vars = df_onehot.columns.values.tolist()
to_keep = [i for i in df_vars if i not in col_vars]
df_onehot = df_onehot[to_keep]
df_onehot.head()

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

with open("Report.html", "rb") as html_file:
    PDFbyte = html_file.read()

st.download_button(label="Download_Report",
                    data=PDFbyte,
                    file_name="Report.html",
                    mime='application/octet-stream')

if st.button('Email Report'):
    webbrowser.open(f'mailto:?subject=Laundry Report&body=The file link attach is the link for the laundry report. \n https://github.com/TwoOneRed/DMProject/blob/main/Report.html')

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

st.image('pairplot.png', caption='My Image', use_column_width=True)

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

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))

# Feature selection using BORUTA
rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", max_depth=5)
boruta = BorutaPy(rf, n_estimators="auto", random_state=1)

y = df_encode["buyDrink"]
X = df_encode.drop("buyDrink", axis = 1)
colnames = X.columns

boruta = pickle.load(open('boruta.fs','rb'))

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

rfe = pickle.load(open('rfe.fs','rb'))

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

top5_nb = pickle.load(open('top5.nb','rb'))

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

top10_nb = pickle.load(open('top10.nb','rb'))

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

top5_nb_smote = pickle.load(open('top5.nb','rb'))

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

top10_nb_smote = pickle.load(open('top10.nb','rb'))

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

#Split train-test dataset
X_train, X_test, y_train, y_test_top5 = train_test_split(X, y, test_size=0.2, stratify = y, random_state = 20)
    
top5_xg = pickle.load(open('top5.xg','rb'))

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
      
top10_xg = pickle.load(open('top10.xg','rb'))

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
    
top5_xg = pickle.load(open('top5.xg','rb'))

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
      
top10_xg = pickle.load(open('top10.xg','rb'))

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


#create X and y dataset
y = top5_df["buyDrink"]
X = top5_df.drop("buyDrink", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

best_model = pickle.load(open('nb.hyperparameter','rb'))

best_model = best_model.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

# accuracy and AUC
st.text('\nAccuracy of Tuned Naive Bayes Using Top-5 features: {:.4f}'.format(best_model.score(X_test, y_test)))

# get the auc score
prob_nb = best_model.predict_proba(X_test)
prob_nb = prob_nb[:, 1]

auc_nb = roc_auc_score(y_test, prob_nb)
st.text('AUC: %.2f' % auc_nb)


#create X and y dataset (top-5 features)
y = top5_df["buyDrink"]
X = top5_df.drop("buyDrink", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

best_model = pickle.load(open('xgboost.hyperparameter','rb'))

# Extract best model from 'grid_xg'
best_model = best_model.best_estimator_

# Predict the test set labels
y_pred = best_model.predict(X_test)

# accuracy and  AUC score
st.text('Accuracy of Tuned Xgboost classifier Using Top-5 features: {:.4f}'.format(best_model.score(X_test, y_test)))

# get the auc score
prob_xg = best_model.predict_proba(X_test)
prob_xg = prob_xg[:, 1]

auc_xg = roc_auc_score(y_test, prob_xg)
st.text('AUC: %.2f' % auc_xg)

# Plot ROC Curve
fpr_NB, tpr_NB, thresholds_NB = roc_curve(y_test, prob_nb) 
fpr_XG, tpr_XG, thresholds_XG = roc_curve(y_test, prob_xg) 

hyper = plt.figure(figsize = (15,12))
plt.plot(fpr_XG, tpr_XG, color='orange', label='XGBoost') 
plt.plot(fpr_NB, tpr_NB, color='blue', label='NB') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Hyperparameter Tuning')
plt.legend()

st.markdown('**ROC For Hyperparameter Tuning For XGBoost and Naive Bayes**')
st.pyplot(hyper)

###########################################################################################################################################################################
###########################################################################################################################################################################

st.subheader('Ensemble')

# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('rf', RandomForestClassifier()))    
    
    # define the stacking ensemble
    level1 = RandomForestClassifier()     
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = cross_val_score(model, X, y, scoring='f1', cv=cv, n_jobs=-1, error_score='raise')
    return scores

y = df_onehot.buyDrink
X = df_onehot.drop("buyDrink", 1)
colnames = X.columns

# Train-Test-Split using 20% test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2) 

# construct a list of models in a dictionary
models = dict()

models['nb'] = GaussianNB()
models['xgboost'] = XGBClassifier()
models['stacking'] = get_stacking()

results, names = list(), list()

for name,model in models.items():
    scores = evaluate_model(model,X,y)
    results.append(scores)
    names.append(name)
    st.text('>%s %.3f (%.3f)' % (name,mean(scores),std(scores)))

# plot model performance for comparison
ensemble = plt.figure(figsize = (15,12))
plt.boxplot(results, labels=names, showmeans=True)
plt.title('Stacking Ensemble Comparisons')
st.pyplot(ensemble)

###########################################################################################################################################################################
###########################################################################################################################################################################

st.subheader('Regression for Linear Regression')

df_LinR = df[['Age_Range','TotalSpent_RM']]
df_LinR = df_LinR.groupby('Age_Range').mean().reset_index()

#Put Data in X and y
X = df_LinR[['Age_Range']]     # drop labels from original data
y = df_LinR[["TotalSpent_RM"]]    # copy the labels to another dataframe/series

#Split The Data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1)
#Train The Data
LinR = LinearRegression().fit(X_train, y_train)

#Predict'
predicted = LinR.predict(X_test)

result = pd.DataFrame()
result['Age_Range'] = X_test
result['Actual'] = y_test
result['Predicted'] = predicted
st.dataframe(result)

from sklearn.metrics import mean_absolute_error, r2_score

# Calculate MAE
mae = mean_absolute_error(y_test, predicted)

# Calculate R-squared
r2 = r2_score(y_test, predicted)

st.text("MAE Score = "+ str(mae))
st.text("R2 Score  = "+ str(r2))

#Plot Line Plot
lr = plt.figure(figsize = (10,8))
plt.rc('axes', labelsize= 14)
plt.rc('xtick', labelsize=13)

x = sns.scatterplot(data=result, y="Actual", x="Age_Range" ,label='Previous TotalSpent_RM')
x = sns.scatterplot(data=result, y="Predicted", x="Age_Range",label='Predicted TotalSpent_RM')
x.set_title('Previous and Predicted TotalSpent_RM',fontsize=14)
plt.xlabel('Age Range')
plt.ylabel('Total Spent (RM)')
st.pyplot(lr)

###########################################################################################################################################################################

st.subheader('Regression for Logistic Regression')
#create X and y dataset
y = df_encode["buyDrink"]
X = df_encode.drop("buyDrink", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=50)

logreg = LogisticRegression(solver='liblinear', max_iter=200)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

st.text('Accuracy of logistic regression: {:.4f}'.format(logreg.score(X_test, y_test)))

# get the auc score
prob_logreg = logreg.predict_proba(X_test)
prob_logreg = prob_logreg[:, 1]

auc_logreg = roc_auc_score(y_test, prob_logreg)
st.text('AUC: %.2f' % auc_logreg)

fpr_logreg, tpr_logreg, thresholds_logreg = roc_curve(y_test, prob_logreg) 

logr = plt.figure(figsize = (10,8))
plt.plot(fpr_logreg, tpr_logreg, color='orange', label='Logistic Regression') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression')
plt.legend()

st.pyplot(logr)


#create X and y dataset
y = smote["buyDrink"]
X = smote.drop("buyDrink", axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=50)

logreg = LogisticRegression(solver='liblinear', max_iter=200)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

st.text('Accuracy of logistic regression: {:.4f}'.format(logreg.score(X_test, y_test)))

# get the auc score
prob_logreg = logreg.predict_proba(X_test)
prob_logreg = prob_logreg[:, 1]

auc_logreg = roc_auc_score(y_test, prob_logreg)
st.text('AUC: %.2f' % auc_logreg)

fpr_XG, tpr_XG, thresholds_XG = roc_curve(y_test, prob_logreg) 

logrsmote = plt.figure(figsize = (10,8))
plt.plot(fpr_XG, tpr_XG, color='orange', label='XGBoost') 
plt.plot([0, 1], [0, 1], color='green', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for Logistic Regression using SMOTE')
plt.legend()

st.pyplot(logrsmote)

###########################################################################################################################################################################
###########################################################################################################################################################################
