#!/usr/bin/env python
# coding: utf-8

# # Determines

# In[ ]:





# In[ ]:





# The 2012 US Army Anthropometric Survey (ANSUR II) was executed by the Natick Soldier Research, Development and Engineering Center (NSRDEC) from October 2010 to April 2012 and is comprised of personnel representing the total US Army force to include the US Army Active Duty, Reserves, and National Guard. In addition to the anthropometric and demographic data described below, the ANSUR II database also consists of 3D whole body, foot, and head scans of Soldier participants. These 3D data are not publicly available out of respect for the privacy of ANSUR II participants. The data from this survey are used for a wide range of equipment design, sizing, and tariffing applications within the military and has many potential commercial, industrial, and academic applications.
# The ANSUR II working databases contain 93 anthropometric measurements which were directly measured, and 15 demographic/administrative variables explained below. The ANSUR II Male working database contains a total sample of 4,082 subjects. The ANSUR II Female working database contains a total sample of 1,986 subjects.
# DATA DICT: https://data.world/datamil/ansur-ii-data-dictionary/workspace/file?filename=ANSUR+II+Databases+Overview.pdf
# To achieve high prediction success, you must understand the data well and develop different approaches that can affect the dependent variable.
# Firstly, try to understand the dataset column by column using pandas module. Do research within the scope of domain (body scales, and race characteristics) knowledge on the internet to get to know the data set in the fastest way.
# You will implement Logistic Regression, Support Vector Machine, XGBoost, Random Forest algorithms. Also, evaluate the success of your models with appropriate performance metrics.
# At the end of the project, choose the most successful model and try to enhance the scores with SMOTE make it ready to deploy. Furthermore, use SHAP to explain how the best model you choose works.

# # Table of Contents

# 1. Exploratory Data Analysis (EDA)
# 
# 2. Data Preprocessing
# 
# 3. Modelling
# 
# 4. SMOTE
# 
# 5. SHAP Analysis
# 
# 6. References
# 

# # EDA

# In[1]:


pip install xgboost


# In[2]:


pip install shap


# In[3]:


pip install pipdeptree


# In[4]:


pip show scikit-learn


# In[5]:


pip install --upgrade scikit-learn


# In[6]:


import numpy as np
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import variation


plt.rcParams["figure.figsize"] = (7,4)
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
#pd.set_option('display.width', 1000)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler




# In[ ]:





# In[7]:


df0 = pd.read_csv('ANSUR II MALE Public.csv', encoding = 'Latin-1')


# In[8]:


df1 = pd.read_csv('ANSUR II FEMALE Public.csv', encoding = 'Latin-1')


# In[9]:


# check if they have the same columns
set(df0.columns) == set(df1.columns)


# In[10]:


# print uncommon items
set(df0.columns) ^ (set(df1.columns))


# In[11]:


df0.rename(columns={'subjectid': 'SubjectId'}, inplace=True)


# In[12]:


set(df0.columns) == set(df1.columns)


# In[13]:


df = pd.concat([df0, df1], axis = 0)


# In[14]:


df.head().T


# In[15]:


# checking for null values
df.isnull().sum().any()


# In[16]:


df.isnull().sum()


# In[17]:


df.info()


# In[18]:


# checking for duplicates
df.duplicated().sum()


# In[19]:


df.shape


# In[20]:


NaN_list =[]
for columns in df.columns:
    if df[columns].isnull().sum()>0:
        print("{name} = {qty}".format(name = columns, qty = df[columns].isnull().sum()))
        NaN_list.append(columns)


# In[21]:


df = df.drop(NaN_list, axis=1)


# In[22]:


df.isnull().sum().any()


# Now, the second thing that caught my eye in the Dataset is; "SubjectNumericRace" and "DODRace" columns
# 
# **SubjectNumericRace**: a single or multi-digit code indicating a subject’s self-reported race or races (verified through interview). Where 1 = White, 2 = Black, 3 = Hispanic, 4 = Asian, 5 = Native American, 6 = Pacific Islander, 8 = Other
# **DODRace**: Department of Defense Race; a single digit indicating a subject’s self-reported preferred single race where selecting multiple races is not an option. This variable is intended to be comparable to the Defense Manpower Data Center demographic data. Where 1 = White, 2 = Black, 3 = Hispanic, 4 = Asian, 5 = Native American, 6 = Pacific Islander, 8 = Other

# In[23]:


df[["DODRace","SubjectNumericRace"]]


# In[24]:


df.drop("SubjectNumericRace", axis = 1, inplace = True)


# In[25]:


df.DODRace.value_counts(dropna = False)


# In[26]:


# Alternative code that does the same:
# df = df[(df["DODRace"] == "White") | (df["DODRace"] == "Black") | (df["DODRace"] == "Hispanic")]
df = df[df["DODRace"].isin([1,2,3])]
df.DODRace.value_counts(dropna = False)


# In[27]:


df.shape


# In[28]:


df["DODRace"].value_counts().plot(kind="pie", autopct='%1.1f%%', figsize=(10, 10))
plt.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
plt.title("Distribution of DODRace")
plt.show()


# In[29]:


df[['Weightlbs','weightkg']]


# In[30]:


# df.drop(["weightkg"], axis = 1, inplace=True)


# In[31]:


df = df[(df['Weightlbs'] >= 100) & (df['Weightlbs'] <= 270)]
df = df[(df['Heightin'] >= 59) & (df['Heightin'] <= 80)]


# In[32]:


for columns in df.select_dtypes(include=[np.number]).columns:
    if df[columns].min() == 0:
        print(columns)


# In[33]:


df["Weightlbs"].argmin()


# In[34]:


df.iloc[824][["Weightlbs","Heightin"]]


# In[35]:


df.select_dtypes(exclude=[np.number]).head().T


# In[36]:


# to find how many unique values object features have
for columns in df.select_dtypes(exclude=[np.number]).columns:
    print(f"{columns} has {df[columns].nunique()} unique value")


# "Component" and "Branch" features explain our data with the following groupings:

# In[37]:


df.groupby(["Component"])["DODRace"].value_counts()


# In[38]:


df.groupby(["Component","Branch"])["DODRace"].value_counts()


# In[39]:


df.groupby(["Component"])["DODRace"].value_counts(normalize = True).plot(kind="barh", figsize=(7,7))


# In[40]:


drop_list_nonnumeric = ["Date", "Installation", "Component","PrimaryMOS"]
df.drop(drop_list_nonnumeric, axis=1, inplace=True)


# In[41]:


df.shape


# In[42]:


df.head().T


# In[43]:


df.drop("SubjectId", axis = 1, inplace = True)


# In[44]:


plt.figure(figsize=(22, 18))
sns.heatmap(df.corr(numeric_only=True), 
            vmin=-1,
            vmax= 1,
            cmap= 'GnBu', 
            linewidths=.12, 
            linecolor='white',
            fmt='.2g',
            square=True);


# In[45]:


correlations = df.corr(numeric_only=True).unstack().sort_values()
highest_corr = correlations[correlations > 0.9]
highest_corr


# In[46]:


df_temp = df.corr()

count = "done"
feature =[]
collinear=[]
for col in df_temp.columns:
    for i in df_temp.index:
        if (df_temp[col][i]> .9 and df_temp[col][i] < 1) or (df_temp[col][i]< -.9 and df_temp[col][i] > -1) :
                feature.append(col)
                collinear.append(i)
                # print(f"multicolinearity alert in between {col} - {i}")
print("Number of strong corelated features:", count)


# In[47]:


df_col = pd.DataFrame([feature, collinear], index=["feature","collinear"]).T
df_col


# In[48]:


df_col.value_counts("feature")


# In[49]:


drop_list2 = [
      "Branch", 
    "Weightlbs", "Heightin"
]

df.drop(columns= drop_list2, inplace=True)


# In[50]:


df["DODRace"] = df.DODRace.map({1 : "White", 2 : "Black", 3 : "Hispanic"})
df.DODRace.value_counts()


# In[51]:


value_counts = df['DODRace'].value_counts()


# In[52]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='DODRace', hue='Gender', saturation=1, edgecolor='k',
              linewidth=2, palette='viridis')
plt.title(f'Distribution of Race based on Gender')
plt.xlabel('Race')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()


# In[53]:


plt.figure(figsize=(14, 9))
sns.countplot(data=df, x='Age', hue='Gender', saturation=1, edgecolor='k',
              linewidth=2, palette='viridis')
plt.title(f'Distribution of Age based on Gender')
plt.xlabel('Age')
plt.ylabel('Count')
plt.legend(title='Gender')
plt.show()


# In[54]:


# visualizing our target distribution
plt.figure(figsize=(8, 5))
sns.barplot(x=value_counts.index, y=value_counts.values,
            saturation=1, edgecolor='k',
            linewidth=2, palette='viridis')
plt.xticks(rotation=45)
plt.title(f'Target Classes')
plt.show()


# In[55]:


value_counts = df['DODRace'].value_counts()

# classes with count of 500 or more
race_classes = list(value_counts[value_counts >= 500].index)
race_classes


# In[56]:


# keep rows that has one of the 3 classes
df = df[df['DODRace'].isin(race_classes)]
df.shape


# # Outliers handling

# In[57]:


df.reset_index(drop=True, inplace=True)


# In[58]:


df.plot(by='DODRace', kind='box', subplots=True, layout=(32, 3),
        figsize=(20, 40), vert=False, sharex=False, sharey=False)
plt.tight_layout()


# In[59]:


subset = df.describe().T
subset = subset[['std', 'mean', 'max', 'min']]
subset


# In[60]:


pip install plotly


# In[61]:


import plotly.express as px


# In[ ]:





# # DATA Preprocessing

# Data Splitting

# In[62]:


# splitting X and y
X = df.drop(columns = ['DODRace'])
y = df['DODRace']


# In[63]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, stratify=y, random_state=42)


# In[64]:


print("Train features shape : ", X_train.shape)
print("Train target shape   : ", y_train.shape)
print("Test features shape  : ", X_test.shape)
print("Test target shape    : ", y_test.shape)


# In[65]:


X_train.shape


# In[66]:


X_test.shape


# In[67]:


y_train.value_counts()


# Column Transformer

# # Modelling

# first **.fit(train)** --> second **.predict(test)** --> third **.predict(train)** --> fourth **cross_val** --> fifth **GridSearchCV** --> sixth **ROC - AUC**

# In[68]:


pip install --upgrade scikit-learn


# In[69]:


#df = pd.get_dummies(df, columns=['Gender_Male'], drop_first=True)


# In[70]:


df.rename(columns={'Gender_Male_1': 'Gender'}, inplace=True)


# In[71]:


df


# In[72]:


def eval_metric(model, X_train, y_train, X_test, y_test):
    '''
    Description:
    This function gets a model, train and test sets and return 
    the confusion matrix and classification report
    
    INPUT:
    model - fitted model
    X_train - input features for the training set
    y_train - target values for training set
    X_test - input features for the testing set
    y_test - target values for testing set
    
    RETURN:
    Nothing

    '''
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))
    print()
    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:





# Scorrer for Hispanic Class

# In[73]:


from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score


# In[74]:


f1_hespanic = make_scorer(f1_score, average=None, labels=['Hispanic'])

precision_hespanic = make_scorer(precision_score, average=None, labels=['Hispanic'])

recall_hespanic = make_scorer(recall_score, average=None, labels=['Hispanic'])


scoring = {'f1_hespanic': f1_hespanic,
           'precision_hespanic': precision_hespanic, 
           'recall_hespanic': recall_hespanic}

scoring 


# # 1. Logistic model

# # Vanilla Logistic Model

# In[75]:


pip install --upgrade scikit-learn


# In[76]:


from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report, confusion_matrix, make_scorer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score



# In[77]:


from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Define your column transformer, scaler, and logistic regression model
column_trans = ColumnTransformer(transformers=[...])  # Define your column transformers here
sc = StandardScaler()
lr = LogisticRegression(max_iter=1000, random_state=42)

# Create a pipeline
lr_pipe = make_pipeline(column_trans, sc, lr)


# In[78]:


cat = X.select_dtypes("object").columns
cat


# In[79]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import make_pipeline




ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat),
                                        remainder='passthrough',
                                        verbose_feature_names_out=False).set_output(transform="pandas")


# In[80]:


column_trans.fit_transform(X_train)


# In[81]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore",
                                                      sparse=False), cat),
                                                      remainder=MinMaxScaler(),
                                                      verbose_feature_names_out=False)


# In[82]:


from sklearn.pipeline import Pipeline
operations = [("OneHotEncoder", column_trans),
              ("log", LogisticRegression(class_weight='balanced',
                                         random_state=101))]

pipe_log_model = Pipeline(steps=operations)


# In[83]:


from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


operations = [("OneHotEncoder", column_trans),
              ("DT_model", DecisionTreeClassifier(random_state=101))]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


# In[88]:


pipe_log_model.fit(X_train,y_train)
eval_metric(pipe_log_model, X_train, y_train, X_test, y_test)


# In[89]:


scaler =StandardScaler() # will be used in the pipelines


# In[90]:


scoring = {"precision_Hispanic" : make_scorer(precision_score, average = None, labels = ["Hispanic"]),
           "recall_Hispanic" : make_scorer(recall_score, average = None, labels =["Hispanic"]),
           "f1_Hispanic" : make_scorer(f1_score, average = None, labels =["Hispanic"])}


# In[92]:


from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression  # or any other relevant imports
from sklearn.pipeline import Pipeline


# In[93]:


operations = [("OneHotEncoder", column_trans), 
              ("log", LogisticRegression(class_weight='balanced',
                                         random_state=101))]

model = Pipeline(steps=operations)


scores = cross_validate(model,
                        X_train, 
                        y_train, 
                        scoring=scoring,
                        cv=10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# # Logistic Model GridsearchCV

# In[94]:


recall_Hispanic =  make_scorer(recall_score, average=None, labels=["Hispanic"])


# In[95]:


log_model = LogisticRegression() # will be used in the pipelines


# In[96]:


log_pipe = Pipeline([("scaler",scaler),("log_model",log_model)]) # pipeline for logistic regression


# In[97]:


param_grid = {
    "log__C": [0.1, 0.2],
    'log__penalty': ["l1", "l2"],
    'log__solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
}


# In[98]:


from sklearn.model_selection import GridSearchCV

# Your other code here

operations = [("OneHotEncoder", column_trans), 
              ("log", LogisticRegression(class_weight='balanced',
                                         random_state=101))]

model = Pipeline(steps=operations)


log_model_grid = GridSearchCV(model,
                              param_grid,
                              scoring=recall_Hispanic,
                              cv=10,
                              n_jobs=-1,
                              return_train_score=True)


# In[99]:


log_model_grid.fit(X_train,y_train)


# In[100]:


log_model_grid.best_estimator_


# In[101]:


pd.DataFrame(log_model_grid.cv_results_).loc[log_model_grid.best_index_, ["mean_test_score", "mean_train_score"]]


# In[102]:


eval_metric(log_model_grid, X_train, y_train, X_test, y_test)


# In[103]:


operations = [("OneHotEncoder", column_trans), 
              ("log", LogisticRegression(C=0.2,
                                         class_weight='balanced',
                                         random_state=101))]

model = Pipeline(steps=operations)


scores = cross_validate(model,
                        X_train, 
                        y_train, 
                        scoring=scoring,
                        cv=10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# In[104]:


from scikitplot.metrics import plot_roc, plot_precision_recall

operations = [("OneHotEncoder", column_trans), 
              ("log", LogisticRegression(C=0.2,
                                         class_weight='balanced',
                                         random_state=101))]

model = Pipeline(steps=operations)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
    
plot_precision_recall(y_test, y_pred_proba)
plt.show();


# In[105]:


pd.get_dummies(y_test).values


# In[106]:


from sklearn.metrics import average_precision_score

y_test_dummies = pd.get_dummies(y_test).values

average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])


# In[107]:


y_pred = log_model_grid.predict(X_test)

log_AP = average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])
log_precision = precision_score(y_test, y_pred, average=None, labels=["Hispanic"])
log_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])


# # Model Performance on Classification Tasks

# In[108]:


from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay,classification_report


# In[109]:


def eval_metric(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("Test_Set")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print()
    print("Train_Set")
    print(confusion_matrix(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))


# In[110]:


ConfusionMatrixDisplay.from_estimator(pipe_model,X_test, y_test,normalize='all')


# In[111]:


eval_metric(pipe_model, X_train, y_train, X_test, y_test)

# Although our data is inbalanced, we will not make any imbalanced treatment to the data
# since our scores are close to each other.


# # Cross Validate

# In[112]:


from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# In[113]:


operations = [("OneHotEncoder", column_trans),
              ("DT_model", DecisionTreeClassifier(random_state=101))]

model = Pipeline(steps=operations)


scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring = ["accuracy",
                                   "precision_micro",
                                   "recall_micro",
                                   "f1_micro"],
                        cv = 10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# # Feature İmportances with Pipeline

# In[114]:


cat


# In[115]:


pipe_model["DT_model"].feature_importances_


# In[116]:


features = pipe_model["OneHotEncoder"].get_feature_names_out()
features

# The new feature order we got from pipe_model is as follows.


# In[117]:


X_train.head(1)


# In[121]:


df_f_i = pd.DataFrame(data=pipe_model["DT_model"].feature_importances_,
                      index=features, #index=X.columns
                      columns=["Feature Importance"])

df_f_i = df_f_i.sort_values("Feature Importance", ascending=False)

df_f_i


# In[122]:


ax = sns.barplot(x = df_f_i.index, y = 'Feature Importance', data = df_f_i)
ax.bar_label(ax.containers[0],fmt="%.3f");
plt.xticks(rotation = 90)
plt.show();


# # Drop most important feature

# The feature that weighs too much on the estimate can sometimes cause overfitting. For this reason,
# the most important feature can be dropped and the scores can be checked again

# In[123]:


X.head(2)


# In[124]:


X2 = X.drop(columns = ["elbowrestheight"])


# In[125]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=101)


# In[126]:


operations = [("OneHotEncoder", column_trans),
              ("DT_model", DecisionTreeClassifier(random_state=101))]

pipe_model2 = Pipeline(steps=operations)

pipe_model2.fit(X_train2, y_train2)


# In[127]:


eval_metric(pipe_model2, X_train2, y_train2, X_test2, y_test2)


# In[128]:


operations = [("OrdinalEncoder", column_trans),
              ("DT_model", DecisionTreeClassifier(random_state=101))]

model = Pipeline(steps=operations)

scores = cross_validate(model,
                        X_train2,
                        y_train2,
                        scoring = ["accuracy",
                                   "precision_micro",
                                   "recall_micro",
                                   "f1_micro"],
                        cv = 10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# # 2. SVC

# In[129]:


operations_svc = [("OneHotEncoder", column_trans), 
                  ("svc", SVC(class_weight="balanced",random_state=101))]

pipe_svc_model = Pipeline(steps=operations_svc)


# In[130]:


pipe_svc_model.fit(X_train, y_train)

eval_metric(pipe_svc_model, X_train, y_train, X_test, y_test)


# In[131]:


model = Pipeline(steps=operations_svc)

scores = cross_validate(model, 
                        X_train, 
                        y_train, 
                        scoring=scoring,
                        cv=10, 
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# # SVC Model GridsearchCV

# In[132]:


recall_Hispanic =  make_scorer(recall_score, average=None, labels=["Hispanic"])


# In[133]:


param_grid = {
    'svc__C': [0.2, 0.3],
    'svc__gamma': ["scale", "auto", 0.01],
    'svc__kernel':['linear', 'rbf']
}


# In[134]:


operations_svc = [("OneHotEncoder", column_trans),
                  ("svc", SVC(class_weight="balanced",random_state=101))]

model = Pipeline(steps=operations_svc)

svm_model_grid = GridSearchCV(model,
                              param_grid,
                              scoring=recall_Hispanic,
                              cv=10,
                              n_jobs=-1,
                              return_train_score=True)


# In[135]:


svm_model_grid.fit(X_train, y_train)


# In[136]:


svm_model_grid.best_estimator_


# In[137]:


pd.DataFrame(svm_model_grid.cv_results_).loc[svm_model_grid.best_index_, ["mean_test_score", "mean_train_score"]]


# In[138]:


operations_svc = [("OneHotEncoder", column_trans),
                  ("svc", SVC(C=0.2, 
                              class_weight='balanced', 
                              kernel='linear', 
                              random_state=101))]

model = Pipeline(steps=operations_svc)

scores = cross_validate(model, 
                        X_train, 
                        y_train, 
                        scoring=scoring,
                        cv=10, 
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# In[139]:


eval_metric(svm_model_grid, X_train, y_train, X_test, y_test)


# In[140]:


operations = [("OneHotEncoder", column_trans), 
              ("svc", SVC(C=0.2, 
                          class_weight='balanced', 
                          kernel='linear', 
                          random_state=101))]

model = Pipeline(steps=operations)

model.fit(X_train, y_train)

decision_function = model.decision_function(X_test)

#y_pred_proba = model.predict_proba(X_test)
    
plot_precision_recall(y_test, decision_function)
plt.show();


# In[141]:


decision_function


# In[142]:


average_precision_score(y_test_dummies[:, 1], decision_function[:, 1])


# In[143]:


y_pred = svm_model_grid.predict(X_test)

svc_AP = average_precision_score(y_test_dummies[:, 1], decision_function[:, 1])
svc_precision = precision_score(y_test, y_pred, average=None, labels=["Hispanic"])
svc_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])


# # 3.RF

# In[146]:


cat


# In[147]:


from sklearn.preprocessing import OrdinalEncoder


ord_enc = OrdinalEncoder(handle_unknown='use_encoded_value', 
                         unknown_value=-1)

column_trans = make_column_transformer((ord_enc, cat), 
                                       remainder='passthrough')


# # Vanilla RF Model 

# In[148]:


from sklearn.ensemble import RandomForestClassifier




# In[149]:


operations_rf = [("OrdinalEncoder", column_trans), 
                 ("RF_model", RandomForestClassifier(class_weight="balanced", 
                                                     random_state=101))]

pipe_model_rf = Pipeline(steps=operations_rf)

pipe_model_rf.fit(X_train, y_train)


# In[150]:


eval_metric(pipe_model_rf, X_train, y_train, X_test, y_test)


# In[151]:


operations_rf = [("OrdinalEncoder", column_trans), 
                 ("RF_model", RandomForestClassifier(class_weight="balanced",
                                                     random_state=101))]

model = Pipeline(steps=operations_rf)

scores = cross_validate(model,
                        X_train, 
                        y_train,
                        scoring=scoring,
                        cv=10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# # RF Model GridsearchCV

# In[152]:


param_grid = {'RF_model__n_estimators':[400,500],
             'RF_model__max_depth':[2,3]} #'min_samples_split':[18,20,22], 'max_features': ['sqrt', 'log2' None, 15, 20]
             


# In[153]:


operations_rf = [("OrdinalEncoder", column_trans),
                 ("RF_model", RandomForestClassifier(class_weight="balanced",
                                                     random_state=101))]

model = Pipeline(steps=operations_rf)
rf_grid_model = GridSearchCV(model,
                             param_grid,
                             scoring=recall_Hispanic,
                             n_jobs=-1,
                             cv=10,
                             return_train_score=True)


# In[154]:


rf_grid_model.fit(X_train,y_train)


# In[155]:


rf_grid_model.best_estimator_


# In[156]:


rf_grid_model.best_params_


# In[157]:


pd.DataFrame(rf_grid_model.cv_results_).loc[rf_grid_model.best_index_, ["mean_test_score", "mean_train_score"]]


# In[158]:


rf_grid_model.best_score_


# In[159]:


eval_metric(rf_grid_model, X_train, y_train, X_test, y_test)


# In[160]:


operations_rf = [("OrdinalEncoder", column_trans), 
                 ("RF_model", RandomForestClassifier(class_weight="balanced",
                                                     max_depth=2, 
                                                     n_estimators=400, 
                                                     random_state=101))]

model = Pipeline(steps=operations_rf)

model.fit(X_train, y_train)

y_pred_proba = model.predict_proba(X_test)
    
plot_precision_recall(y_test, y_pred_proba)
plt.show();


# In[161]:


average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])


# In[162]:


y_pred = rf_grid_model.predict(X_test)

rf_AP = average_precision_score(y_test_dummies[:, 1], y_pred_proba[:, 1])
rf_precision = precision_score(y_test, y_pred, average=None, labels=["Hispanic"])
rf_recall = recall_score(y_test, y_pred, average=None, labels=["Hispanic"])


# # 4. XGBoost¶

# # Vanilla XGBoost Model

# In[163]:


from xgboost import XGBClassifier




# In[164]:


operations_xgb = [("OrdinalEncoder", column_trans), 
                  ("XGB_model", XGBClassifier(random_state=101))]

pipe_model_xgb = Pipeline(steps=operations_xgb)

y_train_xgb = y_train.map({"Black":0, "Hispanic":1, "White":2}) # sıralama classification_report ile aynı olacak.
y_test_xgb = y_test.map({"Black":0, "Hispanic":1, "White":2})
# xgb 1.6 ve üzeri versiyonlarda target numeric olmaz ise hata döndürüyor. Bu sebeple manuel olarak dönüşümü yapıyoruz.


pipe_model_xgb.fit(X_train, y_train_xgb)


# In[165]:


eval_metric(pipe_model_xgb, X_train, y_train_xgb, X_test, y_test_xgb)


# In[166]:


from sklearn.utils import class_weight
classes_weights = class_weight.compute_sample_weight(class_weight='balanced', 
                                                     y=y_train_xgb)
classes_weights


# In[167]:


my_dict = {"weights": classes_weights, "label":y_train_xgb}

comp = pd.DataFrame(my_dict)

comp.head()


# In[168]:


comp.groupby("label").value_counts()


# In[169]:


pipe_model_xgb.fit(X_train,
                   y_train_xgb,  
                   XGB_model__sample_weight=classes_weights)
# weight parameter in XGBoost is per instance not per class. Therefore, we need to assign the weight of each class to its 
# instances, which is the same thing.


# In[170]:


eval_metric(pipe_model_xgb, X_train, y_train_xgb, X_test, y_test_xgb)


# In[171]:


scoring_xgb = {"precision_Hispanic" : make_scorer(precision_score, average = None, labels =[1]),
           "recall_Hispanic" : make_scorer(recall_score, average = None, labels =[1]),
           "f1_Hispanic" : make_scorer(f1_score, average = None, labels =[1])}

#the XGBoost classifier does not accept categorical target variables. To work with it, you are converting the categorical target variable "Hispanic" to a numeric class, specifically 1. This allows you to use numeric values for scoring.


# In[172]:


operations_xgb = [("OrdinalEncoder", column_trans), 
                  ("XGB_model", XGBClassifier(random_state=101))]

model = Pipeline(steps=operations_xgb)

scores = cross_validate(model, 
                        X_train, 
                        y_train_xgb, 
                        scoring=scoring_xgb,
                        cv=5, 
                        return_train_score=True,
                        fit_params={"XGB_model__sample_weight":classes_weights})
df_scores = pd.DataFrame(scores, index = range(1, 6))
df_scores.mean()[2:]


# # XGBoost Model GridsearchCV

# In[173]:


param_grid = {"XGB_model__n_estimators":[20, 40],
              'XGB_model__max_depth':[1,2],
              "XGB_model__learning_rate": [0.03, 0.05],
              "XGB_model__subsample":[0.8, 1],
              "XGB_model__colsample_bytree":[0.8, 1]}


# In[174]:


operations_xgb = [("OrdinalEncoder", column_trans),
                  ("XGB_model", XGBClassifier(random_state=101))]

model = Pipeline(steps=operations_xgb)

xgb_grid_model = GridSearchCV(model, 
                              param_grid, 
                              scoring=make_scorer(recall_score, average = None, labels =[1]),
                              cv=5,
                              n_jobs=-1,
                              return_train_score=True)


# In[175]:


xgb_grid_model.fit(X_train,
                   y_train_xgb,
                   XGB_model__sample_weight=classes_weights)


# In[176]:


xgb_grid_model.best_estimator_


# In[177]:


xgb_grid_model.best_params_


# In[178]:


pd.DataFrame(xgb_grid_model.cv_results_).loc[xgb_grid_model.best_index_, ["mean_test_score", "mean_train_score"]]


# In[179]:


xgb_grid_model.best_score_


# In[180]:


eval_metric(xgb_grid_model, X_train, y_train_xgb, X_test, y_test_xgb)


# In[181]:


from scikitplot.metrics import plot_roc, precision_recall_curve


operations_xgb = [("OrdinalEncoder", column_trans), 
                  ("XGB_model", XGBClassifier(colsample_bytree=0.8,
                                              learning_rate=0.05,
                                              max_depth=2,
                                              n_estimators=40,
                                              subsample=1,
                                              random_state=101))]

model = Pipeline(steps=operations_xgb)

model.fit(X_train, 
          y_train_xgb, 
          XGB_model__sample_weight=classes_weights)

y_pred_proba = model.predict_proba(X_test)
    
plot_precision_recall(y_test_xgb, y_pred_proba)
plt.show()


# In[182]:


y_test_xgb_dummies = pd.get_dummies(y_test_xgb).values


# In[183]:


average_precision_score(y_test_xgb_dummies[:, 1], y_pred_proba[:, 1])


# In[184]:


y_pred = xgb_grid_model.predict(X_test)

xgb_AP = average_precision_score(y_test_xgb_dummies[:, 1], y_pred_proba[:, 1])
xgb_precision = precision_score(y_test_xgb, y_pred, average=None, labels=[1])
xgb_recall = recall_score(y_test_xgb, y_pred, average=None, labels=[1])


# # Comparing Models

# In[185]:


compare= pd.DataFrame({"Model": ["Logistic Regression", "SVM",  "Random Forest", "XGBoost"],
                       "Precision": [log_precision[0], svc_precision[0], rf_precision[0], xgb_precision[0]],
                       "Recall": [log_recall[0], svc_recall[0], rf_recall[0], xgb_recall[0]],
                       "AP": [log_AP, svc_AP, rf_AP, xgb_AP]})

compare


# In[186]:


plt.figure(figsize=(14,10))
plt.subplot(311)
compare = compare.sort_values(by="Precision", ascending=False)
ax=sns.barplot(x="Precision", y="Model", data=compare, palette='viridis')
ax.bar_label(ax.containers[0],fmt="%.3f")

plt.subplot(312)
compare = compare.sort_values(by="Recall", ascending=False)
ax=sns.barplot(x="Recall", y="Model", data=compare, palette='viridis')
ax.bar_label(ax.containers[0],fmt="%.3f")

plt.subplot(313)
compare = compare.sort_values(by="AP", ascending=False)
ax=sns.barplot(x="AP", y="Model", data=compare, palette='viridis')
ax.bar_label(ax.containers[0],fmt="%.3f")
plt.show();


# Before the Deployment
# Choose the model that works best based on your chosen metric
# For final step, fit the best model with whole dataset to get better performance.
# And your model ready to deploy, dump your model and scaler.

# In[187]:


column_trans_final = make_column_transformer((OneHotEncoder(handle_unknown="ignore", 
                                                            sparse=False), cat),
                                              remainder=MinMaxScaler())

operations_final = [("OneHotEncoder",column_trans_final),
                    ("log", LogisticRegression(C=0.2,
                                               class_weight='balanced',
                                               random_state=101))]

final_model = Pipeline(steps=operations_final)


# In[188]:


final_model.fit(X, y)


# In[189]:


X[X.Gender=="Male"].describe()


# In[190]:


male_mean_human = X[X.Gender=="Male"].describe(include="all").loc["mean"]
male_mean_human


# In[191]:


male_mean_human["Gender"] = "Male"
male_mean_human["SubjectsBirthLocation"] = "California"
male_mean_human["WritingPreference"] = "Right hand"


# In[192]:


pd.DataFrame(male_mean_human).T


# In[193]:


final_model.predict(pd.DataFrame(male_mean_human).T)


# In[194]:


from sklearn.metrics import matthews_corrcoef

y_pred = final_model.predict(X_test)

matthews_corrcoef(y_test, y_pred)


# In imbalanced datasets, accuracy is an unreliable metric. Therefore, in imbalanced datasets, metrics like Matthews correlation coefficient (MCC) and Cohen's Kappa score can be used instead of accuracy.

# In[195]:


from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(y_test, y_pred)


# # SMOTE

# In[196]:


get_ipython().system('pip install imblearn')


# In[197]:


pip install imbalanced-learn


# In[198]:


pip install --upgrade scikit-learn


# In[199]:


pip install --upgrade imbalanced-learn scikit-learn


# In[200]:


from imblearn.over_sampling import SMOTE # azınlık olan classları çoğunluk classa eşitler veya yakınlaştırır.
from imblearn.under_sampling import RandomUnderSampler # çoğunluk olan classı azınlık olan classa eşitler veya yakınlaştırır
from imblearn.pipeline import Pipeline as imbpipeline


# In[201]:


column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", 
                                                      sparse=False), cat), 
                                       remainder=MinMaxScaler())


# In[202]:


X_train_ohe = column_trans.fit_transform(X_train)


# In[203]:


over = SMOTE()
X_train_over, y_train_over = over.fit_resample(X_train_ohe, y_train)


# In[204]:


X_train_over.shape


# In[205]:


y_train_over.value_counts()


# In[206]:


under = RandomUnderSampler()
X_train_under, y_train_under = under.fit_resample(X_train_ohe, y_train)


# In[207]:


X_train_under.shape


# In[208]:


y_train_under.value_counts()


# In[209]:


over = SMOTE(sampling_strategy={"Hispanic": 1000})
under = RandomUnderSampler(sampling_strategy={"White": 2500})


# In[210]:


y_train.value_counts()


# In[211]:


X_resampled_over, y_resampled_over = over.fit_resample(X_train_ohe, y_train)


# In[212]:


y_resampled_over.value_counts()


# In[213]:


X_resampled_under, y_resampled_under = under.fit_resample(X_train_ohe, y_train)


# In[214]:


y_resampled_under.value_counts()


# In[215]:


# Yaptığımız over_sampling ve under_sampling işlemlerimizi otomotize hale getiriyoruz.
steps = [('o', over), ('u', under)]


pipeline = imbpipeline(steps=steps)

#önce hispanic clasının sayısı bizim verdiğimiz talimat kapsamında 1000'e çıkarılıp, sonra white clası 2500'e indirilecek.
X_resampled, y_resampled = pipeline.fit_resample(X_train_ohe, y_train)


# In[216]:


y_resampled.value_counts()


# In[217]:


y_train.value_counts()


# # Logistic Regression Over/Under Sampling

# In[ ]:


Şimdi yukarda yaptığımız tüm dönüşümleri ve modele verme işlemlerini otomotize hale getirelim.

get_ipython().run_line_magic('pinfo', 'do')

smote_pipeline.fit(X_train, y_train) 

--> column_trans.fit_transform(X_train) #(Onehooencoder and minmaxscaler for numeric features)

--> over.fit_resample(X_train_transform, y_train) 
                                     
--> under.fit_resample(X_train_transform_over, y_train_over)

--> log_model.fit(X_train_transform_over_under, y_train_over_under)


for predict, over and under sumpling algortims do nothing for X_test.

smote_pipeline.predict(X_test)

--> column_trans.transform(X_test) #(Onehotoencoder and minmaxscaler for numeric features) 

--> log_model.predict(X_test_transform)


# In[218]:


column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore",
                                                      sparse=False), cat), 
                                       remainder=MinMaxScaler())


# In[219]:


operations = [("OneHotEncoder",column_trans), 
              ('o', over),
              ('u', under), 
              ("log", LogisticRegression(C=0.2, 
                                         random_state=101))] #("scaler", MinMaxScaler())

# over veya under sampling dataya uygulandığında kesinlikle class_weight="balanced" kullanılmaz.


# In[220]:


smote_pipeline = imbpipeline(steps=operations)


# In[221]:


smote_pipeline.fit(X_train, y_train)


# In[222]:


eval_metric(smote_pipeline, X_train, y_train, X_test, y_test)


# In[223]:


model = imbpipeline(steps=operations)

scores = cross_validate(model,
                        X_train,
                        y_train,
                        scoring=scoring,
                        cv=10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# # SHAP

# In[224]:


column_trans_shap = make_column_transformer((OneHotEncoder(handle_unknown="ignore",
                                                           sparse=False), cat), 
                                             remainder=MinMaxScaler(),
                                             verbose_feature_names_out=False)

X_train_trans = column_trans_shap.fit_transform(X_train)
X_test_trans = column_trans_shap.transform(X_test)

model_shap = LogisticRegression(C=0.4,
                                class_weight='balanced',
                                random_state=101,
                                penalty="l1",
                                solver='saga')

model_shap.fit(X_train_trans, y_train)

# X-train ve X_test'e onehotencoder dönüşümü uyguluyoruz.
# shap pipeline ile kurulmuş model ile çalışmadığından dönüşüm işlemlerini manuel olarak yapacağız.


# In[225]:


eval_metric(model_shap, X_train_trans, y_train, X_test_trans, y_test)


# In[226]:


operations = [("OneHotEncoder", column_trans_shap),
              ("log", LogisticRegression(C=0.4,
                                         class_weight='balanced',
                                         random_state=101,
                                         penalty="l1",
                                         solver='saga'))]

model = Pipeline(steps=operations)

scores = cross_validate(model,
                        X_train,
                        y_train, 
                        scoring=scoring,
                        cv=10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]

# overfitting yok.


# In[227]:


features = column_trans_shap.get_feature_names_out()
features


# # SHAP values for Feature Selection (train data)¶

# In[228]:


import shap


# In[229]:


explainer = shap.LinearExplainer(model_shap, X_train_trans)
# model_shap modeli ve X_train_trans datası üzerinde SHAP değerlerini hesaplamak için bir LinearExplainer nesnesi oluşturuluyor
# ve explainer değişkenine atanıyor. lineer bir modelin SHAP değerlerini hesaplamak için kullanılır.

shap_values = explainer.shap_values(X_train_trans)
# explainer nesnesi ile shap değerleri hesaplanıyor.

shap.summary_plot(shap_values, 
                  max_display=300, 
                  feature_names = features, 
                  plot_size=(20,100), 
                  class_names=["black", "hispanic", "white"])

# maviler black, pembe white ve yeşil hispanic
# gördüğünüz gibi en önemli ilk 10 feature içerisinde hispanic'in tahminine katkı sağlayan feature yok.
# hispaniclerin predictionlara katkısı olan 15 feature seçiyoruz.


# In[230]:


shap.summary_plot(shap_values[1], 
                  max_display=300, 
                  feature_names = features, 
                  plot_size=(20,100), 
                  plot_type="bar") 
# hispanicleri ayrıştırmada iyi olan featurların sıralaması.


# In[231]:


hispanic=['footbreadthhorizontal',
          'headlength',
          'SubjectsBirthLocation',
          'bimalleolarbreadth',
          'tragiontopofhead',
          'Age',
          'earlength',
          'forearmcircumferenceflexed',
          'bizygomaticbreadth',
          'waistfrontlengthsitting',
          'trochanterionheight',
          'wristcircumference',
          'buttockheight',
          'elbowrestheight',
          'mentonsellionlength']  


# In[232]:


len(hispanic)


# In[233]:


#X.columns


# In[ ]:


X2 = X[hispanic]
X2.head()


# In[234]:


cat_new = X2.select_dtypes("object").columns
cat_new


# In[235]:


X2.shape


# In[236]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y, test_size=0.2, random_state=42, stratify =y)


# In[237]:


column_trans_shap = make_column_transformer((OneHotEncoder(handle_unknown="ignore", 
                                                           sparse=False), cat_new), 
                                             remainder=MinMaxScaler(),
                                             verbose_feature_names_out=False)

operations_shap = [("OneHotEncoder",column_trans_shap),
                   ("log", LogisticRegression(C=0.4, 
                                              class_weight='balanced',
                                              random_state=101,
                                              penalty="l1",
                                              solver='saga'))]

pipe_shap_model = Pipeline(steps=operations_shap)
pipe_shap_model.fit(X_train2, y_train2)


# In[238]:


X_test2


# In[239]:


eval_metric(pipe_shap_model, X_train2, y_train2, X_test2, y_test2)


# In[240]:


model = Pipeline(steps=operations_shap)

scores = cross_validate(model, 
                        X_train2,
                        y_train2, 
                        scoring=scoring,
                        cv=10,
                        return_train_score=True)
df_scores = pd.DataFrame(scores, index = range(1, 11))
df_scores.mean()[2:]


# In[241]:


from scikitplot.metrics import plot_roc, precision_recall_curve
from scikitplot.metrics import plot_precision_recall


model = Pipeline(steps=operations_shap)

model.fit(X_train2, y_train2)

y_pred_proba = model.predict_proba(X_test2)
    
plot_precision_recall(y_test2, y_pred_proba)
plt.show();


# In[ ]:




