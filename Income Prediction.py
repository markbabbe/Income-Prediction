
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')






# Read in the training data
train = os.path.join("..", "income-prediction", "income_train.csv")
train = pd.read_csv("C://Users/babbe/OneDrive/GitPractice/income-prediction/income_train.csv")
train[:5]


# In[3]:


# Check to see which columns contain null values
train.isnull().sum()


# In[4]:


# Investigate column details that have null values
null_columns = ['hispanic_origin', 'state_of_previous_residence', 'migration_msa', 'migration_reg',
       'migration_within_reg', 'migration_sunbelt',
       'country_father', 'country_mother', 'country_self']
null_columns_data = train[null_columns]
null_columns_data.head()


# In[5]:


# Investigate the counts of the null data with income level above 50k
above = train[train['income_level']==50000]
above.hispanic_origin.value_counts() # Distribution shows that it will not be a factor
above.state_of_previous_residence.value_counts() # Distribution shows that it will not be a factor
above.migration_msa.value_counts() # Distribution shows that it will not be a factor
above.country_mother.value_counts()
train.citizenship.value_counts() 


# In[6]:


# We drop columns we intuitively know are irrelivant, and drop those with too many missing values
train_drop_missing = train.drop(['hispanic_origin', 'state_of_previous_residence', 'migration_msa', 
                        'migration_reg', 'migration_within_reg', 'migration_sunbelt',
                                 'country_father', 'country_mother', 'country_self'], axis = 1)


# In[7]:


# Let's recode the income level category into 0 for under 50,000 and 1 for over
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
income_level = train_drop_missing["income_level"]
train_drop_missing["income_level"] = encoder.fit_transform(income_level)
train_drop_missing.income_level.value_counts()


# In[8]:


# Fix the imbalanced data
from sklearn.utils import resample

# Create a copy 
train_new = train_drop_missing.copy()

# Separate majority and minority classes
df_majority = train_new[train_new.income_level==0]
df_minority = train_new[train_new.income_level==1]

# Downsample majority class
df_majority_downsampled = resample(df_majority, 
                                 replace=False,    # sample without replacement
                                 n_samples=12382,     # to match minority class
                                 random_state=123) # reproducible results

# Combine minority class with downsampled majority class
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
# Display new class counts
df_downsampled.income_level.value_counts()


# In[9]:


# Rename and copy
train_downsample = df_downsampled.copy()


# In[10]:


# Let's view the correlation matrix with income level
corr_matrix = train_downsample.corr()
corr_matrix["income_level"].sort_values(ascending = False)


# In[11]:


# Sift through and continue dropping those we know will have no effect, or have a way off distribution
train_downsample = train_downsample.drop(['industry_code', 'occupation_code', 'enrolled_in_edu_inst_lastwk', 'major_industry_code',
'major_occupation_code', 'year', 'occupation_code', 'business_or_self_employed', 'veterans_benefits' ], axis = 1)


# In[12]:


# Pick final columns based on human intuition
train_downsample.columns
final_columns = ['age', 'class_of_worker', 'education', 'wage_per_hour',
       'marital_status', 'race', 'sex', 'capital_gains', 'citizenship', 'weeks_worked_in_year', 'income_level' ]
train_final = train_downsample[final_columns]


# In[13]:


# List categorical and continuous variables
continuous_variables = ['age', 'capital_gains', 'wage_per_hour', 'weeks_worked_in_year']
categorical_variables = ['class_of_worker', 'education', 'marital_status', 'race', 'sex', 'citizenship']


# In[14]:


# Visual Correlation 
scatter_matrix(train_final[continuous_variables], figsize = (12,8))


# In[15]:


# Confirm no missing values
train_final.isnull().sum()


# In[16]:


binarized_data = pd.get_dummies(train_final, columns=categorical_variables)


# In[17]:


X = binarized_data.drop('income_level', axis = 1)
y = binarized_data['income_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# In[18]:


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(X_train)
x_test_scaled = scaler.fit_transform(X_test)


# In[19]:


y_train_1 = (y_train == 1) # true for all high incomes, false for all others
y_test_1 = (y_test == 1)


# In[20]:


sgd = SGDClassifier()
sgd.fit(x_train_scaled, y_train)
sgd_pred = sgd.predict(x_train_scaled)

confusion_matrix(y_train, sgd_pred)


# In[21]:


# SGD with Cross Validation
print(cross_val_score(sgd, x_train_scaled, y_train, cv = 3, scoring = "accuracy"))

sgd_cv_pred = cross_val_predict(sgd, x_train_scaled, y_train, cv = 3)

confusion_matrix(y_train, sgd_cv_pred)


# In[22]:


def prec_recall_score1(sgd_pred, sgd_cv_pred):
    sgd_p = precision_score(y_train, sgd_pred)
    sgd_r = recall_score(y_train, sgd_pred)
    sgd_cv_p = precision_score(y_train, sgd_cv_pred)
    sgd_cv_r = recall_score(y_train, sgd_cv_pred)
    sgd_f1 = f1_score(y_train, sgd_pred)
    sgd_cv_f1 = f1_score(y_train, sgd_cv_pred)
    index = ['Precision Score', 'Recall_Score', 'F1_Score']
    return pd.DataFrame({'Metric':index, 'SGD': [sgd_p, sgd_r, sgd_f1],
                         'SGD_CV': [sgd_cv_p, sgd_cv_r, sgd_cv_f1],}).set_index('Metric')
    
prec_recall_score1(sgd_pred, sgd_cv_pred)


# In[23]:


# Now let's look at an ROC Curve of our CV SGD
fpr, tpr, thresholds = roc_curve(y_train, sgd_cv_pred)

def plot_roc_curve(fpr, trp, label = None):
    plt.plot(fpr, tpr, linewidth = 2, label = label)
    plt.plot([0,1], [0,1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr, tpr)
plt.show()


# In[24]:


# Let's see how it compares to a powerful model, Random Forest CV
forest_clf = RandomForestClassifier(random_state = 42)
forest_cv_pred = cross_val_predict(forest_clf, x_train_scaled, y_train, cv = 3)

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, forest_cv_pred)

plt.plot(fpr, tpr, "b:", label = "SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest CV")
plt.legend(loc = "bottom right")
plt.show()


# In[25]:


forest_clf.fit(x_train_scaled, y_train)
forest_pred = forest_clf.predict(x_train_scaled)
print(confusion_matrix(y_train, forest_pred))


# In[26]:


def prec_recall_score2(forest_pred, forest_cv_pred):
    forest_p = precision_score(y_train, forest_pred)
    forest_r = recall_score(y_train, forest_pred)
    forest_cv_p = precision_score(y_train, forest_cv_pred)
    forest_cv_r = recall_score(y_train, forest_cv_pred)
    forest_f1 = f1_score(y_train, forest_pred)
    forest_cv_f1 = f1_score(y_train, forest_cv_pred)
    index = ['Precision Score', 'Recall_Score', 'F1_Score']
    return pd.DataFrame({'Metric':index, 'RF': [forest_p, forest_r, forest_f1],
                         'RF_CV': [forest_cv_p, forest_cv_r, forest_cv_f1],}).set_index('Metric')
    
prec_recall_score2(forest_pred, forest_cv_pred)


# In[27]:


# Let's view all four on a graph
fpr_sgd, tpr_sgd, thresholds_sgd = roc_curve(y_train, sgd_pred)
fpr_sgd_cv, tpr_sgd_cv, thresholds_sgd_cv = roc_curve(y_train, sgd_cv_pred)
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, forest_pred)
fpr_forest_cv, tpr_forest_cv, thresholds_forest_cv = roc_curve(y_train, forest_cv_pred)

plt.plot(fpr_sgd, tpr_sgd, "-", label = "SGD")
plt.plot(fpr_sgd_cv, tpr_sgd_cv, "-", label = "SGD_CV")
plt.plot(fpr_forest, tpr_forest, "-", label = "Random Forest")
plot_roc_curve(fpr_forest_cv, tpr_forest_cv, "Random Forest_CV")
plt.legend(loc = "bottom right")
plt.show()


# In[28]:


param_grid = [{'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
             {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}]

forest_class = RandomForestClassifier()

grid_search = GridSearchCV(forest_class, param_grid, cv=5, scoring = 'neg_mean_squared_error')

grid_search.fit(x_train_scaled, y_train)


# In[29]:


grid_search.best_params_


# In[30]:


grid_search.best_estimator_


# In[31]:


grid_pred = grid_search.predict(x_train_scaled)
confusion_matrix(y_train, grid_pred)

prec_recall_score2(forest_pred, grid_pred)


# ## Predictions on the test set

# In[32]:


def test_performance(prediction, prediction_cv):
    forest_p = precision_score(y_test, prediction)
    forest_r = recall_score(y_test, prediction)
    forest_cv_p = precision_score(y_test, prediction_cv)
    forest_cv_r = recall_score(y_test, prediction_cv)
    forest_f1 = f1_score(y_test, prediction)
    forest_cv_f1 = f1_score(y_test, prediction_cv)
    index = ['Precision Score', 'Recall_Score', 'F1_Score']
    return pd.DataFrame({'Metric':index, 'RF': [forest_p, forest_r, forest_f1],
                         'RF_CV': [forest_cv_p, forest_cv_r, forest_cv_f1],}).set_index('Metric')


# In[33]:


forest_clf.fit(x_train_scaled, y_train)
forest_pred_test = forest_clf.predict(x_test_scaled)
grid_pred_test = grid_search.predict(x_test_scaled)

test_performance(forest_pred_test, grid_pred_test)


# In[45]:


from xgboost import XGBClassifier
xgboost = XGBClassifier()
xgboost.fit(x_train_scaled, y_train)
xgb_pred = xgboost.predict(x_test_scaled)
confusion_matrix(y_test, xgb_pred)


# In[69]:


class Modeling():
    """Taking care of cleaning and modeling."""
    def __init__(self, X_train, X_test, y_train, y_test, scaler):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaler = scaler
        
    def train_prediction(self, model):
        x_train_scaled = scaler.fit_transform(X_train)
        model.fit(x_train_scaled, y_train)
        prediction = model.predict(x_train_scaled)
        return prediction
    
    def train_matrix(self, prediction):
        return confusion_matrix(y_train, prediction)
    
    def train_performance(self, prediction):
        precision = precision_score(y_train, prediction)
        recall = recall_score(y_train, prediction)
        f1 = f1_score(y_train, prediction)
        index = ['Precision Score', 'Recall_Score', 'F1_Score']
        return pd.DataFrame({'Metric':index, 'Train': [precision, recall, f1],}).set_index('Metric')
        
    
#     def test_matrix
    
#     def test performance


# In[70]:


model = Modeling(X_train, X_test, y_train, y_test, StandardScaler())
rf_pred = model.train_prediction(RandomForestClassifier(random_state = 42))
model.train_matrix(rf_pred)
model.train_performance(rf_pred)


# In[74]:


from sklearn.naive_bayes import GaussianNB
nb_pred = model.train_prediction(GaussianNB())
model.train_matrix(nb_pred)
#model.train_performance(nb_pred)


# In[5]:


#from autosklearn.classification import AutoSklearnClassifier() 
import autosklearn.classification 
import sklearn.model_selection
#auto_pred = model.train_prediction()


# In[ ]:


class Evaluation():
    


# In[62]:


yeet = Modeling(X_train, X_test, y_train, y_test)
RF_pred = yeet.model_prediction(RandomForestClassifier(), StandardScaler())
yeet.train_matrix(RF_pred)


# # Final Predictions on newest data

# In[34]:


test = pd.read_csv("C://Users/babbe/OneDrive/Practice Machine Learning/income_test.csv")
test_new = test[final_columns]
test_new.head()


# In[35]:


income_level = test_new["income_level"]
test_new["income_level"] = encoder.fit_transform(income_level)
test_new.income_level.value_counts()


# In[36]:


test_final = pd.get_dummies(test_new, columns=categorical_variables)


# In[37]:


X_test_data_final = test_final.drop('income_level', axis = 1)
y_test_data_final = test_final['income_level']
X_test_scaled_final = scaler.fit_transform(X_test_data_final)


# In[38]:


final_prediction = grid_search.predict(X_test_scaled_final)
final_prediction_2 = forest_clf.predict(X_test_scaled_final)
confusion_matrix(y_test_data_final, final_prediction)
confusion_matrix(y_test_data_final, final_prediction_2)


# In[39]:


#test_performance(grid_pred_test, final_prediction)


# In[40]:


# Transform our categorical features into numbers, then convert to one-hot vectors
# We one-hot encode because we know the distance between categories means nothing
from sklearn.preprocessing import LabelBinarizer
LB = LabelBinarizer()
train_cat = train_final[categorical_variables]
train_cat_one_hot = LB.fit_transform(train_cat)

