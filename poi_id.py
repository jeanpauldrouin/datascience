#!/usr/bin/python

import sys
import pickle
#sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#define features

payment_features = ['salary',
                    'bonus',
                    'long_term_incentive',
                    'deferred_income',
                    'deferral_payments',
                    'loan_advances',
                    'other',
                    'expenses',                
                    'director_fees', 
                    'total_payments']

stock_features = ['exercised_stock_options',
                  'restricted_stock',
                  'restricted_stock_deferred',
                  'total_stock_value']

email_features = ['to_messages',
                  'from_messages',
                  'from_poi_to_this_person',
                  'from_this_person_to_poi',
                  'shared_receipt_with_poi']


financial_features = payment_features + stock_features
features_list = ['poi'] + financial_features + email_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
 
    
### Task 2: Remove outliers
#converting dict to dataFrame and remove items
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df[features_list]
df = df.replace('NaN', np.nan)

#Removing keys
df.drop(axis=0, labels=['TOTAL','THE TRAVEL AGENCY IN THE PARK'], inplace=True)

#replace NaN by 0 in financial dataframe
df[financial_features] = df[financial_features].fillna(0)

df[financial_features].describe()
#The summary indicate that many financial features have a large std

#for email features I'll replace the NaN by the mean
from sklearn.preprocessing import Imputer
# Fill in the NaN email data with the mean
imp = Imputer(missing_values='NaN', strategy = 'mean', axis=0)

df[email_features]=imp.fit_transform(df[email_features])

# Retrieve the incorrect data for Belfer
belfer_financial = df.ix['BELFER ROBERT', 1:15].tolist()

# Delete the first element to shift left and add on a 0 to end as indicated in financial data
belfer_financial.pop(0)
belfer_financial.append(0)

# Reinsert corrected data
df.ix['BELFER ROBERT', 1:15] = belfer_financial

# Retrieve the incorrect data for Bhatnagar
bhatnagar_financial = df.ix['BHATNAGAR SANJAY', 1:15].tolist()

# Delete the last element to shift right and add on a 0 to beginning
bhatnagar_financial.pop(-1)
bhatnagar_financial = [0] + bhatnagar_financial

# Reinsert corrected data
df.ix['BHATNAGAR SANJAY', 1:15] = bhatnagar_financial

#calculate the std of each features
std=df[['poi']+financial_features].apply(lambda x: np.abs(x-x.median()) / x.std())
std=pd.DataFrame(std)

outliers=std.apply(lambda x: x>6).any(axis=1)
df_outlier= pd.DataFrame(index=df[outliers].index)

#Removing outliers
df.drop(axis=0, labels=['BHATNAGAR SANJAY',
                        'DERRICK JR. JAMES V',
                        'FREVERT MARK A',
                        'LAVORATO JOHN J',
                        'MARTIN AMANDA K',
                        'WHITE JR THOMAS E'], inplace=True)


### Task 3: Create new feature(s)
# Create the new financial features and add to the dataframe
df['money_total'] =  df['salary'] + df['bonus'] - df['expenses']
df['stock_total'] = df['total_stock_value']+df['exercised_stock_options'] + df['restricted_stock']
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments']   

features_list.append('money_total')
features_list.append('stock_total')
features_list.append('bonus_to_salary')
features_list.append('bonus_to_total')  

# Create the new email features
df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']

features_list.append('to_poi_ratio')
features_list.append('from_poi_ratio')
features_list.append('shared_poi_ratio')

# Fill any NaN data with the mean if any
df=df.fillna(df.mean(),inplace=True)

#Scoring with SelectKbest
poi=df['poi']
from sklearn.feature_selection import SelectKBest
selector=SelectKBest()
selector.fit(df,poi.tolist())
scores={df.columns[i]:selector.scores_[i] for i in range(len(df.columns))}
sorted_features=sorted(scores,key=scores.get,reverse=True)

# Get features with a score upper than 10 
my_feature_list = sorted_features[:13]

### Store to my_dataset for easy export below.
from sklearn.preprocessing import scale

#Scaling the dataset and send it back to a dictionary
scaled_df = df[my_feature_list]
scaled_df.ix[:,1:] = scale(scaled_df.ix[:,1:])
my_dataset = scaled_df.to_dict(orient='index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Create and test the Decision Tree classifier with Tuned parameters
clf = DecisionTreeClassifier(criterion='entropy', max_depth=10, min_samples_leaf=7)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)