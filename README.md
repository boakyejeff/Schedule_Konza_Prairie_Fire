# Schedule_Konza_Prairie_Fire
Prairie Burn Detection with Multitemporal Landsat TM

Machine learning has been around for decades, but deep learning is the new focus of study within machine learning. The goals of implementing deep learning into remote sensing are resulting in much faster and accurate results for much larger amounts of data. The field of remote sensing has focused on increasing the accuracy of land classification and change detection. A possible solution for increasing accuracy is the use of a deep learning network that have been producing greater accuracy in image classifying and change detection. This study focuses on classification and change detection within the Konza Prairie in Geary County, Kansas. The images are from Landsat4 and Landsat5 spanning the years 1985-2011 and are trained and tested through a deep learning network. The experimental results show the possibility of a deep learning network producing results that could then be implement for much larger regions.



Shape Length: 480 m

Shape Area: 14400 m

Shape Side: 120 m

Pixel Size = 30x30 m or 15x15 m

4x4 or 8x8 area?

Discussion of Data Collection from ArcGIS

Image to Wide Table

Wide Row is one field

6 Bands

Data Carpentry

Wide Table conversion to Fields in Rows of Years

requires: /dsa/data/geospatial/Prairie/Burn_FinalTable.csv

requires: ./plots subfolder.

Convert Flat to Fields



produces: ./plots/pseudo_image_N.csv,

where N is the ObjectID from Burn_FinalTable

Wide Table conversion to Stacked Data

Prep for statistical analysis and visualization along bands

Prep for row-level ML baselines

requires: /dsa/data/geospatial/Prairie/Burn_FinalTable.csv

Convert Flat to Stacked



produces: ./FullStacked_data.csv

Descriptive statistical analysis and visualization along bands

Also generates the normalized data in the [0,1] Range

requires: ./FullStacked_data.csv

Stacked to Visualizations



produces: ./normalized_global_bands_mean_stdev.csv

Geospatial Visualization of Plots

requires: ./FullStacked_data.csv

requires: /dsa/data/geospatial/Prairie/tm_stats.gdb

Visualize GeoDB



Basic ML using Row-Records (Baseline)

Raw Features

requires: ./FullStacked_data.csv

Stacked_BaseLine_ML



Normalized [0,1] Features

requires: ./normalized_global_bands_mean_stdev.csv

NormalizedData_BaseLine_ML



Standard Scaler (z-score) Features

requires: ./FullStacked_data.csv

Z-Score_Scaled_BaseLine_ML



Type Markdown and LaTeX: α2


Step1_Convert_Flat_to_Fields


File Tweaks Professor Made in Advance

Converted to CSV



Found that some column names were zero padded on Month:

---------------------------------------------------------------------------

KeyError Traceback (most recent call last)

<ipython-input-22-50d2ee988660> in <module>

39 for f in feats: # Combine with features

40 data_key = "_".join([f,b,t]) # build up the key

---> 41 data = row[data_key]

42 #print("{} = {}".format(data_key,data))

43 row_data.append(data)



KeyError: 'mean_B1_1988_4'

Confirmed with Linux tools

[scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_09'

Burn_1995_09

mean_B1_1995_09

stdev_B1_1995_09

min_B1_1995_09

Tested a SED fix on _07

[scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_07' | sed -e 's/_07/_7/g'

Burn_2009_7

mean_B1_2009_7

stdev_B1_2009_7

Applied the fix

[scottgs@ballast BurntEnds]$ sed -i -e 's/_07/_7/g' Burn_FinalTable.csv

[scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_07'

[scottgs@ballast BurntEnds]$ sed -i -e 's/_09/_9/g' -e 's/_06/_6/g' -e 's/_06/_6/g' -e 's/_05/_5/g' -e 's/_04/_4/g' Burn_FinalTable.csv

import csv

import pandas as pd

DATAFILE_WIDE = '/dsa/data/geospatial/Prairie/Burn_FinalTable.csv'

def Get_Years_Months_From_Header(header):

'''

Years and Months look like 'Burn_1983_8'

This function filters to columns that start with "Burn_" and pulls out

the YYYY_M key

'''

burn_ym_prefix = "Burn_"

years_months = []

for c in header: # for each header column

if c.startswith(burn_ym_prefix): # if a Burn_*

years_months.append(c[len(burn_ym_prefix):]) # get the characters after Burn_

​

return years_months # send back this list

​

​

def BurntEnds_to_Sandwich(row,years_mo,bands,feats):


object_id = row['\ufeffOBJECTID']

print(object_id) #


###########################

# Each Year becomes a pseudo-image row

###########################

df_rows = []

for t in years_mo:

#print("*"*20, t)

isBurnt = row["Burn_"+t]

#print("Burnt {}".format(isBurnt))


# First Column is the Year_Mo

row_data = [t]


# Walk the Bands (B1-B6)

for b in bands:

for f in feats: # Combine with features

data_key = "_".join([f,b,t]) # build up the key

data = row[data_key]

#print("{} = {}".format(data_key,data))

row_data.append(data)


# Last Column is the Burn

row_data.append(isBurnt)


#print(row_data)

df_rows.append(row_data)


# All years processed, create a data fram

df = pd.DataFrame(df_rows)

cols = ["YrMo"]

for b in bands:

for f in feats: # Combine with features

col_name = "_".join([f,b]) # build up the key

cols.append(col_name)

cols.append("isBurnt")

df.columns = cols

df.to_csv("plots/pseudo_image_{}.csv".format(object_id), index=False, header=True)


return True

bands = ["B1","B2","B3","B4","B5","B6"]

feats = ["mean","stdev","min","max"]

​

​

Test one Row

​

# OBJECTID * Shape * Burn_1983_8 mean_B1_1983_8

# 1 Polygon 1 1712.5

​

​

with open(DATAFILE_WIDE, 'r') as read_obj:

csv_reader = csv.DictReader(read_obj)

​

###########################

# Preprocess the Header

###########################

​

column_names = csv_reader.fieldnames

#print(column_names) # ['\ufeffOBJECTID', 'Shape *', 'Burn_1983_8', 'mean_B1_1983_8' ...


years_mo = Get_Years_Months_From_Header(column_names) # see function above

#print(years_mo) # ['1983_8', '1984_9', '1985_4', '1985_9', ...


###########################

# Now read a row of data

###########################

row = next(csv_reader)

#print(row) # OrderedDict([('\ufeffOBJECTID', '1'),

# ('Shape *', 'Polygon'), ('Burn_1983_8', '1'),

# ('mean_B1_1983_8', '1712.5'), ...



BurntEnds_to_Sandwich(row,years_mo,bands,feats)



# header = next(csv_reader)

# print(header)

1

Chew through the data

bands = ["B1","B2","B3","B4","B5","B6"]

feats = ["mean","stdev","min","max"]

​

with open(DATAFILE_WIDE, 'r') as read_obj:

csv_reader = csv.DictReader(read_obj)

​

###########################

# Preprocess the Header

###########################

​

column_names = csv_reader.fieldnames

#print(column_names) # ['\ufeffOBJECTID', 'Shape *', 'Burn_1983_8', 'mean_B1_1983_8' ...


years_mo = Get_Years_Months_From_Header(column_names) # see function above

#print(years_mo) # ['1983_8', '1984_9', '1985_4', '1985_9', ...


###########################

# For each row, write out the file.

###########################

for row in csv_reader:

#print(row) # OrderedDict([('\ufeffOBJECTID', '1'),

# ('Shape *', 'Polygon'), ('Burn_1983_8', '1'),

# ('mean_B1_1983_8', '1712.5'), ...

BurntEnds_to_Sandwich(row,years_mo,bands,feats)


The data has been split into fields (polygons/plots)

The sub-folder plots has a all the data.

Example plots/pseudo_image_2646.csv



YrMo,mean_B1,stdev_B1,min_B1,max_B1,mean_B2,stdev_B2,min_B2,max_B2,mean_B3,stdev_B3,min_B3,max_B3,mean_B4,stdev_B4,min_B4,max_B4,mean_B5,stdev_B5,min_B5,max_B5,mean_B6,stdev_B6,min_B6,max_B6,isBurnt

1983_8,2171.5,3885.317078,0,8940,2427.25,4342.109664,0,9814,2348.5,4202.212132,0,9674,4131,7436.119696,0,17790,3423.8125,6125.547477,0,13945,2692.625,4824.232994,0,11611,1

1984_9,2300.0625,4114.751632,0,9309,2553.25,4567.609696,0,10335,2631.3125,4707.677003,0,10703,3757.125,6729.286251,0,15757,3893.125,6971.439802,0,16244,3113.375,5571.094673,0,12884,1

1985_4,2199.3125,3934.839323,0,9009,2422.625,4334.572672,0,9962,2375.6875,4252.603449,0,10006,4209.3125,7549.136434,0,17656,3527.1875,6310.554769,0,14454,2740,4905.740542,0,11516,1


Step2_Convert_Flat_to_Stacked

File Tweaks Professor Made in Advance

Converted to CSV



Found that some column names were zero padded on Month:

---------------------------------------------------------------------------

KeyError Traceback (most recent call last)

<ipython-input-22-50d2ee988660> in <module>

39 for f in feats: # Combine with features

40 data_key = "_".join([f,b,t]) # build up the key

---> 41 data = row[data_key]

42 #print("{} = {}".format(data_key,data))

43 row_data.append(data)



KeyError: 'mean_B1_1988_4'

Confirmed with Linux tools

[scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_09'

Burn_1995_09

mean_B1_1995_09

stdev_B1_1995_09

min_B1_1995_09

Tested a SED fix on _07

[scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_07' | sed -e 's/_07/_7/g'

Burn_2009_7

mean_B1_2009_7

stdev_B1_2009_7

Applied the fix

[scottgs@ballast BurntEnds]$ sed -i -e 's/_07/_7/g' Burn_FinalTable.csv

[scottgs@ballast BurntEnds]$ head -n1 Burn_FinalTable.csv | awk -F',' '{for (i=1;i<=NF;i++) printf("%s\n",$i) }' | grep '_07'

[scottgs@ballast BurntEnds]$ sed -i -e 's/_09/_9/g' -e 's/_06/_6/g' -e 's/_06/_6/g' -e 's/_05/_5/g' -e 's/_04/_4/g' Burn_FinalTable.csv

import csv

import pandas as pd

import numpy as np

def Get_Years_Months_From_Header(header):

'''

Years and Months look like 'Burn_1983_8'

This function filters to columns that start with "Burn_" and pulls out

the YYYY_M key

'''

burn_ym_prefix = "Burn_"

years_months = []

for c in header: # for each header column

if c.startswith(burn_ym_prefix): # if a Burn_*

years_months.append(c[len(burn_ym_prefix):]) # get the characters after Burn_

​

return years_months # send back this list

​

​

def BurntEnds_to_Stackable(row,years_mo,bands,feats):


object_id = row['\ufeffOBJECTID']

print(object_id) #


###########################

# Each Year becomes a pseudo-image row

###########################

df_rows = []

for t in years_mo:

#print("*"*20, t)

isBurnt = row["Burn_"+t]

#print("Burnt {}".format(isBurnt))


# First Column is the Object_ID

row_data = [object_id]


# Second Column is the Year_Mo

row_data.append(t)


# Walk the Bands (B1-B6)

for b in bands:

for f in feats: # Combine with features

data_key = "_".join([f,b,t]) # build up the key

data = row[data_key]

#print("{} = {}".format(data_key,data))

row_data.append(data)


# Last Column is the Burn

row_data.append(isBurnt)


#print(row_data)

df_rows.append(row_data)


# All years processed, create a data fram

df = pd.DataFrame(df_rows)

cols = ["OBJECTID","YrMo"]

for b in bands:

for f in feats: # Combine with features

col_name = "_".join([f,b]) # build up the key

cols.append(col_name)

cols.append("isBurnt")

df.columns = cols

return df

​

bands = ["B1","B2","B3","B4","B5","B6"]

feats = ["mean","stdev","min","max"]

​

DATAFILE_WIDE = '/dsa/data/geospatial/Prairie/Burn_FinalTable.csv'

Test one Row

​

# OBJECTID * Shape * Burn_1983_8 mean_B1_1983_8

# 1 Polygon 1 1712.5

​

​

with open(DATAFILE_WIDE, 'r') as read_obj:

csv_reader = csv.DictReader(read_obj)

​

###########################

# Preprocess the Header

###########################

​

column_names = csv_reader.fieldnames

#print(column_names) # ['\ufeffOBJECTID', 'Shape *', 'Burn_1983_8', 'mean_B1_1983_8' ...


years_mo = Get_Years_Months_From_Header(column_names) # see function above

#print(years_mo) # ['1983_8', '1984_9', '1985_4', '1985_9', ...


###########################

# Now read a row of data

###########################

row = next(csv_reader)

#print(row) # OrderedDict([('\ufeffOBJECTID', '1'),

# ('Shape *', 'Polygon'), ('Burn_1983_8', '1'),

# ('mean_B1_1983_8', '1712.5'), ...



df = BurntEnds_to_Stackable(row,years_mo,bands,feats)

print(df.head())


# header = next(csv_reader)

# print(header)

1

OBJECTID YrMo mean_B1 stdev_B1 min_B1 max_B1 mean_B2 \

0 1 1983_8 1712.5 3687.547351 0 9785 1931.8125

1 1 1984_9 1816.9375 3907.934006 0 10046 2020.5

2 1 1985_4 1723.625 3710.420028 0 9773 1918.0625

3 1 1985_9 1940.4375 4173.412688 0 10694 2124.0625

4 1 1986_3 1908.4375 4105.39208 0 10604 2118.5



stdev_B2 min_B2 max_B2 ... max_B4 mean_B5 stdev_B5 min_B5 \

0 4165.967614 0 11318 ... 17814 2992.1875 6438.507366 0

1 4349.113672 0 11431 ... 16276 3253.5625 6995.835799 0

2 4129.869287 0 10903 ... 18473 3078.625 6626.889767 0

3 4570.859503 0 11947 ... 14949 3430 7376.279776 0

4 4557.881365 0 11821 ... 15721 3633.9375 7816.161475 0



max_B5 mean_B6 stdev_B6 min_B6 max_B6 isBurnt

0 16687 2364.6875 5101.795994 0 13821 1

1 17642 2594.4375 5580.837428 0 14371 1

2 17401 2356.125 5080.62622 0 13745 1

3 18565 2845.8125 6120.183159 0 15478 1

4 19983 2961.25 6372.768258 0 16607 1



[5 rows x 27 columns]

Chew through the data

bands = ["B1","B2","B3","B4","B5","B6"]

feats = ["mean","stdev","min","max"]

​

with open(DATAFILE_WIDE, 'r') as read_obj:

csv_reader = csv.DictReader(read_obj)

​

###########################

# Preprocess the Header

###########################

​

column_names = csv_reader.fieldnames

#print(column_names) # ['\ufeffOBJECTID', 'Shape *', 'Burn_1983_8', 'mean_B1_1983_8' ...


years_mo = Get_Years_Months_From_Header(column_names) # see function above

#print(years_mo) # ['1983_8', '1984_9', '1985_4', '1985_9', ...


###########################

# For each row, write out the file.

###########################

frames = [ BurntEnds_to_Stackable(row,years_mo,bands,feats) for row in csv_reader ]

result = pd.concat(frames)

​

print(result.shape)


Step5_Stacked_BaseLine_ML

Stacked Data Baseline ML Tests
Required Data File ./FullStacked_data.csv

Basic Data Preparation
import os, sys
import numpy as np
import pandas as pd
​
# Dataset location
DATASET = 'FullStacked_data.csv'
assert os.path.exists(DATASET)
​
# Load and shuffle
dataset = pd.read_csv(DATASET).sample(frac = 1).reset_index(drop=True)
Note: Becaues we are using sample(frac = 1) we are randomizing all the data. Therefore, results will vary from time to time based on the data set reading.
dataset.head()
Unnamed: 0	OBJECTID	YrMo	mean_B1	stdev_B1	min_B1	max_B1	mean_B2	stdev_B2	min_B2	...	max_B4	mean_B5	stdev_B5	min_B5	max_B5	mean_B6	stdev_B6	min_B6	max_B6	isBurnt
0	26	1921	1998_4	9627.0625	148.36014	9380.0	9835.0	10666.8750	174.07541	10366.0	...	14582.0	17694.625	554.26610	16626.0	18582.0	13989.2500	372.94925	13346.0	14460.0	2
1	5	2359	1987_4	9020.9375	142.20946	8735.0	9207.0	10164.5000	175.25867	9805.0	...	18744.0	16703.500	385.31630	15828.0	17192.0	12714.3750	317.25485	12123.0	13289.0	1
2	37	1007	2004_8	9632.1250	160.83444	9246.0	9890.0	10515.0000	172.83981	10231.0	...	14694.0	16077.500	433.96130	15396.0	16759.0	13572.3125	264.32043	13030.0	13937.0	1
3	37	520	2004_8	10097.1875	108.13092	9854.0	10287.0	11038.4375	130.20854	10799.0	...	14448.0	17772.000	474.47235	16878.0	18519.0	14370.6250	289.18573	13827.0	14866.0	1
4	4	1637	1986_3	9573.7500	209.65480	9237.0	9917.0	10458.2500	339.71234	9847.0	...	15213.0	18303.938	639.81836	17111.0	19374.0	14178.4375	312.28876	13791.0	14823.0	1
5 rows × 28 columns

# Drop first 3 columns and isBurnt label
# 0 index of columns - so ",3" drops  {0,1,2}
X = np.array(dataset.iloc[:,3:-1])
y = np.array(dataset.isBurnt)
y = y - 1  #shift from {1.2} to {0,1} for non-burn, burn
Test Base Line ML Classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
​
Baseline a resubstitution Logistic Regression
# Create an instance of a model that can be trained
model = LogisticRegression()
​
# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)       
LR_RESUB_SCORE = model.score(X, y)
print("Logistic Regression: {0:6.5f}".format(LR_RESUB_SCORE))
/opt/conda/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
Logistic Regression: 0.88643
Baseline a resubstitution KNeighborsClassifier
# Create an instance of a model that can be trained
model = KNeighborsClassifier()
​
# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)   
KNN_RESUB_SCORE = model.score(X, y)
print("KNN : {0:6.5f}".format(KNN_RESUB_SCORE))
KNN : 0.93314
Baseline a resubstitution Decision Tree
# Create an instance of a model that can be trained
model = DecisionTreeClassifier()
​
# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)       
DT_RESUB_SCORE = model.score(X, y)
print("Decision Tree: {0:6.5f}".format(DT_RESUB_SCORE))
Decision Tree: 0.99982
Baseline a resubstitution LinearSVC
# Create an instance of a model that can be trained
model = LinearSVC()
​
# fit = "train model parameters using this data and expected outcomes"
model.fit(X, y)       
SVC_RESUB_SCORE = model.score(X, y)
print("Linear SVC Regression: {0:6.5f}".format(SVC_RESUB_SCORE))
Linear SVC Regression: 0.81846
/opt/conda/lib/python3.7/site-packages/sklearn/svm/base.py:931: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
  "the number of iterations.", ConvergenceWarning)
Resubstitution Model Summary
Logistic Regression: 0.88639
K(5) Nearest Neighbors: 0.93313
Decision Tree: 0.99982
Linear SVC: 0.80398
Cross-Fold Analysis of Classifier Generalizability
We are going to do a 5-fold cross validation for each model. Then, compare the degrade.

import sklearn.model_selection
XFOLD = 5
# Hide the pesky warnings from Logit
import warnings; warnings.simplefilter('ignore')
​
# new model
model = LogisticRegression()
# Show Prior
print("Resub Logistic Regression: {0:6.5f}".format(LR_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)
​
for i,acc in enumerate(cv_results):
    change = (acc-LR_RESUB_SCORE)/LR_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i,acc,change))
​
print("Average Logit Acc {:5.2f}%".format(np.mean(cv_results)*100))
Resub Logistic Regression: 0.88643
Fold 0: 0.88622, change -0.02%
Fold 1: 0.88655, change  0.01%
Fold 2: 0.88584, change -0.07%
Fold 3: 0.88751, change  0.12%
Fold 4: 0.88584, change -0.07%
Average Logit Acc 88.64%
​
# new model
model = KNeighborsClassifier()
# Show Prior
print("Resub KNN: {0:6.5f}".format(KNN_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)
​
for i,acc in enumerate(cv_results):
    change = (acc-KNN_RESUB_SCORE)/KNN_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i,acc,change))
    
print("Average KNN Acc {:5.2f}%".format(np.mean(cv_results)*100))
Resub KNN: 0.93314
Fold 0: 0.90660, change -2.84%
Fold 1: 0.90870, change -2.62%
Fold 2: 0.90878, change -2.61%
Fold 3: 0.90726, change -2.77%
Fold 4: 0.90804, change -2.69%
Average KNN Acc 90.79%
# new model
model = DecisionTreeClassifier()
# Show Prior
print("Resub Decision Tree: {0:6.5f}".format(DT_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)
​
for i,acc in enumerate(cv_results):
    change = (acc-DT_RESUB_SCORE)/DT_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i,acc,change))
    
print("Average Decision Tree Acc {:5.2f}%".format(np.mean(cv_results)*100))
Resub Decision Tree: 0.99982
Fold 0: 0.87447, change -12.54%
Fold 1: 0.88110, change -11.87%
Fold 2: 0.87565, change -12.42%
Fold 3: 0.87917, change -12.07%
Fold 4: 0.87895, change -12.09%
Average Decision Tree Acc 87.79%
# new model
model = LinearSVC()
# Show Prior
print("Resub SVC: {0:6.5f}".format(SVC_RESUB_SCORE))
# Run Cross Val
cv_results = sklearn.model_selection.cross_val_score(model, X, y, cv=XFOLD)
​
for i,acc in enumerate(cv_results):
    change = (acc-SVC_RESUB_SCORE)/SVC_RESUB_SCORE * 100
    print("Fold {}: {:6.5f}, change {:5.2f}%".format(i,acc,change))
    
print("Average Linear SVC Acc {:5.2f}%".format(np.mean(cv_results)*100))
Resub SVC: 0.81846
Fold 0: 0.86817, change  6.07%
Fold 1: 0.26266, change -67.91%
Fold 2: 0.86728, change  5.96%
Fold 3: 0.78662, change -3.89%
Fold 4: 0.85839, change  4.88%
Average Linear SVC Acc 72.86%
Notes
Average Logit Acc 88.64%
Average KNN Acc 90.67%
Average Decision Tree Acc 87.67%
Average Linear SVC Acc 78.55%
The high-performing decision tree seems overfit .
The linear Support Vector Machine is very inconsistent
The best is the KNN with an average Accuracy of 90.67%
