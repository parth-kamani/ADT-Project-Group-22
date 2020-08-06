import pandas as pd
import datetime                                        # To handle dates
import calendar                                        # To get month
import statsmodels.formula.api as sm
import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
import sklearn.metrics as metrics                      # To get regression metrics
import scipy as sp
import time                                            # To do time complexity analysis
import random
import copy
import profile
import cProfile
from sklearn.cluster import KMeans                     # perform clustering operation

# =============================================================================
# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)
# =============================================================================

# Data preprocessing:
#loading in raw data
features_df = pd.read_csv("features.csv")
stores_df = pd.read_csv("stores.csv")
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# merging the data
 
# =============================================================================
#  (train + Store + Feature)
#  (test + Stoee + Feature)
#  
# =============================================================================

# =============================================================================
# train_bt = pd.merge(train_df,stores_df) 
# train_df = pd.merge(train_bt,features_df)
# 
# test_bt = pd.merge(test_df,stores_df)
# test_df= pd.merge(test_bt,features_df)
# 
# =============================================================================
# =============================================================================
# print(features_df.head())
# print(features_df.describe())
# 
# print(train_df.head())
# print(train_df.describe())
# print(train_df.tail())
# 
# =============================================================================
# =============================================================================
# print(test_df.head(2))
# print(test_df.describe())
# 
# print(train_df.info())
# =============================================================================



# Creating a custom season dictionary to identify the season in each month
seasons_dict = {
    1:"Winter",
    2:"Winter",
    3:"Spring",
    4:"Spring",
    5:"Spring",
    6:"Summer",
    7:"Summer",
    8:"Summer",
    9:"Fall",
    10:"Fall",
    11:"Fall",
    12:"Winter"
}


# Creating the master dataset
master_df = train_df.merge(stores_df, on='Store', how='left')
master_df = master_df.merge(features_df, on=['Store', 'Date'], how='left')

d = copy.deepcopy(master_df)

d1 = d["Weekly_Sales"]

print(d["Weekly_Sales"].describe())

print("Percentile less than 3% provides only negative value : ",d["Weekly_Sales"].quantile(0.003))


x = np.concatenate((d1[d["Weekly_Sales"] < 0], d1[d["Weekly_Sales"] > 0]))

plt.hist(x, density=True)

plt.xlim([-70496, 200000])
plt.xlabel('Weekly Sales Values')
plt.ylabel('Normalized Sales Values')
plt.title('Normalized distribution of sales values')
plt.show()

print(master_df.head())

# Filling empty markdown columns
master_df['MarkDown1'] = master_df['MarkDown1'].fillna(0)
master_df['MarkDown2'] = master_df['MarkDown2'].fillna(0)
master_df['MarkDown3'] = master_df['MarkDown3'].fillna(0)
master_df['MarkDown4'] = master_df['MarkDown4'].fillna(0)
master_df['MarkDown5'] = master_df['MarkDown5'].fillna(0)

# Cleaning holiday columns
master_df['isHoliday'] = master_df['IsHoliday_x']
master_df = master_df.drop(columns=['IsHoliday_x', 'IsHoliday_y'])

# Handling Date and time
master_df['Date'] = pd.to_datetime(master_df['Date'], format='%Y-%m-%d')
master_df['Week_Number'] = master_df['Date'].dt.week
master_df['Quarter'] = master_df['Date'].dt.quarter
master_df['Month'] = master_df['Date'].dt.month.apply(lambda x: calendar.month_abbr[x])
master_df['Season'] = (master_df['Date'].apply(lambda dt: (dt.month%12 + 3)//3)).map(seasons_dict)
master_df["Year"] = master_df["Date"].dt.year

#Creating lagged variables based on time
master_df=master_df.sort_values(by=['Store', 'Dept', 'Year', 'Week_Number'], ascending=True)    

# Previous week sales
shifted_sales = master_df.shift(1)
master_df_new_var = master_df.join(shifted_sales[['Store', 'Dept', 'Week_Number', 'Weekly_Sales', 'Year']], rsuffix='_Lag')
master_df_new_var.loc[(master_df_new_var.Dept != master_df_new_var.Dept_Lag) |  (master_df_new_var.Store != master_df_new_var.Store_Lag), 'Weekly_Sales_Lag'] = -2

#Creating dummy variables for categorical values

#forming categorarical variables
master_df_new_var = master_df_new_var.join(pd.get_dummies(master_df['Quarter'], prefix='Quarter'))
master_df_new_var = master_df_new_var.join(pd.get_dummies(master_df['Season'], prefix='Season'))
master_df_new_var = master_df_new_var.join(pd.get_dummies(master_df['Store'], prefix='Store'))
master_df_new_var = master_df_new_var.join(pd.get_dummies(master_df['Dept'], prefix='Dept'))
master_df_new_var = master_df_new_var.join(pd.get_dummies(master_df['Type'], prefix='Type'))
master_df_new_var = master_df_new_var.join(pd.get_dummies(master_df['Week_Number'], prefix='Week_Number'))

# Removing wrongly recorded data points
master_df_new_var = master_df_new_var.dropna()
#master_df_new_var = master_df_new_var.loc[master_df_new_var['Week_Number'] > 4]
#master_df_new_var = master_df_new_var.loc[master_df_new_var['Monthly_Sales_Lag'] > 0]
master_df_new_var = master_df_new_var.loc[master_df_new_var['Weekly_Sales_Lag'] > 0]
master_df_new_var = master_df_new_var.loc[master_df_new_var['Weekly_Sales'] > 0] #keeping points which are only positive in value for the sales

# Creating interaction variable
master_df_new_var['MarkDown'] = master_df_new_var['MarkDown1'] + master_df_new_var['MarkDown2'] + master_df_new_var['MarkDown3'] + master_df_new_var['MarkDown4'] + master_df_new_var['MarkDown5']
master_df_new_var['MarkDown*Weekly_Sales_Lag'] = master_df_new_var['MarkDown']*master_df_new_var['Weekly_Sales_Lag']



# Model Building
# Dividing the dataset into test and train dataset

#segregating the data
data_test = master_df_new_var[master_df_new_var.Year == 2012]
data_train = master_df_new_var[master_df_new_var.Year != 2012]

# Building full model linear regression
features = list(master_df_new_var)
for x in ('Quarter',
 'Month',
 'Season',
 'Year',
 'Store',
 'Dept',
 'Date',
 'Type',
 'Weekly_Sales',
 'Weekly_Sales_Lag'):
    features.remove(x)
#features

# Writing the formula
equals_to_str = ""
for i in features:
    equals_to_str = equals_to_str + str(i) + " + "

equals_to_str = str('Weekly_Sales ~ ') + equals_to_str
equals_to_str = equals_to_str[:-3] 
equals_to_str_promotion = equals_to_str + "Markdown1"

result_fullmodel = sm.ols(formula=equals_to_str, data = data_train).fit()
print("Model 1 without Promotion \n",result_fullmodel.summary())



unique_dept_values = master_df_new_var.Dept.unique() #getting unique values of store by department type
unique_data_string = []    #creating null list
mark = []

#creating unique list containing names of department for creating labels for the graph
for counter in range(0, len(unique_dept_values)):
    if unique_dept_values[counter] == 38 or unique_dept_values[counter] == 92 or unique_dept_values[counter] == 95:
        unique_data_string.append("Dept-" + str(unique_dept_values[counter]))
        mark.append("+")
    else:
        unique_data_string.append("")
        mark.append("o")


mean_sales = []   #creating list of mean sales
for intcounter in unique_dept_values:
    mean_sales.append(data_train[data_train["Dept_" + str(intcounter)]==1]["Weekly_Sales"].mean()) #estimating mean sales for store by department type

#creating formula for regression
equals_to_str_dept = "Weekly_Sales ~ Weekly_Sales_Lag + isHoliday + Temperature + Type_A + Type_B + Type_C+ Week_Number_50 + Week_Number_51 + " 
for intcounter in range(len(unique_dept_values)):
    if intcounter != len(unique_dept_values)-1:
        equals_to_str_dept = equals_to_str_dept + "Dept_" + str(unique_dept_values[intcounter]) + " + "
    else:
        equals_to_str_dept = equals_to_str_dept + "Dept_" + str(unique_dept_values[intcounter])

#running regression with relevant variables and all the department type to calculate which department type impacts the weekly sales the most
result_with_dpt = sm.ols(formula=equals_to_str_dept, data = data_train).fit()
result_list = result_with_dpt.tvalues #calculating t value to zero in on important parameters
result_tstat = []
 
for intcounter in range(9,90):
    result_tstat.append(result_list[intcounter]) #getting t stat value for each department and store type
 
fig, ax = plt.subplots()
ax.scatter(result_tstat, mean_sales) #making a scatter plot to highlight high performing departments
plt.xlabel('T- statistics')
plt.ylabel('Mean Weekly Sales')
plt.title('Mean weekly sales by department type')
plt.show()

for i, txt in enumerate(unique_data_string):
    ax.annotate(txt, (result_tstat[i],mean_sales[i])) #identifying the high performing graph and labelling them
    
    
    
ax= sns.barplot(x="Store", y="Weekly_Sales",  data=master_df_new_var)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
plt.tight_layout()
plt.show()

sns.barplot(x="Year", y="Weekly_Sales", hue="Type", data=master_df)

df_corr = master_df.corr()
ax=df_corr[['Weekly_Sales']].plot(kind='bar')
plt.xlabel('Attribute')
plt.ylabel('Correlation')
plt.title('Correlation of Weekly sales with other variables')
plt.tight_layout()
plt.show()
sns.heatmap(df_corr)
