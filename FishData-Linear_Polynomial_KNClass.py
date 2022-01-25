#!/usr/bin/env python
# coding: utf-8

# In[1]:


#part1 #importing

import pandas as pd
needed_data = pd.read_csv(r"/Users/david/Desktop/fish.csv")
print(needed_data)


# In[2]:


#part1 #a
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
needed_data_encoded = ordinal_encoder.fit_transform(needed_data)


# In[3]:


#part1 #a #continuation
from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
needed_data_1hot = cat_encoder.fit_transform(needed_data)
needed_data_1hot.toarray()


# In[4]:


#part1 #b
columns = needed_data[['Length1','Length2','Length3','Height','Width']].values.astype(float)

columns

normalized = (columns-columns.min())/(columns.max()-columns.min())

normalized


# In[5]:


#part3 #a
#randomising the order of the rows
import pandas as pd
import numpy as np

shuffled_data = pd.DataFrame(needed_data)

shuffled_data = shuffled_data.sample(frac=1)

print(shuffled_data)




# In[6]:


#part3 #a #continuation

# from sklearn.datasets import shuffled_data
shuffled_data
df = pd.DataFrame(shuffled_data, columns = ['Species','Weight','Length1','Length2','Length3','Height','Width'])


# Selecting the features
features = ['Weight','Length1','Length2','Length3','Height', 'Width']
x = df[features]

# Target Variable
y = df['Species']


from sklearn.model_selection import train_test_split
#Splitting the dataset into the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.40, random_state = 25 )


# In[7]:


#part3 #b

from sklearn.linear_model import SGDClassifier

# Fitting SGD Classifier to the Training set
model = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)
model.fit(x_train, y_train)

# Predicting the results
y_pred = model.predict(x_test)


# In[11]:


#part3 #b #continuation

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_score

# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)

# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)

# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print('SGD Classifier Accuracy of the model: {:.2f}%'.format(accuracy*100))



# In[6]:





# In[8]:


#part3 #c

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

x, y = make_classification(n_samples=160, n_features=12, 
                            n_classes=4, n_clusters_per_class=1)

xtrain, xtest, ytrain, ytest=train_test_split(x, y, test_size=0.40)

knc = KNeighborsClassifier(n_neighbors=50)
print(knc)

knc.fit(xtrain, ytrain)

score = knc.score(xtrain, ytrain)
print("Training score: ", score)

ypred = knc.predict(xtest)
cm = confusion_matrix(ytest, ypred)
print(cm)

cr = classification_report(ytest, ypred)
print(cr)


# In[9]:


#par2 #a
# Selecting the features
features = ['Species','Length1','Length2','Length3','Height', 'Width']
a = df[features]
# Target Variable
b = df['Weight']

a = shuffled_data.iloc[:,:-1].values
b=shuffled_data.iloc[:,:-1].values

from sklearn.model_selection import train_test_split
# Splitting the dataset into the training and test set
a_train, a_test, b_train, b_test = train_test_split(a, b, test_size = 0.40, random_state = 25 )


# In[10]:


#part2 #b

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np


np.set_printoptions(suppress=True)
labelencoder_exp = LabelEncoder()
shuffled_data['Species']=labelencoder_exp.fit_transform(shuffled_data['Species'])

ct = ColumnTransformer([("Species", OneHotEncoder(), [0])], remainder = 'passthrough')
shuffled_data_encoded = ct.fit_transform(shuffled_data)


# In[11]:


#part2 #b #continuation

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

from sklearn.linear_model import LinearRegression

labelencoder_exp = LabelEncoder()
shuffled_data['Species']=labelencoder_exp.fit_transform(shuffled_data['Species'])

ct = ColumnTransformer([("Species", OneHotEncoder(), [0])], remainder = 'passthrough')
shuffled_data_encoded = ct.fit_transform(shuffled_data)

reg=LinearRegression()
#random_fish_df_encoded = random_fish_df_encoded.drop('Weight',axis='columns')
a_train, a_test = train_test_split(shuffled_data_encoded, test_size=0.4)
b_train, b_test = train_test_split(needed_data.Weight, test_size=0.4)
a=shuffled_data_encoded
y=needed_data.Weight

reg_pred = reg.fit(a_train,b_train)


# In[12]:


#part2 #b #continuation
print(reg.coef_)


# In[13]:


#part2 #b #continuation
b_pred = reg.predict(a_test)

from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(b_test, b_pred))
print('Mean Squared Error:', metrics.mean_squared_error(b_test, b_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(b_test, b_pred)))


# In[ ]:


#part2 #c 
#polynomial regression

from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=1, include_bias=False)
a_train_poly = poly_features.fit_transform(a_train)
a_test_poly = poly_features.fit_transform(a_test)


# In[25]:





# In[26]:





# In[29]:


#part2 #c #continuation

poly_model =LinearRegression()
#print(x_train_poly)
#print(type(y_train))
b_train = b_train.to_numpy(dtype='float')
b_train = b_train.reshape(-1,1)
b_test = b_test.to_numpy(dtype='float')
b_test = b_test.reshape(-1,1)
#print(y_train)
#y_train = y_train.to_numpy(dtype='float')

poly_model.fit(a_train_poly, b_train)


# In[30]:


poly_model.coef_


# In[31]:


poly_model.intercept_


# In[32]:


poly_train_pred = poly_model.predict(a_train_poly)
poly_test_pred = poly_model.predict(a_test_poly)


# In[33]:


print('Root Mean Squared training Error:', np.sqrt(metrics.mean_squared_error(b_train, poly_train_pred)))
print('Root Mean Squared test Error:', np.sqrt(metrics.mean_squared_error(b_test, poly_test_pred)))


# In[ ]:




