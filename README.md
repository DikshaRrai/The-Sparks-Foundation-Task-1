# The-Sparks-Foundation-Task-1
Prediction using supervised MaName: Diksha Rai
Internship: Data Science and Business Analytics
Organisation: The Sparks Foundation
Step 1: Importing the required libraries and dataset.
Starting with importing relevant libraries that will help us in carrying out the given task.

In [1]:
import pandas as pd
import numpy as nm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
%matplotlib inline
Loading the required dataset

In [2]:
data=pd.read_csv('C:\\Users\\Ashray Wadhwa\\Desktop\\The Sparks Foundation\\Dataset.csv')
print('Data Imported Successfully')
data
Data Imported Successfully
Out[2]:
Hours (X)	Scores (Y)
0	2.5	21
1	5.1	47
2	3.2	27
3	8.5	75
4	3.5	30
5	1.5	20
6	9.2	88
7	5.5	60
8	8.3	81
9	2.7	25
10	7.7	85
11	5.9	62
12	4.5	41
13	3.3	42
14	1.1	17
15	8.9	95
16	2.5	30
17	1.9	24
18	6.1	67
19	7.4	69
20	2.7	30
21	4.8	54
22	3.8	35
23	6.9	76
24	7.8	86
Step 2: Visual presentation of the data imported.
In [3]:
data.plot(x='Hours (X)', y='Scores (Y)', style='^', markerfacecolor='red', markeredgecolor="red")  
plt.title('Hours studied vs Percentage scored')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
print('Diagram Loaded Successfully')
plt.show()
Diagram Loaded Successfully

We can clearly see the presence of a positive correlation between the variables - Hours Studied and Percentage Scored. Above stated diagram shows this relation graphically. Now, we'll explain it numerically with the use of an appropriate function in Python.
In [4]:
data.corr()
Out[4]:
Hours (X)	Scores (Y)
Hours (X)	1.000000	0.976191
Scores (Y)	0.976191	1.000000
A high positive correlation of 0.976191 is seen between the variables - 'Hours Studied' and 'Percentage Scored'. Thus, validating the information that was depicted in the above diagram. So, a linear relationship exists between these two variables.

Step 3: Distinguishing the data.
Now first we'll distinguish the data into independent variable and deppendent variable, which means we'll assign the independent variable tag to 'Hours Studied' and dependent variable tag to 'Percentage scored' becasue our task demands the effect of hours of study on percentage scored.

In [5]:
X = data.iloc[:, :-1].values #Independent Variable.
y = data.iloc[:, 1].values #Dependent Variable.
Since, the task assigned to us deals with predicting using supervised ML, we'll like to bring in a model from sklearn (or scikit-learn) which is nothing but a Python library that offers various features for data processing that can be used for classification, clustering, and model selection.

Before proceeding further, I would like to give a brief about what one means by Supervised ML.

Supervision means to oversee or direct a certain activity and make sure it is done correctly. In this type of learning, the machine learns under guidance. Just like at schools, our teachers guided us and taught us similarly in Supervised Learning, machines are taught by feeding them label data and explicitly telling them this is the input and this is exactly how the output must look.

If you have one dataset, you'll need to split it by using the Sklearn train_test_split function first.

train_test_split is a function in Sklearn model selection for splitting data arrays into two subsets: for training data and for testing data. With this function, you don't need to divide the dataset manually. By default, Sklearn train_test_split will make random partitions for the two subsets. However, you can also specify a random state for the operation.</b>

In [6]:
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
In the above model we divided the model into 70:30 ratio where 70% of the dataset will be used as a training set while 30% has been allocated for the testing purpose.

Step 4: Training the algorithm of the model.
In order to train the model in linear regression, we'll import the linear regression model from scikit-learn library.

In [7]:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print("Training complete.")
Training complete.
In [8]:
line = regressor.coef_*X+regressor.intercept_
# Plotting for the test data
plt.scatter(X, y, color = 'red')
plt.plot(X, line, color='green')
plt.title('Hours studied vs Percentage scored')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
print('Diagram Loaded Successfully')
plt.show()
Diagram Loaded Successfully

Above stated line is nothing but the best fit line which the model analyzed based on the 70% of the data set that was allocated it. The line is of the form, Y = (Beta 1) + (Beta 2)*X which is nothing but the equation of a simple linear regression.

Step 5: Testing the model based on the allocated dataset for testing.
Using the same dataset for both training and testing leaves room for miscalculations, thus increasing the chances of inaccurate predictions.

The train_test_split function allows you to break a dataset with ease while pursuing an ideal model. Also, one should keep in mind that your model should not be overfitted or underfitted.</b>

In [9]:
print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting Scores.
[[1.5]
 [3.2]
 [7.4]
 [2.5]
 [5.9]
 [3.8]
 [1.9]
 [7.8]]
In [10]:
regressor.coef_
Out[10]:
array([9.78856669])
In [11]:
regressor.intercept_
Out[11]:
2.370815382341881
The above stated function displays the 30% of dataset allotted for the testing of the model.

In [12]:
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df
Out[12]:
Actual	Predicted
0	20	17.053665
1	27	33.694229
2	69	74.806209
3	30	26.842232
4	62	60.123359
5	35	39.567369
6	24	20.969092
7	86	78.721636
Comparing the 'Actual' value of Percentage Scored with the'predicted' value of Percentage Scored. Though our model is not very precise, the predicted percentages are close to the actual ones. Moving further, our task asks us to predict the value of percentage scored when number of hours studied is 9.25 hours. So, we can do that too using our model in the following manner.

In [13]:
hours = [[9.25]]
pred_new = regressor.predict(hours)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred_new[0]))
No of Hours = [[9.25]]
Predicted Score = 92.91505723477056
Step 6: Evaluating the model.
The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For regression algorithms, three evaluation metrics are commonly used:

Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
Let's find the values for these metrics using our test data. Execute the following code:</b>

In [14]:
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', nm.sqrt(metrics.mean_squared_error(y_test, y_pred)))
Mean Absolute Error: 4.419727808027652
Mean Squared Error: 22.96509721270043
Root Mean Squared Error: 4.792191274636315
You can see that the value of root mean squared error is 4.79, which is less than 10% of the mean value of the percentages of all the students i.e. 51.48%. This means that our algorithm did a satisfactory job.

chine Learning
My first repository on Github
