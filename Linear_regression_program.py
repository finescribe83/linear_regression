import pandas as pd
import numpy as np
from sklearn.linear_model import  LinearRegression
from sklearn.model_selection import train_test_split
from statistics import mean

#setting the date as the index since it is located in 3rd column from left
#data is with saved as AP.csv which is available in the same directory as the code.
df = pd.read_csv('AP.csv', parse_dates =[0], index_col=[2])


df = df[['Price']]


#number of days to be forecasted
forecast_out = 30

#creating column for another variable

df['Prediction']=df[['Price']].shift(-1)


#create independent data set (x)
#Convert the data frame to a numpy array

X = np.array(df.drop(['Prediction'],1)) #it will remove prediction from np array

#Remove the last forecast_out rows
X=X[:-forecast_out]


##create dependent variable (y)
#convert the data frame to a numpy array (all of the values including the NaN's

y=np.array(df['Prediction'])

#get all of the y values except the last n rows

y=y[:-forecast_out]


#split the data into 80% training and 20% testing

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Create and train Linear regression model

lr = LinearRegression()
#training using the linear regression model
lr.fit(x_train,y_train)
lr_confidence =  lr.score(x_test,y_test)
print("Our confidence score for linear regression is: ", lr_confidence*100, "%")


#Set forecast to the last 30 rows from the original data set

x_forecast=np.array(df.drop(['Prediction'],1))[-forecast_out:]



#print the predictions for the next n days for linear regression model

lr_prediction = lr.predict(x_forecast)


#this part will create simulation 10,000 times based on sim_rate count
sim_rate = 10000
gain_rate =[]

while sim_rate>0:
  
  pred_price_list = []
  pred_array = lr.predict([[lr_prediction[29]]])
  pred_price_list.append(pred_array[0])
  proj_day=30

  while proj_day>0:
    new_proj = lr.predict([[pred_price_list[(len(pred_price_list)-1)]]])
    pred_price_list.append(new_proj[0])
    proj_day-=1

  

  proj_gain = (( pred_price_list[(len(pred_price_list)-1)] - pred_array)/pred_array)*100

  gain_rate.append(proj_gain[0])
  sim_rate-=1

print("Average projected gain will be at : ",  mean(gain_rate), " %")

print("The following are the proejected price (30 days trading):","\n", pred_price_list)




