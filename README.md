# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculus output value.Next define the neural network model in three layers.First layer have four neurons and second layer have three neurons,third layer have two neurons.The neural network model take input and produce actual output using regression.

## Neural Network Model

![neural net](https://user-images.githubusercontent.com/75235402/187143493-a5eaebb1-4666-4c94-9a08-f9250f7893ca.png)

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM

# Importing Required packages
```python3

from google.colab import auth
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import gspread
import pandas as pd
from google.auth import default
import pandas as pd

# Authenticate the Google sheet

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('dl model').sheet1

# Construct Data frame using Rows and columns

rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head()
df=df.astype({'X':'float'})
df=df.astype({'Y':'float'})
X=df[['X']].values
Y=df[['Y']].values

# Split the testing and training data

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=50)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_t_scaled = scaler.transform(x_train)
x_t_scaled

# Build the Deep learning Model

ai_brain = Sequential([
    Dense(3,activation='relu'),
    Dense(2,activation='relu'),
    Dense(1,activation='relu')
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=x_t_scaled,y=y_train,epochs=2000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()

# Evaluate the Model

scal_x_test=scaler.transform(x_test)
ai_brain.evaluate(scal_x_test,y_test)
input=[[105]]
input_scaled=scaler.transform(input)
ai_brain.predict(input_scaled)
```

## Dataset Information

![image](https://user-images.githubusercontent.com/75235402/187143647-503a17eb-9f89-494b-b1c9-700a98749308.png)


## OUTPUT

### Training Loss Vs Iteration Plot

![download](https://user-images.githubusercontent.com/75235402/187143425-1dac030a-de6f-4d31-8898-d52a3355dab0.png)


### Test Data Root Mean Squared Error

![image](https://user-images.githubusercontent.com/75235402/187143841-9784e6a9-92b6-4191-89b6-9ecd6128f889.png)


### New Sample Data Prediction

![image](https://user-images.githubusercontent.com/75235402/187143891-ca505882-400d-41dc-9328-3b8aabed3b63.png)


## RESULT
Thus the Neural network for Regression model is Implemented successfully.
