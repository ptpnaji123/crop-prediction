import joblib
import pandas as pd

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


df = pd.read_csv("dataset.csv")

X = df[['Precipitation'	,'Min_Temp'	,'Cloud_Cover'	,'Vapour_pressure'	,'Area']] 
Y = df['Production']
 
x_train,x_test,y_train,y_test=train_test_split(X.to_numpy(),Y.to_numpy(), test_size=0.2, random_state=1)

model=linear_model.LinearRegression()
model.fit(x_train,y_train)


# prediction
y_pred=model.predict(x_test)

# Coefficients
print('Coefficients: ', model.coef_)

# R-squared score
print('MSE: ', mean_squared_error(y_test, y_pred))

# Save Model
joblib.dump(model, "model.joblib")