import joblib

model = joblib.load("model.joblib")

print("Enter Data")

x1 = float(input("Precipitation: "))
x2 = float(input("Min Temperature: "))
x3 = float(input("Cloud Cover: "))
x4 = float(input("Vapour Pressure: "))
x5 = float(input("Area: "))

prediction = model.predict([[x1, x2, x3, x4, x5]])[0]

print("\nProduction =", prediction)