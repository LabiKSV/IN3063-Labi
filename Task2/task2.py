import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# test 1
x = np.array([-75, -48, -32, -22, -19, -10, 3, 18, 29, 33, 44, 50, 78, 90, 103])
y = np.array([-80, -45, -20, -15, -10, -6, 7, 14, 25, 37, 44, 54, 73, 86, 100])
x = x.reshape((-1, 1))

print("X values")
print(x)
print("Y values")
print(y)

linearRegressionModel = LinearRegression().fit(x, y)
rSquared = linearRegressionModel.score(x, y)
yPredictions = linearRegressionModel.predict(x)
print("Mean squared error: %.4f" % mean_squared_error(y, yPredictions))
print('Coefficient of determination = ', rSquared)
print('Intercept = ', linearRegressionModel.intercept_)
print('Slope = ', linearRegressionModel.coef_)
print('Predictions -> ', yPredictions, sep='\n')

# Graphically show predictions to understand outliers, accuracy, etc.
plt.scatter(x, y, color="red", label="originals")
plt.plot(x, yPredictions, color="green", linewidth=2, label="predictions")
plt.xticks(())
plt.yticks(())
plt.legend(loc="upper right")
plt.title("Linear regression predictions (1)")
plt.show()

# test 2
x = np.array([-31, -11, 9, 29, 49, 69, 89, 109])
y = np.array([-33, -15, 10, 30, 45, 67, 92, 113])
x = x.reshape((-1, 1))

print("X values")
print(x)
print("Y values")
print(y)

linearRegressionModel = LinearRegression().fit(x, y)
rSquared = linearRegressionModel.score(x, y)
yPredictions = linearRegressionModel.predict(x)

print("Mean squared error: %.4f" % mean_squared_error(y, yPredictions))
print('Coefficient of determination = ', rSquared)
print('Intercept = ', linearRegressionModel.intercept_)
print('Slope = ', linearRegressionModel.coef_)
print('Predictions -> ', yPredictions, sep='\n')

# Graphically show predictions to understand outliers, accuracy, etc.
plt.scatter(x, y, color="red", label="originals")
plt.plot(x, yPredictions, color="green", linewidth=2, label="predictions")
plt.xticks(())
plt.yticks(())
plt.legend(loc="upper right")
plt.title("Linear regression predictions (2)")
plt.show()

# test 3
x = np.array([15, 78, 157, 199, 202, 203, 250, 288])
y = np.array([20, 60, 130, 180, 200, 210, 230, 250])
x = x.reshape((-1, 1))

print("X values")
print(x)
print("Y values")
print(y)

linearRegressionModel = LinearRegression().fit(x, y)
rSquared = linearRegressionModel.score(x, y)
yPredictions = linearRegressionModel.predict(x)

print("Mean squared error: %.4f" % mean_squared_error(y, yPredictions))
print('Coefficient of determination = ', rSquared)
print('Intercept = ', linearRegressionModel.intercept_)
print('Slope = ', linearRegressionModel.coef_)
print('Predictions -> ', yPredictions, sep='\n')

# Graphically show predictions to understand outliers, accuracy, etc.
plt.scatter(x, y, color="red", label="originals")
plt.plot(x, yPredictions, color="green", linewidth=2, label="predictions")
plt.xticks(())
plt.yticks(())
plt.legend(loc="upper right")
plt.title("Linear regression predictions (3)")
plt.show()
