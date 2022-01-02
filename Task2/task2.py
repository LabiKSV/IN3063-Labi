import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def testData():
    # test 1
    x1 = np.array([-75, -48, -32, -22, -19, -10, 3, 18, 29, 33, 44, 50, 78, 90, 103])
    y1 = np.array([-80, -45, -20, -15, -10, -6, 7, 14, 25, 37, 44, 54, 73, 86, 100])
    # test 2
    x2 = np.array([-31, -11, 9, 29, 49, 69, 89, 109])
    y2 = np.array([-33, -15, 10, 30, 45, 67, 92, 113])
    # test 3
    x3 = np.array([15, 78, 157, 199, 202, 203, 250, 288])
    y3 = np.array([20, 60, 130, 180, 200, 210, 30, 250])
    return x1, y1, x2, y2, x3, y3


def linearRegressionAndPlot(x, y, title, normalize=False, applyScaler=False):
    # adjust x values
    x = x.reshape((-1, 1))

    # scaling
    if applyScaler:
        scaler = StandardScaler()
        scaler.fit(x)
        x = scaler.transform(x)

    print("X values (scaling = ", applyScaler, ")")
    print(x)
    print("Y values")
    print(y)

    linearRegressionModel = LinearRegression(normalize=normalize).fit(x, y)
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
    plt.title(title)
    plt.show()


# prepare test data
xyValues = testData()
x1 = xyValues[0]
y1 = xyValues[1]
x2 = xyValues[2]
y2 = xyValues[3]
x3 = xyValues[4]
y3 = xyValues[5]

# Linear regression with defaults
linearRegressionAndPlot(x1, y1, "Linear regression predictions - default settings (1)")
linearRegressionAndPlot(x2, y2, "Linear regression predictions - default settings (2)")
linearRegressionAndPlot(x3, y3, "Linear regression predictions - default settings (3)")

# Now again all the above, normalized
linearRegressionAndPlot(x1, y1, "Linear regression predictions - normalized (1)", normalize=True)
linearRegressionAndPlot(x2, y2, "Linear regression predictions - normalized (2)", normalize=True)
linearRegressionAndPlot(x3, y3, "Linear regression predictions - normalized (3)", normalize=True)

# Now again all the above, scaled and normalized
linearRegressionAndPlot(x1, y1, "Linear regression predictions - normalized and scaled (1)", normalize=True,
                        applyScaler=True)
linearRegressionAndPlot(x2, y2, "Linear regression predictions - normalized and scaled (2)", normalize=True,
                        applyScaler=True)
linearRegressionAndPlot(x3, y3, "Linear regression predictions - normalized and scaled (3)", normalize=True,
                        applyScaler=True)
