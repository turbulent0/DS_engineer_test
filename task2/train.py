import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pandas as pd
from math import sqrt

if __name__ == '__main__':

    # we have strict quadratic relation between feature 6 and target. So we can use feature engineering and LR.

    df = pd.read_csv('data/df_train.csv')

    # Split the data into training and validating sets
    X_train, X_val, y_train, y_val = train_test_split(df[['6_squared']], df['target'], test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the trained model to a file
    model_filename = 'models/linear_regression_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Trained model saved to '{model_filename}'")

    train_pred = model.predict(X_train[['6_squared']])
    val_pred = model.predict(X_val[['6_squared']])
    # Evaluate the model
    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)
    print(f"Train R^2 Score: {train_score:.4f}")
    print(f"Val R^2 Score: {val_score:.4f}")
    print(f'Train RMSE:  {sqrt(mean_squared_error(y_train, train_pred)):.4f}')
    print(f"Val RMSE: {sqrt(mean_squared_error(y_val, val_pred)):.4f}")

    # # cross val
    # cross_val_scores = cross_val_score(model, X_train, y_train, cv=5)  # 5-fold cross-validation
    # average_cross_val_score = np.mean(cross_val_scores)
    # print(average_cross_val_score)

    # # Predict using the model
    # y_pred = model.predict(X_val)

    #  check residuals distribution
    # (y_pred - y_val).hist()



    # # Plot the original data and the regression line
    # plt.scatter(X_val['6_squared'], y_val, label='Original data')
    # plt.plot(X_val['6_squared'], y_pred, color='red', label='Regression line')
    # plt.xlabel('X')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()


  # # for complex non-linear relation you can use neural net
    # import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras import layers

    # # Generate synthetic data
    # x = np.linspace(-10, 10, 1000)
    # y = x ** 2

    # # Define the neural network
    # model = keras.Sequential([
    #     layers.Input(shape=(1,)),  # Input layer
    #     layers.Dense(32, activation='relu'),  # Hidden layer 2
    #     layers.Dense(32, activation='relu'), 
    #     layers.Dense(1)  # Output layer
    # ])

    # # Compile the model
    # model.compile(optimizer='adam', loss='mean_squared_error')

    # # Train the model
    # model.fit(x, y, epochs=100, batch_size=32)

    # # Evaluate the model
    # loss = model.evaluate(x, y)
    # print("Mean Squared Error:", loss)

    # # Predict using the trained model
    # y_pred = model.predict(x)

    # # Plot the original data and the predicted values
    # import matplotlib.pyplot as plt

    # plt.scatter(x, y, label='Original data')
    # plt.plot(x, y_pred, color='red', label='Predicted curve')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.legend()
    # plt.show()
