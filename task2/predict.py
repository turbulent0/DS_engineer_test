import joblib
import pandas as pd

if __name__ == '__main__':
    df_test = pd.read_csv('data/hidden_test.csv', usecols=['6'])
    df_test['6_squared'] = df_test['6'] ** 2
    model_filename = 'models/linear_regression_model.joblib'
    # Load the model from the file
    loaded_model = joblib.load(model_filename)
    # predict on test
    df_test['y_pred'] = loaded_model.predict(df_test[['6_squared']])
    df_test[['y_pred']].to_csv('prediction.csv', index=False)