import joblib
import pandas as pd
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Get data source
df_src = pd.read_csv("diabetes_prediction_dataset.csv", low_memory=False)


# Prepare data

def prepare_data(df):
    # Use one hot encoding for gender
    df['Male'] = df['gender'].map(lambda x: 1 if x == 'Male' else 0)
    df['Female'] = df['gender'].map(lambda x: 1 if x == 'Female' else 0)
    df['Other'] = df['gender'].map(lambda x: 1 if x == 'Other' else 0)

    # Use one hot encoding for smoking history. Merge similar smoking answers into "Formerly_Smoked".
    df['Has_Never_Smoked'] = df['smoking_history'].map(lambda x: 1 if x == 'never' else 0)
    df['Formerly_Smoked'] = df['smoking_history'].map(lambda x: 1 if x == ('ever' or 'former' or 'not current') else 0)
    df['Currently_Smokes'] = df['smoking_history'].map(lambda x: 1 if x == 'current' else 0)

    return df[
        ['age', 'Male', 'Female', 'Other', 'Has_Never_Smoked', 'Formerly_Smoked', 'Currently_Smokes', 'hypertension',
         'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'diabetes']]


# Settings for pandas console output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print the data for visualization
print(df_src.head(20))

training_df = prepare_data(df_src)
# Print the data for visualization
print(training_df.head(20))

# Split the data for training and testing
features = training_df[
    ['age', 'Male', 'Female', 'Other', 'Has_Never_Smoked', 'Formerly_Smoked', 'Currently_Smokes', 'hypertension',
     'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']].values
diabetes = training_df['diabetes'].values
x_train, x_test, y_train, y_test = train_test_split(features, diabetes, test_size=0.3)

# Standardize the data
scaler = StandardScaler()
training_features = scaler.fit_transform(x_train)

training_features_df = pd.DataFrame(data=training_features,
                                    columns=['age', 'Male', 'Female', 'Other', 'Has_Never_Smoked', 'Formerly_Smoked',
                                             'Currently_Smokes', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
                                             'blood_glucose_level'])
# Print the data for visualization
print(training_features_df.head(20))

testing_features = scaler.transform(x_test)


# Create the model
def create_model(training_features):
    model = LogisticRegression()
    model.fit(training_features, y_train)
    return model


# Test the model
def test_model(model, training_features, testing_features, y_train, y_test):
    training_score = model.score(training_features, y_train)
    testing_score = model.score(testing_features, y_test)
    y_predict = model.predict(testing_features)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_predict)
    accuracy = accuracy_score(y_test, y_predict)
    precision = precision_score(y_test, y_predict)
    recall = recall_score(y_test, y_predict)
    f1 = f1_score(y_test, y_predict)
    return {'confusion matrix': confusion_matrix, 'training score': training_score, 'testing score': testing_score,
            'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}


# Call the functions to create the model and run the tests
model = create_model(training_features)
test_results = test_model(model, training_features, testing_features, y_train, y_test)

# Save the scaler, generated model, and test results as python objects using joblib
joblib.dump(scaler, "diabetes_scaler.save")
joblib.dump(model, "diabetes_model.save")
joblib.dump(test_results, "test_results.save")
