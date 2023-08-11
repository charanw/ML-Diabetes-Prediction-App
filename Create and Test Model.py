import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
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


# Split the data for training and testing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Print the data for visualization
print(df_src.head(20))

training_df = prepare_data(df_src)
# Print the data for visualization
print(training_df.head(20))

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
joblib.dump(scaler, "diabetes_scaler.save")


# Create the model
def create_model(training_features, y_train):
    model = LogisticRegression()
    model.fit(training_features, y_train)
    training_score = model.score(training_features, y_train)
    return model, training_score


# Test the model
def test_model(model, testing_features, y_test):
    testing_score = model.score(testing_features, y_test)
    y_predict = model.predict(testing_features)
    return testing_score, y_predict


model, training_score = create_model(training_features, y_train)
testing_score, y_predict = test_model(model, testing_features, y_test)
confusion_matrix = confusion_matrix(y_test, y_predict)
scores = [training_score, testing_score]

# Save the generated model, confusion matrix, and testing scores as python objects using joblib
joblib.dump(model, "diabetes_model.save")
joblib.dump(confusion_matrix, "diabetes_confusion_matrix.save")
joblib.dump(scores, "training_and_testing_scores.save")
