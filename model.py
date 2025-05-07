import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib


def load_data():
    url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
    # Reads the data from a .tsv (tab-separated values) file online.
    data = pd.read_csv(url, sep="\t", header=None, names=["label", "text"])
    # convert the labels
    data["label"] = data["label"].map({"ham": 0, "spam": 1})
    return data


def train_model(data):
    x = data["text"]
    y = data["label"]
    #  tool to convert the text messages into numbers based on word importance
    vectorizer = TfidfVectorizer()
    x = vectorizer.fit_transform(x)
    # split the data into training (80%) and testing (20%) sets.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    # creates a Naive Bayes model
    model = MultinomialNB()
    # train the model using the training data.
    model.fit(x_train, y_train)
    # use the model to predict the labels (spam or ham) for the test data.
    y_pred = model.predict(x_test)
    # measure how well the model did
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    # saves the trained model and vectorizer to .pkl files to reuse them later (without retraining).
    joblib.dump(model, "spam_detector_model.pkl")
    joblib.dump(vectorizer, "spam_detector_vectorizer.pkl")
    return accuracy, report


data = load_data()
accuracy, report = train_model(data)
print("Accuracy", accuracy)
print("Classification report\n", report)
