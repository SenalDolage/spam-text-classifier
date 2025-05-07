import joblib
import sys

# load the saved machine learning model and TF-IDF vectorizer 
def load_model():
    model = joblib.load("spam_detector_model.pkl")
    vectorizer = joblib.load("spam_detector_vectorizer.pkl")
    return model, vectorizer


def predict(text):
    model, vectorizer = load_model()
    # transform the input text into numbers using the vectorizer.
    text_transformed = vectorizer.transform([text])
    # runs prediction using the model.
    prediction = model.predict(text_transformed)
    # returns the result as "Spam" or "Not Spam".
    return "Spam" if prediction[0] == 1 else "Not Spam"


text = sys.argv[1]
prediction = predict(text)
print("This message is: ", prediction)
