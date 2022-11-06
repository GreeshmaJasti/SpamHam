import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# def speak(str):
#     from win32com.client import Dispatch
#     spk = Dispatch("SAPI.SpVoice")
#     spk.Speak(str)

# def talk():
#     speak("This Project helps you to check your mail content is either SPAM or HAM")
#     speak("Enter the content in the provided Text-Area")

app = Flask(__name__)

@app.route('/')
def home():
    #talk()
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    df = pd.read_csv("spam.csv", encoding="latin-1")
    print(df.shape)
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df = df.drop_duplicates()
    # Features and Labels
    df['label'] = df['class'].map({'ham': 0, 'spam': 1})
    X = df['message']
    y = df['label']
    # Extract Feature With CountVectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    #print(X.toarray())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    # Naive Bayes Classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    print('Accuracy = ',clf.score(X_test, y_test)*100,'%')
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        # if my_prediction[0] == 0:
        #     speak("This is Not A Spam Email")
        # else:
        #     #speak("This is A Spam Email")
    return render_template('index.html',prediction=my_prediction)


if __name__ == '__main__':
    app.run()
