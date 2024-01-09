import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

class User:
    def __init__(self, first_name, last_name):
        self.first_name = first_name
        self.last_name = last_name

class ChatBot:
    def __init__(self, data_file='sequences.csv'):
        self.df = pd.read_csv(data_file)
        self.model = None
    
    def preprocess_data(self):
        self.df = self.df.dropna(subset=["Input Dialog", "Output Dialog"])
        columns_to_remove = self.df.columns[self.df.T.duplicated(keep='first')]
        self.df = self.df.drop(columns=columns_to_remove)

        X = self.df["Input Dialog"].values
        y = self.df["Output Dialog"].values
        return X, y

    def train(self):
        X, y = self.preprocess_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = make_pipeline(CountVectorizer(), MultinomialNB())
        model.fit(X_train, y_train)
        print()

        y_pred = model.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy * 100:.2f}%")

        self.model = model
        print("Training completed successfully.")

    def get_response(self, user_input):
        prediction = self.model.predict([user_input])
        return prediction[0]

user = User("John", "Doe")
chatbot = ChatBot()

try:
    chatbot.train()
    user_input = "Chandler!!! Chandler!!! Chandler, I saw what you were doing through the window! Chandler, I saw what you were doing to my sister! Now get out here!"
    response = chatbot.get_response(user_input)
    print("Response:", response)
except Exception as e:
    print(f"Error: {str(e)}")
