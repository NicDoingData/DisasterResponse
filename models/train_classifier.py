import sys
import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
nltk.download(['punkt','wordnet', 'stopwords'])

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table(table_name = database_filepath, con = engine)
    X = df.iloc[:,1]
    Y = df.iloc[:,4:]
    category_names = Y.columns
    return X, Y, category_names
    pass


def tokenize(text):
    #normalize text
    text = text.lower()
    #remove non-letter symbols
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    #tokenize
    text = word_tokenize(text)
    #remove stop words
    text = [word for word in text if word not in stopwords.words('english')]
    #lemmatize
    text = [WordNetLemmatizer().lemmatize(word) for word in text]
    
    return text
    pass


def build_model():
    model = Pipeline([
        ('CountVect', CountVectorizer(tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(estimator = RandomForestClassifier()))
    ])
    return model
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(data = Y_pred, index = X_test.index, columns = Y_test.columns)
    columns = Y_test.columns

    for col in columns:
        print(col)
        print(classification_report(y_true = Y_test[col], y_pred = Y_pred_df[col]))
        pass


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        classifier = pickle.dump(model, f)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()