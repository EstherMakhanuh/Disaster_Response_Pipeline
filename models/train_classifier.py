import sys
import os
import nltk
nltk.download(['punkt','stopwords','wordnet'])

# import libraries
import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import pickle
import warnings
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    """
    Load the clean data from the SQL database
    Args:
    database_filepath
    Returns:
    X: features dataframe
    y: target dataframe
    """
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = os.path.basename(database_filepath).split('.')[0]
    df = pd.read_sql_table(table_name,con=engine)
    
    X = df.message.values
    
    ## getting the categorical values 
    categorical_col = df.columns[4:]
    y = df[categorical_col]
    category_names = categorical_col
    
    return X,y,category_names


def tokenize(text):
    
     """
    Process the raw texts includes:
        1. replace any urls with the string 'urlplaceholder'
        2. remove punctuation
        3. tokenize texts
        4. remove stop words
        5. normalize and lemmatize texts
    Args:
    text (str): raw texts
    Return: a list of clean words in their roots form
    """
     url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
     detected_urls = re.findall(url_regex,text)
    # replace each url in text strings with placeholder
     for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')

     # remove puntuation characters
     text = re.sub(r"[^a-zA-Z0-9]", " ",text)

     #tokenizing the text
     tokens = word_tokenize(text)
     token_list =[]
     #initializing lemmatizer
     lemmatizer = WordNetLemmatizer()

        # remove stop words
     for tok in tokens:
            if tok not in stopwords.words("english"):
                token_list.append(tok)

     clean_tokens = []
     for tok in token_list:
         #lemmatizing, case normalization and removing leading and trailing whitespace
              clean_tok = lemmatizer.lemmatize(tok).lower().strip()
              clean_tokens.append(clean_tok)   

     return clean_tokens

# A custom transformer which will identify buzzwords signaling disaster
class DisasterWordExtractor(BaseEstimator, TransformerMixin):

    def disaster_words(self, text):
        """
        INPUT: text - string, raw text data
        OUTPUT: bool -bool object, True or False
        """
        # list of words that are commonly used during a disaster event
        disaster_words = ['food','hunger','hungry','starving','water','drink','eat','thirsty',
                 'need','hospital','medicine','medical','ill','pain','disease','injured','falling',
                 'wound','blood','dying','death','dead','aid','help','assistance','cloth','cold','wet','shelter',
                 'hurricane','earthquake','flood','whirlpool','live','alive','child','people','shortage','blocked',
                 'trap','rob','gas','pregnant','baby','cry','fire','blizard','freezing','blackout','drought',
                 'hailstorm','heat','pressure','lightning','tornado','tsunami']

        # lemmatize the buzzwords
        lemmatized_words = [WordNetLemmatizer().lemmatize(w, pos='v') for w in disaster_words]
        # Get the stem words of each word in lemmatized_words
        stem_disaster_words = [PorterStemmer().stem(w) for w in lemmatized_words]

        # tokenize the input text
        clean_tokens = tokenize(text)
        for token in clean_tokens:
            if token in stem_disaster_words:
                return True
        return False

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        X_disaster_words = pd.Series(X).apply(self.disaster_words)
        return pd.DataFrame(X_disaster_words)


def build_model():
    """
    A pipeline that includes text processing steps and a classifier (random forest).
    Note: GridSearch wasn't used as the accuracy was better without it
    """
    # instantiate the pipeline
    model = Pipeline([
        ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
        ('vect',CountVectorizer(tokenizer=tokenize)),
        ('tfidf',TfidfTransformer())])),
        ('disaster_words',DisasterWordExtractor())
        ])),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
        ])

  
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    This function evaluates the model performance for each category of
    Args:
        model: the classification returned with optimized parameters
        X_test: feature variable from test set
        y_test: target variable from test set
    OUTPUT
        Classification report and accuracy score
    """
    # predict
    y_pred = model.predict(X_test)

    # classification report
    print('Classification Report')
    i =0
    for col in Y_test:
        print('Feature {}:{}'.format(i+1,col))
        print(classification_report(Y_test[col],y_pred[:,i]))
        i=i+1
    
    # accuracy score
    accuracy = (y_pred == Y_test).mean()
    print('The model accuracy score is {}'.format(accuracy))
    

def save_model(model, model_filepath):
    """ This function saves the pipeline to local disk """

    pickle.dump(model, open(model_filepath, 'wb'))


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