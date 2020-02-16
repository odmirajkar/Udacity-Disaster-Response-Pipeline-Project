import sys
from sqlalchemy import create_engine
import pandas as pd 
import re
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
import pickle


def load_data(database_filepath):
    """
    Load the filepath and return the data
    Param
    database_filepath - sqllite file storing clean dataset, created by process_Data.py
    Returns
    X - is input data which is message sent 
    Y - categories which we need to predict 
    category_names
    """
    name = 'sqlite:///' + database_filepath
    engine = create_engine(name)
    df = pd.read_sql_table('Disasters', con=engine) 
    print(df.head())
    # X is input data which is message sent 
    # Y is 36 categories which we need to predict which are last 36 columns 
    X = df['message']
    y = df[df.columns[-36:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    """
    function to tokenize and preprocess text
    Param - text to be tokenize 
    1 normalize the text by converting it into lower case
    2. remove punctuation symboles and replace them with space
    3 tokenize it using word tokenizer
    4 remove stop words
    5 lamitize it using  
    """
    #1 normalize the text by converting it into lower case
    text=text.lower()
    #remove punctuation symboles and replace them with space
    text=re.sub('[^ a-zA-Z0-9]',' ',text)
    #tokenize it using word tokenizer
    tokens=word_tokenize(text)
    #remove stop words
    stwords=stopwords.words('english')
    tokens_without_stop=[w for w in tokens if w not in stwords]
    #lamitize using noun 
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in tokens_without_stop]
    #lamitize using verb
    clean_tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    
    return clean_tokens


def build_model():
    """
    Return Grid Search model with pipeline and Classifier
    Returns 
    model - classifier model
    
    """
    moc = MultiOutputClassifier(RandomForestClassifier())

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', moc)
        ])

    #removing 'clf__estimator__min_samples_leaf':[1,2,5], to speed up training 
    parameters = {'clf__estimator__max_depth': [5,10,None],
              'clf__estimator__max_leaf_nodes':[5,10,None]}
    
    model = GridSearchCV(pipeline, parameters)
    return model

# Get results and add them to a dataframe.
def get_classification_report(y_test, y_pred):
    """
    To get F1 score,precision for each category
    Param:
    y_test - actual y values 
    y_pred - predicted y values
    """
    for ind,cat in enumerate(y_test.keys()): 
        print("Classification report for {}".format(cat))
        print(classification_report(y_test.iloc[:,ind], y_pred[:,ind]))
    #return results

def evaluate_model(model, X_test, Y_test, category_names):
    """Print model results
    INPUT
    model -- required, estimator-object
    X_test -- required
    y_test -- required
    category_names = required, list of category strings
    OUTPUT
    None
    """
    # Get results and add them to a dataframe.
    y_pred = model.predict(X_test)
    get_classification_report(Y_test, y_pred)
    #print(classification_report(Y_test, y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    """Save model as pickle file"""
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
        #python train_classifier.py ../data/DisasterResponse.db classifier.pkl


if __name__ == '__main__':
    main()