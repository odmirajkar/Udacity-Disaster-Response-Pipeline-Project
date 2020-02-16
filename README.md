# Disaster Response Pipeline Project

### Summary of project
This udacity data science nano degree project. Purpose of this project is to correctly classify the disaster request into 3 predefined categories.
This porject has three parts
	- ETL pipeline  this is to extract data from csv file, clean it, and store it in sqlite database
    - ML pipeline  I used multi lable classifier to correctly classify message in one or more than one of predefined 36 categories 
    - web application Flask based web application to display result

### Files included in project

- app folder is for web application
	- it has run.py main file to run web application
    - template folder contains html tempates
- model folder is for machine learning pipeline
	- train_classifier uses grid search & random forest algorithm to train the model for multi lable classifier. it has accuracy of 94%, I got similar accurcy when I used Adaboost algorithm. Notebook for ML pipeline has implementation of AdaBoost algorithm 
- data folder is for csv data and ETL pipeline
	- process_data.py is for ETL pipeline. 
- notebook folder is for notebooks used to create this project
    
### Libriries used
1. NLTK - for processing raw text data.
	- It is used to remove Stopwords, tokenize & Lemmatize the text
2. sklearn 
	- MultiOutputClassifier is used as Multi Lable classifier, RandomForest is used as a classifier for each lable by MultiOutputClassifier
    - classification_report is used to output F1 score for each category
3. Flask 
	- To create webapplication
4. plotly
	- To visualize ghraph on webpage
5. pandas
	- to process csv data files
6. sqlalchemy
	- to store and retrieve pandas dataframe in sqlite database


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### screenshots:
