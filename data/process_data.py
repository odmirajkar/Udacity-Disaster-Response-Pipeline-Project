import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load dataframe from filepaths
    Params
    messages_filepath -- str, link to message file
    categories_filepath -- str, link to categories file
    returns
    df - pandas DataFrame
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,on='id')
    return df
    


def clean_data(df):
    """Clean data included in the DataFrame and transform categories part
    Params
    df -- pandas DataFrame created by load_data function
    Returns
    df -- cleaned pandas DataFrame
    """
    categories = df['categories'].str.split(';',expand=True)
    row = categories.loc[0]
    category_colnames = []
    for category in row:
        category_colnames.append(category[:-2])
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] =  categories[column].astype(int)    
    # drop the original categories column from `df`
    df.drop('categories',axis=1,inplace=True)   
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    return df
    

def save_data(df, database_filename):
    """
    Saves DataFrame (df) to database path
    param
    df - pandas dataframe, cleaned by clean_data function
    database_filename - path of database file to store cleaned dataframe
    """
    name = 'sqlite:///' + database_filename
    engine = create_engine(name)
    df.to_sql('Disasters', engine, index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()