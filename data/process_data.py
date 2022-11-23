import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    A function to read in a category & a message data file, merge the two and re-shape the categories into separate columns.
    
    INPUT:
    messages_filepath - A string filepath to the messages csv file
    categories_filepath - A string filepath to the categories csv file
    
    OUTPUT:
    df - a dataframe containing the message, message metadata, and a binary column for each category
    """
    
    #read in messages file
    messages = pd.read_csv(messages_filepath)
    
    #read in categories file
    categories = pd.read_csv(categories_filepath)
    
    #merge the two datasets on the id column
    df = messages.merge(categories, on ='id')
    
    #split the strings in the categories column, expanding them into a new column for each category
    categories = df.categories.str.split(pat = ';', expand = True)
    
    #select the first row and iterate over it to obtain the column names for the different category columns, dropping the final 2 digits as they are '-' and then '0' or '1'
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # iterarte over each column, extract the final digit (0/1) as the column value, and convert it to numeric
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = categories[column].astype("int64")
    
    # drop the original category column and concatenate the new category columns onto the dataframe 
    df.drop(columns = "categories", inplace = True)
    df = pd.concat([df,categories], axis = 1)
    
    return df
    pass


def clean_data(df):
    """ A function to remove any duplicates from the data
    INPUT: df - A dataframe of messages & categories as prepared in the load_data() function
    OUTPUT: df - A dataframe with duplicates removed
    """
    # use the pandas drop-duplicates method as a first filter
    df.drop_duplicates(inplace = True)
    
    # iterate over rows to remove any duplicate ids
    ids = []
    for row in df.index:
        if df.loc[row,'id'] in ids:
            df.drop(index = row, inplace = True)
        else:
            ids.append(df.loc[row, 'id'])
            
    # Inspect the data to check whether there are any categories that have less than 2 cases in the dataset as that would corrupt any model training
    for i in df.columns:
        if str(df[i].dtype) == "int64":
            if df[i].sum(axis = 0) < 2:
                df.drop(columns = i, inplace = True)
                
    return df
    pass


def save_data(df, database_filename):
    """
    A function to upload a dataframe to a database using sqlalchemy
    
    INPUT:
    df - a dataframe
    database_filename - the string filepath for the database used to create the engine
    
    OUTPUT:
    None
    """
    #create the engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists = "replace")
    pass  


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
