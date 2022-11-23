## General Project Description
This project provides code for a web app classifying disaster response messages.
![image](https://user-images.githubusercontent.com/110040452/203553405-4e2f6176-0be8-4a35-bb8f-4a4c6768a338.png)

## Installations

The code was written using python 3 and html. Any other necessary libraries are being loaded / installed in the code.

## Project Motivation

The project was completed as part of the Udacity Data Science Nanodegree, making use of pre-provided template files and following the instructions which can be found at the bottom of this file.

## File Descriptions

app - folder containing all necessary html & python files to run the web-app

data - folder containing: A python file to process the data from csv input files and load them into a database, csv file for disaster messages, csv file for disaster message categories, and the database files output by the python file.

models - folder containing classifier.pkl - the pickle file of the model created in train_classifier.py - a python file reading in data from the database, training, testing & saving a model.

## Note on model performance, limitations & data quality
The model performance for this model is very poor. A Grid Search was attempted but when iterating over more than 2 parameters, the workspace fell asleep before the Grid Search could be completed. To avoid further complications. The optimal parameters found in the Grid Search were hard coded into the model. Alternative classifiers such as an SVC were also tried but were found to perform even worse. In a different environment & with more time & processing power available, model performance could likely be improved. It was also noticeable that for a lot of categories there weren't enough examples contained in the data. The child_alone category for example had to be completely removed for containing not enough entries to permit splitting into test and training data. Another example is that the current model does not recognise messages containing the word "water" as being related to water, as there are very few messages of that sort available. One possible solution could be the artificial creation of further messages containing keywords for underrepresented categories into the dataset. Alternatively, cross validation could be considered as an alternative to train-test splitting.

## License, Acknowledgements, and Authors
I worked on this project myself, making use of relevant library documentations and any materials provided by Udacity. This included answers given by mentors and other students on the programm. 

This code is based on templates created by udacity and should therefore only be re-used in line with their stipulations.

# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Click the `PREVIEW` button to open the homepage
