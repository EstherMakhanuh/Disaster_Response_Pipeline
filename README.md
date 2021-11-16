# Disaster Response Pipeline Project 
## Table of Contents 

1. Project Overview
2. Installation  
3. Project Components
4. File Description 


## Project Overview 
The Disaster Response Pipeline project is part of the Udacity Data Science Nano Degree. The project analyzes a dataset containng real messages that are sent during disaster events. The goal of the project was to build an ETL pipeline to load and process data and a machine learning pipeline to classify those messages.

## Installation
The project has the following dependencies;
* python(> = 3.7)
* pandas
* numpy
* sqalchemy
* nltk
* plotly
* sys
* sklearn
* json
* pickle
* flask
## Project Components
There are 3 main components in this project

### 1. ETL Pipeline
* Loads the message.csv and categories.csv files
* Merge the two datasets
* cleans the data
* stores the data in a SQLite database
### 2. ML Pipeline 
* Loads the clean data from the SQLite database
* Splits the data to train sets
* Builds a text processing and machine learning model
* Evaluates the model
* Export the final model as a pickle file
### 3. Flask Web App
A web application displays some visualization about the dataset. Users can type in any new messages in the web app and receive the categories that the message may belong to.
![This is an image](https://github.com/EstherMakhanuh/Disaster_Response_Pipeline/blob/main/header.PNG)
Input field that takes new message
![This is an image](https://github.com/EstherMakhanuh/Disaster_Response_Pipeline/blob/main/message_genre.PNG)
Overview of Training dataset
![This is an image](https://github.com/EstherMakhanuh/Disaster_Response_Pipeline/blob/main/top10_categories.PNG)
Top 10 Categories
![This is an image](https://github.com/EstherMakhanuh/Disaster_Response_Pipeline/blob/main/categories.PNG)
Distribution of messages by Category
![This is an image](https://github.com/EstherMakhanuh/Disaster_Response_Pipeline/blob/main/message_example.PNG)
An example of a message

## File Description
* Read.md: read me file
* app 
  * run.py : flask file to run the app
  * templates
    * master.html: main page of the web app
    * go.html : result web page
* data
  * disaster_categories.csv : categories dataset
  * disaster_messages.csv : messages dataset
  * DisasterResponse.db : disaster response databse
  * process_data. py : ETL process to clean data
  * ETL Pipeline Preparation.ipynb : ETL pipeline preparation code
* model
  * train_classifier.py : classification code
  * M Pipeline Preparation.ipynb : ML pipeline preparation code
  * classifier.pkl :model saved as pickle file 
### Running it
git clone https://github.com/EstherMakhanuh/Disaster_Response_Pipeline

While in the project's root directory disaster-response-pipeline run the ETL pipeline that cleans and stores data in database.
 - python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

Next, run the ML pipeline that trains the classifier and save it.
- python model/train_classifier.py data/DisasterResponse.db model/classifier.pkl

Next, change directory into the app directory and run the Python file run.py.
- cd app
- python run.py

Finally, go to http://0.0.0.0:3001/ or http://localhost:3001/ in your web-browser. Type a message input box and click on the Classify Message button to see the various categories that your message falls into.


