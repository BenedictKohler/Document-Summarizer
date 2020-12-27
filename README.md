# DocUSum Web App

## Description:
A web application built using the Django Framework that allows users to obtain summaries to documents, text and website URL’s entered. Users are able to save their summaries to an SQLite Database as well as ask questions or provide feedback on a blog forum. The text summarizer, which is a Transformer Machine Learning Model, was created using Google Trax and TensorFlow.

## Getting Started:

### Text Summarizer:
In order to further train or develop the Trax summarizer you can download it separately from the **TraxSummarizer Folder**. From here you will need to install Google Trax and TensorFlow Datasets. Thereafter you will be able to train the summarizer on a dataset of your choice. (The cnn_dailymail dataset is the default).

### Web App with Summarizer:
In order to run the program in its entirety with an already trained text summarizer please follow the instructions below.

**Instructions:**
- Open a linux terminal or subsystem
- Update the linux system using `sudo apt-get update`
- Install pip3 using `sudo apt install python3-pip` and enter `y` to continue
- Install transformers using `pip3 install transformers`
- Install pytorch using `pip3 install torch`
- Install django using `pip3 install django`
- Install crispy forms using `pip3 install django-crispy-forms`
- Install beautiful soup using `pip3 install bs4`
- Pull website code using: `git clone https://github.com/BenedictKohler/Document-Summarizer.git`
- Navigate to folder ~/Document-Summarizer/ForumDevSite
- Start the website using `python3 manage.py runserver`
- Create a summary to initialize all values and needed training data (this may take a while as it needs to download 2GB of files)

## Running the program:
Upon completing the previous steps, you will be able to run the program on your PC’s Localhost. Navigate to the folder that contains the python manage.py file and then enter `python3 manage.py runserver` on your command line. From here you will be able to create an account and then post feedback on the blog forum as well as receive summaries to text/documents entered.
