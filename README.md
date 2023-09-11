##to complete before Thursday (once model has been selected)

# Deepfake Detection 👁️

We aim to distinguish between real images and AI-generated and deepfake images, contributing to the fight against disinformation and fraud. Here's why it matters:

Deepfake technology has the potential to create highly convincing fake videos and images, which can be used for malicious purposes. By developing effective deepfake detection methods, we can help protect individuals and organizations from falling victim to misinformation and scams. Our project focuses on detecting these AI-generated alterations and enhancing media trustworthiness. 🕵️‍♂️

Pitch link: Deepfake Detection Presentation

Explanation of the dataset: Real and Fake Face Detection Dataset

This is the link to the repository: DeepFakeDetection Repository

Collaborators:
@RayVinc
@Lebaozki
@MalvinaGP
@sergioestebanez


# 1️⃣ Local Setup 🏠

## 1.1) Working with a Local Environment Decision 🐍
We are utilizing the Le Wagon environment, with the possibility of requiring additional packages. 📦 This choice is informed by its compatibility with Kaggle, which we are using for data exploration, preprocessing, modeling, and fitting. Kaggle notebooks provide quick access to our data. After this phase, we will transition our code to Python files.

## 1.2) Define the Deepfake Package Structure 📂
Our project's directory structure is designed for organization and easy access. 🧩

```bash
. # Challenge folder root
├── Makefile          # 🚪 Your command "launcher". Use it extensively (launch training, tests, etc...) #Trello_task
├── README.md         # The file you are reading right now! daily #Trello_task
├── notebooks
│   └── deepfake.ipynb   # Content result of Data Analysis of the data set, preprocessor tasks & model.
├── requirements.txt   # List all third-party packages to add to your local environment
├── setup.py           # Enable `pip install` for your package
├── deepfake           # The code logic for this package
│   ├── __init__.py
│   ├── interface
│   │   ├── __init__.py
│   │   └── main_local.py  # 🚪 Your main Python entry point containing all "routes"
│   └── ml_logic
│   |    ├── __init__.py
│   |    ├── data.py           # Save, load and clean data
│   |    ├── encoders.py       # Custom encoder utilities
│   |    ├── model.py          # TensorFlow model
│   |    ├── preprocessor.py   # Sklearn preprocessing pipelines
│   |    ├── registry.py       # Save and load models
|   ├── utils.py    # # Useful python functions with no dependencies on deepfake logic
|   ├── params.py   # Global project params
|
├── tests  # Tests to run using `make test_...`
│   ├── ...
│   └── ...  #Trello_task Create tests to check our code as we move forward (discuss if necessary)
├── .gitignore

```


# 2️⃣ Data Scientist Approach 🧪

## 2.1) Load Data 📂

Loading the Dataset
In this step, we'll guide you through the process of loading the deepfake dataset for your project. We'll assume that you're working with the dataset available on Kaggle, which you can find here.

### Step 1: Dataset Preparation

Before loading the dataset, make sure you've downloaded it from Kaggle and organized it appropriately in your project directory. Typically, you'll have two folders: one for fake images and another for real images.

### Step 2: Exploring the Loaded Data

After running this code, you'll have the fake and real dictionaries containing your loaded image data. You can explore, preprocess, and use this data for further analysis, including training machine learning models for deepfake detection.

Remember to adjust the paths in the code to match your actual dataset directory structure.

By following these steps, you'll have successfully loaded the deepfake dataset into your project, setting the stage for the next stages of your deepfake detection pipeline.

## 2.2) Exploratory Data Analysis 🔍

## 2.3) Baseline Score and Model 📊

## 2.4) Researching the Best Model 🧠

## 2.5) Preprocess Data 🧹

## 2.6) Architectural Model 🏗️

## 2.7) Evaluate Model 📈

## 2.8) Fine-tuning 🛠️

## 2.9) Extra Data 📚



# 3️⃣ Package Your Code 📦

Our goal is to make the deepface.interface.main_local module runnable as seen below.

# 4️⃣ Investigate Scalability 📈


# 5️⃣ Video Processing 📹

## 5.1 Incremental Processing 🔄

## 5.2 Incremental Learning 📚
