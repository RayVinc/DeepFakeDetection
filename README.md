
### Deepfake Detection

We want to be able to diferenciate images from AI generated images.
(Maybe also a nice explanation of WHY?)

Pitch link:
https://docs.google.com/presentation/d/1y4D4UnabuaEQm8CRmkhLp0dVFm8lmjB40ltkDNpZ3pU/edit#slide=id.g278f5df32ba_1_2865

Explanation of the dataset:
https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection


This is the link of the repository: https://github.com/RayVinc/DeepFakeDetection

Collaborators:
@RayVinc
@sergioestebanez
@Lebaozki
@MalvinaGP

# 1️⃣ Local Setup

### 1.1) Workin with a local enviroment decision [🐍 lewagon]

We use Le Wagon enviroment, we are still mising if we are going to need any extra package. #Trello_task
We are working with Kaggle for the data exploring, the preprocessing part, the modeling and the fit; Kaggle notebook is replacing Jupyter nOTEBOO becasue of the quick access to our data.
After this part we will bring our code to Python files. #Trello_task

### 1.2) Define the Deep-fake package structure
#Trello_task (don't forget to actulize during the packaging process)


```bash
. # Challenge folder root
├── Makefile          # 🚪 Your command "launcher". Use it extensively (launch training, tests, etc...) #Trello_task
├── README.md         # The file you are reading right now! daily #Trello_task
├── notebooks
│   └── deepfake.ipynb   # Content result of Data Analys of the data set, preprocessor tasks & model.
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
│   └── ...  #Trello_task Create tests to check our code as we move forward (discuss if necesary)
├── .gitignore
```


# 2️⃣ Data Scientist approach.

### 2.1) Load data

### 2.1) Exploratory data analisys

### 2.2) Baselina score and model

### 2.2) Researching best model

### 2.3) Preprocess data (maybe compress/divide data set with labels/consider some featuring engeneering)

### 2.4) Arquitechture model

### 2.5) Evaluation model

### 2.6) Finetunning

### 2.7) Extra data

# 3️⃣ Package Your Code

🎯 THe goal is to be able to run the `deepface.interface.main_local` module as seen below

# 4️⃣ Investigate Scalability

# 5️⃣ Video processing

## 5️.1 Incremental Processing
## 5️.2 Incremental Learning
