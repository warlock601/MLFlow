# MLFlow
Repo for end-to-end MLOps workflow using MLFlow

### Prerequisites:
- Upgrade pip
```bash
pip install --upgrade pip
```
- Create & Activate a Virtual env.
```bash
python3 -m venv mlflow-env
source mlflow-env/bin/activate
```
- Install MLflow
```bash
pip install mlflow
```
This installs: MLflow Tracking, MLflow Projects, MLflow Models, MLflow CLI / UI
Can start MLflow UI on your local machine:
```bash
mlflow ui
```

- There's another way using Conda. First install Conda. Then crete a virtual env using conda and then create a requirements.txt file and add "mlflow" to it.
```bash
conda create -p venv python==3.10
conda activate venv/                                // to activate the virtual environment
pip install -r requirements.txt                     // mention mlflow in requirements.txt to install it
```
- Then in the console, type this command to activate the MLFlow tracking server or the MLFlow UI
```bash
mlflow ui
```
<img width="1457" height="296" alt="image" src="https://github.com/user-attachments/assets/a462294d-c5f9-46f9-9823-50eb7ea035e2" />

MLflow UI will be accessible at: http://127.0.0.1:5000

MLFlow Tracking server: Whenever we create any project, we can track that project and all the other projects, for that we need a server that has all those MLflow capabilities. We can see all the Runs, Evaluation, Traces etc.

### Steps to create Experiments, Plots in MLFlow:

- First create a notebook using .ipynb extension to run the commands & Check whether everything is running fine or not.
```bash
import mlflow                                                           ##do this only after "mlflow ui"     
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Check localhost connection")                     ##just to see whether everything is working fine
```
After we run this set_experiment, we can see in the MLFlow UI that a new experiment is added. </br>
<img width="1900" height="511" alt="image" src="https://github.com/user-attachments/assets/ec1630bd-22ff-43b2-a283-0b5de6e105c6" />


- Now check whether we're abe to store any parameters in my server or not or whether we're able to track any parameters or not. 
```bash
with mlflow.start_run():
  mlflow.log_metric("test",1)
  mlflow.log_metric("vivek",2)
```
We can put ML experiment's loss value, accuracy value, training accuracy, test accuracy etc. over here. </br>
After running these, we can click on the experiment name "Check localhost connection" > Runs. We can see these in the metrics.
<img width="1841" height="748" alt="image" src="https://github.com/user-attachments/assets/a47d835b-aced-4ad7-8ea3-0e8a9c8b6ed8" />

We can also compare multiple experiments as MLFlow also provides visualizations such as Scatter plot, box plot etc...
<img width="1860" height="656" alt="image" src="https://github.com/user-attachments/assets/3fe082d5-6c30-4d66-91e2-0b810b563059" />

</br>
Since we are developing an end-to-end ML project so we will need other libraries as well such as scikit-learn, pandas, numpy...etc so add these all in requirements.txt

### Implementation of end-to-end ML project
We use MLFlow here to track various parameters and metrics. We'll train a ML model and while we're training with various parameters, each & every parameter of that specific experiment will be logged.

- In the requirements.txt add these:
```bash
mlflow
scikit-learn
pandas
numpy
```
Then run the command:
```bash
pip install -r requirements.txt
```

- Import pandas and datasets from Scikit-learn beacause we're going to consider some datasets that are available in the scikit-learn library. We're going to use Logistic Regression ML algo so that will also be imported from sklearn.
```bash
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.models import infer_signature
import pandas as pd
```
- Set the tracking uri
```bash
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
```

- Load the dataset. We're going to use "Iris" dataset which is already available insid ethe dataset library that we have imported. In Iris dataset we have 4 features: Petal length, petal width, sepal length, sepal width and based on that our output category will be like a flower. Three output categories: 012 and so these three categories will try to predict based on the input feature.
```bash
X,y=datasets.load_iris(return_X_y=True)
```
- Split the data into training and test sets. Test size is set to 0.20 which means 20% of data will be our test data.
```bash
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)  
```

- Define Hyperparameters. We can get info about logisitc reg hyperparameters from below. </br>
Scikit-learn hyperparameters: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
```bash
params = {"penalty":"12", "solver":"lbfgs", "max_iter":1000, "multi_class": "auto", "random_state":8888}
```
Here we are considering that these parameters are best for our model.

- Trin the model.
```bash
lr=LogisticRegression(**params)
lr.fit(X_train,y_train)
```
we'll get something like this:
<img width="1227" height="433" alt="image" src="https://github.com/user-attachments/assets/03325e12-95b0-438a-921d-8896b69179cd" />

- Prediction on the test set. The values that we get from y_pred are based on the inputs that we provided using X_test
```bash
y_pred=lr.predict(X_test)
y_pred                              ## to print y_pred
```
- Calculate Accuracy.
```bash
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
```
<img width="959" height="295" alt="image" src="https://github.com/user-attachments/assets/64bb57e7-9b6b-4a99-b1dc-9d8950875eda" />

- MLFlow Tracking. Start the MLFlow UI and then run this block of code.
```bash

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

mlflow.set_experiment("MLFlow Quickstart")                             # create a new experiment

with mlflow.start_run():                                               # start the MLFlow run
  mlflow.log_params(params)                                            # log the hyperparameters
  mlflow.log_metric("acccuracy",accuracy)                              # log the accuracy metrics
  mlflow.set_tag("Training Info","basic LR model for iris data")       # set a tag that we can use to remind ourselves what this run was for
  signature=infer_signature(X_train,lr.predict(X_train))

  # log the model
  model_info=mlflow.sklearn.log_model(
      sk_model=lr,                                                     # lr is the model name
      artifact_path="iris_model",
      signature=signature,
      input_example=X_train,
      registered_model_name="tracking-quickstart"
  )                                
  
```
infer_signature() is used to infer model signature form the training data(input), model predictions(output) and parameters(for inference). The signature represents model input and output as data frames with named columns. This method will raise an exception if the user data contains incomptible types. 
In "mlruns" folder we can see all the artifacts, metrics like accuracy, parameters like max_iter...
