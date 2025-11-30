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

### Steps:

- First create a notebook using .ipynb extension to run the commands & Check whether everything is running fine or not.
```bash
import mlflow                                                           //do this only after "mlflow ui"     
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Check localhost connection")                     //just to see whether everything is working fine
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


