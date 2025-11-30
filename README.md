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

MLflow UI will be available at: http://127.0.0.1:5000


