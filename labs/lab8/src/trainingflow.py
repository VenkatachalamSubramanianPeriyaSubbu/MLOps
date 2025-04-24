from metaflow import FlowSpec, step, Parameter
import mlflow
from dataprocessing import load_and_preprocess_data
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class TrainingFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    cv_folds = Parameter("cv_folds", default=5)

    @step
    def start(self):
        print("Loading and preprocessing data...")
        self.X_train, self.X_test, self.y_train, self.y_test = load_and_preprocess_data(seed=self.seed)
        self.next(self.train)

    @step
    def train(self):
        print("Training and logging all models...")
        param_grid = {"n_estimators": [50, 100], "max_depth": [3, 5, None]}
        best_score = -np.inf
        self.best_model = None
        self.best_params = None

        for params in ParameterGrid(param_grid):
            with mlflow.start_run(nested=True):
                clf = RandomForestClassifier(**params, random_state=self.seed)
                clf.fit(self.X_train, self.y_train)
                score = clf.score(self.X_test, self.y_test)

                # Log params and metric
                mlflow.log_params(params)
                mlflow.log_metric("accuracy", score)
                mlflow.sklearn.log_model(clf, "model")

                print(f"Logged model with params={params} and accuracy={score}")

                if score > best_score:
                    best_score = score
                    self.best_model = clf
                    self.best_params = params
                    self.best_score = score

        self.next(self.register_model)

    @step
    def register_model(self):
        print("Registering best model to MLflow...")
        with mlflow.start_run():
            mlflow.log_params(self.best_params)
            mlflow.log_metric("best_accuracy", self.best_score)
            mlflow.sklearn.log_model(
                sk_model=self.best_model,
                artifact_path="model",
                registered_model_name="BestModel"
            )
        self.next(self.end)

    @step
    def end(self):
        print("Training flow complete.")

if __name__ == "__main__":
    TrainingFlow()
