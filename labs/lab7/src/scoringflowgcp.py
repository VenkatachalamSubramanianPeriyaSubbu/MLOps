from metaflow import FlowSpec, step, Parameter, kubernetes, conda_base, timeout, retry, catch
import mlflow
import mlflow.pyfunc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

mlflow.set_tracking_uri("http://127.0.0.1:5000")

@conda_base(python="3.10", packages={
    "mlflow": "2.11.3",
    "scikit-learn": "1.4.1.post1",
    "matplotlib": "3.8.3",
    "pandas": "2.2.2",
    "seaborn": "0.13.2"
})
class ScoringFlow(FlowSpec):

    seed = Parameter("seed", default=42)
    model_name = Parameter("model_name", default="BestModel")
    model_stage = Parameter("model_stage", default="None")

    @step
    @kubernetes(cpu=1)
    @timeout(seconds=300)
    def start(self):
        print("Loading holdout data...")
        iris = load_iris()
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=self.seed
        )
        scaler = StandardScaler()
        scaler.fit(X_train)
        self.X_new = pd.DataFrame(scaler.transform(X_holdout), columns=iris.feature_names)
        self.y_true = y_holdout
        self.next(self.load_model)

    @step
    @timeout(seconds=120)
    def load_model(self):
        print(f"Loading model from registry: {self.model_name}")
        model_uri = f"models:/{self.model_name}/{self.model_stage}" if self.model_stage != "None" else f"models:/{self.model_name}/latest"
        self.model = mlflow.pyfunc.load_model(model_uri)
        self.next(self.predict)

    @step
    def predict(self):
        print("Making predictions...")
        self.preds = self.model.predict(self.X_new)
        self.next(self.output)

    @step
    def output(self):
        print("Generating and saving confusion matrix image...")
        cm = confusion_matrix(self.y_true, self.preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        print("confusion_matrix.png saved.")
        self.next(self.end)

    @step
    def end(self):
        print("Scoring flow complete.")

if __name__ == "__main__":
    ScoringFlow()
