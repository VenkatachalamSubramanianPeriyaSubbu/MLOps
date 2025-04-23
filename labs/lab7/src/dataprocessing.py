from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(seed=42):
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=seed)
    scaler = StandardScaler()
    return scaler.fit_transform(X_train), scaler.transform(X_test), y_train, y_test
