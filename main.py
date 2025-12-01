from nn import NetworkConfig, NeuralNetwork
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    print("Loading dataset...")
    df = pd.read_csv("data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1).values
    y = df["diagnosis"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    def one_hot(label):
        return np.array([[1.0], [0.0]]) if label == 0 else np.array([[0.0], [1.0]])

    training_data = [(x.reshape(30, 1), one_hot(y)) for x, y in zip(X_train, y_train)]
    test_data = [(x.reshape(30, 1), y) for x, y in zip(X_test, y_test)]

    print("Building Neural Network...")
    config = NetworkConfig(
        layer_sizes=[30, 16, 2],
        activation="relu",
        cost_function="cross_entropy",
        learning_rate=0.0003,
        epochs=350,
        mini_batch_size=4,
        l1_lambda=0.005,
        l2_lambda=0.02,
        verbose=True,
    )

    nn = NeuralNetwork(config)
    print("Training...")
    nn.SGD(training_data, test_data)

    print(
        "Final Accuracy Percentage:",
        nn.evaluate(test_data) / len(test_data) * 100,
        "%",
    )

    def predict(sample):
        output = nn.feedforward(sample.reshape(30, 1))
        return np.argmax(output)

    sample = X_test[0]
    print("Prediction:", predict(sample))
    print("True Label:", y_test[0])


if __name__ == "__main__":
    main()
