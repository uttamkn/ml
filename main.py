import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

from nn import NetworkConfig, NeuralNetwork


def main():
    df = pd.read_csv("data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1).values
    y = df["diagnosis"].values.reshape(-1, 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    training_data = [(x.reshape(30, 1), y) for x, y in zip(X_train, y_train)]
    test_data = [(x.reshape(30, 1), y) for x, y in zip(X_test, y_test)]

    print("\n=== Baseline Model Comparison ===")

    baselines = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "SVM (RBF Kernel)": SVC(kernel="rbf"),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
    }

    for name, model in baselines.items():
        model.fit(X_train, y_train.ravel())
        acc = model.score(X_test, y_test.ravel()) * 100
        print(f"{name}: {acc:.2f}%")

    experiments = [
        ("No Regularization", 0.0, 0.0),
        ("L2 (1e-3)", 0.0, 1e-3),
        ("L2 (1e-2)", 0.0, 1e-2),
        ("L1 (1e-3)", 1e-3, 0.0),
    ]

    results = {}

    for name, l1, l2 in experiments:
        print(f"\n=== Neural Network: {name} ===")

        config = NetworkConfig(
            layer_sizes=[30, 16, 1],
            activation="relu",
            output_activation="sigmoid",
            cost_function="cross_entropy",
            learning_rate=0.005,
            epochs=200,
            mini_batch_size=16,
            l1_lambda=l1,
            l2_lambda=l2,
            verbose=False,
        )

        nn = NeuralNetwork(config)
        nn.SGD(training_data, test_data)

        final_acc = nn.test_accuracies[-1]
        results[name] = final_acc

        print(f"Final Test Accuracy: {final_acc:.2f}%")

        plt.figure()
        plt.plot(nn.train_losses, label="Train Loss")
        plt.plot(nn.test_losses, label="Test Loss")
        plt.title(f"Loss Curve - {name}")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"loss_curve_{name}.png")

    print("\n=== Summary: Neural Network Results ===")
    for k, v in results.items():
        print(f"{k}: {v:.2f}%")


if __name__ == "__main__":
    main()
