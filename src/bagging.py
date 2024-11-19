import typing as t
import numpy as np
import torch
import torch.optim as optim
from .utils import WeakClassifier


def custom_loss(outputs, targets):
    # Implement binary cross-entropy loss without built-in functions
    probs = torch.sigmoid(outputs)
    loss = - (targets * torch.log(probs + 1e-10) + (1 - targets) * torch.log(1 - probs + 1e-10))
    return loss.mean()


class BaggingClassifier:
    def __init__(self, input_dim: int) -> None:
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(10)
        ]

    def fit(self, X_train, y_train, num_epochs: int, learning_rate: float):
        """Implement your code here"""
        X_train = X_train.to_numpy(dtype=float)
        n_samples = X_train.shape[0]
        losses_of_models = []

        for model in self.learners:
            # Generate bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample = X_train[indices]
            y_sample = y_train[indices]

            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs):
                # Convert data and weights to tensors
                X_tensor = torch.tensor(X_sample, dtype=torch.float32)
                y_tensor = torch.tensor(y_sample, dtype=torch.float32)

                # Compute prediction and loss
                pred = model(X_tensor).squeeze()
                loss = custom_loss(pred, y_tensor)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses_of_models.append(loss.item())

        return losses_of_models

    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        # raise NotImplementedError
        X = X.to_numpy(dtype=float)
        final_predictions = np.zeros(X.shape[0])
        learner_probs = []

        for model in self.learners:
            with torch.no_grad():
                outputs = model(torch.tensor(X, dtype=torch.float32)).squeeze()
                predictions = torch.sigmoid(outputs).numpy()
                learner_probs.append(predictions)
                final_predictions += (predictions > 0.5).astype(int)

        # Majority vote
        final_predictions = (final_predictions >= len(self.learners) / 2).astype(int)
        return final_predictions.tolist(), learner_probs

    def compute_feature_importance(self, X) -> t.Sequence[float]:
        """Implement your code here"""
        # raise NotImplementedError
        feature_importance = np.zeros(X.shape[1])
        for model in self.learners:
            layer1_weights = model.model[0].weight.detach().numpy()
            importance = np.abs(layer1_weights).sum(axis=0)
            feature_importance += importance

        return feature_importance / len(self.learners)
