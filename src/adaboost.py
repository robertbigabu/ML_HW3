import typing as t
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .utils import WeakClassifier

# Define a custom binary cross-entropy loss function
def binary_cross_entropy_loss(outputs, targets, weights):
    # Sigmoid activation on outputs
    probs = torch.sigmoid(outputs)
    # Binary cross-entropy formula
    loss = -weights * (targets * torch.log(probs + 1e-10) + (1 - targets) * torch.log(1 - probs + 1e-10))
    return loss.mean()

class AdaBoostClassifier:
    def __init__(self, input_dim: int, num_learners: int = 10) -> None:
        self.sample_weights = None
        # create 10 learners, dont change.
        self.learners = [
            WeakClassifier(input_dim=input_dim) for _ in range(num_learners)
        ]
        self.alphas = []

    def fit(self, X_train, y_train, num_epochs: int = 500, learning_rate: float = 0.001):
        """Implement your code here"""
        losses_of_models = []
        n_samples = X_train.shape[0]
        self.sample_weights = np.ones(n_samples) / n_samples # Uniform weight distribution

        # Convert X_train to NumPy arrays
        X_train = X_train.to_numpy()

        for model in self.learners:
            optimizer = optim.Adam (model.parameters(), lr=learning_rate)

            for epoch in range(num_epochs): # Train each WeakClassifier
                # Convert data and weights to tensors
                X_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.float32)
                weights_tensor = torch.tensor(self.sample_weights, dtype=torch.float32)

                # Compute prediction and loss
                pred = model(X_tensor).squeeze()
                loss = binary_cross_entropy_loss(pred, y_tensor, weights_tensor)
                
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # print(f"WeakClassifier Training - Epoch {epoch}, Loss: {loss.item()}")

            losses_of_models.append(loss.item())

            # Calculate predicted class and error for updating sample weights
            with torch.no_grad():
                pred_class = (pred.numpy() > 0).astype(int)
            error = np.sum(self.sample_weights * (pred_class != y_train)) / np.sum(self.sample_weights)
            print(error)

            # Calculate alpha for the current learner
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))
            print(f"alpha: {alpha}")
            self.alphas.append(alpha)
            
            # Update sample weights
            self.sample_weights *= np.exp(-alpha * (2 * y_train - 1) * (2 * pred_class - 1))
            self.sample_weights /= np.sum(self.sample_weights)  # Normalize

        return losses_of_models


    def predict_learners(self, X) -> t.Union[t.Sequence[int], t.Sequence[float]]:
        """Implement your code here"""
        # raise NotImplementedError
        final_predictions = np.zeros(X.shape[0])
        learner_probs = []
        # Convert X to NumPy arrays
        X = X.to_numpy()
        for model, alpha in zip(self.learners, self.alphas):
            with torch.no_grad():
                preds = model(torch.tensor(X, dtype=torch.float32)).squeeze()
                pred_probs = torch.sigmoid(preds).numpy()
                learner_probs.append(pred_probs)
                final_predictions += alpha * (2 * (pred_probs > 0.5) - 1)
        
        final_pred_classes = (final_predictions > 0).astype(int)
        return final_pred_classes.tolist(), learner_probs

    def compute_feature_importance(self) -> t.Sequence[float]:
        """Implement your code here"""
        # raise NotImplementedError
        feature_importance = np.zeros(13)
        for model, alpha in zip(self.learners, self.alphas):
            layer1_weights = model.linear_relu_stack[0].weight.detach().numpy()
            importance = np.abs(layer1_weights).sum(axis=0)
            feature_importance += alpha * importance

        return feature_importance / np.sum(self.alphas) 