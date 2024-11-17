import typing as t
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc



def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col], _ = pd.factorize(df[col])
        
        df[col] = (df[col] - df[col].mean()) / df[col].std() # Normalize

    return df


class WeakClassifier(nn.Module): #create a class that inherits from nn.Module.
    """
    Use pyTorch to implement a 1 ~ 2 layers model.
    Here, for example:
        - Linear(input_dim, 1) is a single-layer model.
        - Linear(input_dim, k) -> Linear(k, 1) is a two-layer model.

    No non-linear activation allowed.
    """
    def __init__(self, input_dim):
        super(WeakClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        # self.model = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def accuracy_score(y_trues, y_preds) -> float:
    # raise NotImplementedError
    y_preds = (y_preds > 0.5)
    return (y_trues == y_preds).mean().item()


def entropy_loss(outputs, targets):
    # raise NotImplementedError
    loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn(outputs, targets)


def plot_learners_roc(
    y_preds: t.List[t.Sequence[float]],
    y_trues: t.Sequence[int],
    fpath='./tmp.png',
):
    # raise NotImplementedError
    plt.figure(figsize=(10, 8))
    for i, y_pred in enumerate(y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_pred)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Learner {i+1} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Learner')
    plt.legend(loc='lower right')
    plt.savefig(fpath)
    plt.close()


def plot_feature_importance(feature_importance, feature_names, fpath='feature_importance.png'):
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importance)), feature_importance, align="center")
    plt.yticks(range(len(feature_importance)), [feature_names[i] for i in range(13)])
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()