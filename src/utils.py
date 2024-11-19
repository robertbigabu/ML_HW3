import typing as t
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


def preprocess(df: pd.DataFrame):
    """
    (TODO): Implement your preprocessing function.
    """
    cols = df.columns
    for col in cols:
        if df[col].dtype == "object":
            df = pd.get_dummies(df, columns=[col])
        else:
            df[col] = (df[col] - df[col].mean()) / df[col].std()  # Normalize

    return df


class WeakClassifier(nn.Module):  # create a class that inherits from nn.Module.
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
        self.model = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.Linear(5, 1),
        )
        # self.model = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.model(x)
        return x


def accuracy_score(y_trues, y_preds) -> float:
    raise NotImplementedError


def entropy_loss(outputs, targets):
    raise NotImplementedError


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

    feature_importance_dict = dict(zip(feature_names, feature_importance))

    category_mapping = {
        'loan_intent': ['loan_intent_VENTURE', 'loan_intent_PERSONAL', 'loan_intent_MEDICAL',
                        'loan_intent_HOMEIMPROVEMENT', 'loan_intent_EDUCATION', 'loan_intent_DEBTCONSOLIDATION'],
        'person_home_ownership': ['person_home_ownership_RENT', 'person_home_ownership_OWN',
                                  'person_home_ownership_OTHER', 'person_home_ownership_MORTGAGE'],
        'person_gender': ['person_gender_male', 'person_gender_female'],
        'previous_loan_defaults_on_file': ['previous_loan_defaults_on_file_Yes', 'previous_loan_defaults_on_file_No'],
        'person_education': ['person_education_Master', 'person_education_High School',
                             'person_education_Doctorate', 'person_education_Bachelor', 'person_education_Associate'],
        'credit_score': ['credit_score'],
        'cb_person_cred_hist_length': ['cb_person_cred_hist_length'],
        'loan_percent_income': ['loan_percent_income'],
        'loan_int_rate': ['loan_int_rate'],
        'loan_amnt': ['loan_amnt'],
        'person_emp_exp': ['person_emp_exp'],
        'person_income': ['person_income'],
        'person_age': ['person_age']
    }
    aggregated_importance = {}
    for category, one_hot_features in category_mapping.items():
        aggregated_importance[category] = sum(
            feature_importance_dict[feat] for feat in one_hot_features if feat in feature_importance_dict
        )

    plt.figure(figsize=(10, 6))
    plt.barh(list(aggregated_importance.keys()), list(aggregated_importance.values()))
    plt.ylabel("Feature")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(fpath)
    plt.close()
