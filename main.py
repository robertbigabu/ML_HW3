import pandas as pd
from loguru import logger
import random

import torch
from src import AdaBoostClassifier, BaggingClassifier, DecisionTree
from src.utils import preprocess, plot_learners_roc, plot_feature_importance


def main():
    """
    Note:
    1) Part of line should not be modified.
    2) You should implement the algorithm by yourself.
    3) You can change the I/O data type as you need.
    4) You can change the hyperparameters as you want.
    5) You can add/modify/remove args in the function, but you need to fit the requirements.
    6) When plot the feature importance, the tick labels of one of the axis should be feature names.
    """
    random.seed(777)  # DON'T CHANGE THIS LINE
    torch.manual_seed(777)  # DON'T CHANGE THIS LINE
    train_df = pd.read_csv('./train.csv')
    test_df = pd.read_csv('./test.csv')

    X_train = train_df.drop(['target'], axis=1)
    y_train = train_df['target'].to_numpy()  # (n_samples, )

    X_test = test_df.drop(['target'], axis=1)
    y_test = test_df['target'].to_numpy()

    # (TODO): Implement you preprocessing function.
    X_train = preprocess(X_train)
    X_test = preprocess(X_test)

    """
    (TODO): Implement your ensemble methods.
    1. You can modify the hyperparameters as you need.
    2. You must print out logs (e.g., accuracy) with loguru.
    """
    # AdaBoost
    clf_adaboost = AdaBoostClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_adaboost.fit(
        X_train,
        y_train,
        num_epochs=1000,
        learning_rate=0.0001,
    )
    y_pred_classes, y_pred_probs = clf_adaboost.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'AdaBoost - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs, 
        y_trues=y_test, 
        fpath='adaboost_roc.png'
    ) 
    feature_importance = clf_adaboost.compute_feature_importance()
    # (TODO) Draw the feature importance
    plot_feature_importance(feature_importance, X_train.columns, fpath='adaboost_feature_importance.png')

    # Bagging
    clf_bagging = BaggingClassifier(
        input_dim=X_train.shape[1],
    )
    _ = clf_bagging.fit(
        X_train,
        y_train,
        num_epochs=500,
        learning_rate=0.01,
    )
    y_pred_classes, y_pred_probs = clf_bagging.predict_learners(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'Bagging - Accuracy: {accuracy_:.4f}')
    plot_learners_roc(
        y_preds=y_pred_probs,
        y_trues=y_test,
        fpath='bagging_roc.png',
    )
    feature_importance = clf_bagging.compute_feature_importance()
    # (TODO) Draw the feature importance
    plot_feature_importance(feature_importance, X_train.columns, fpath='bagging_feature_importance.png')
    
    
    # Decision Tree
    clf_tree = DecisionTree(
        max_depth=2,
    )
    clf_tree.fit(X_train, y_train)
    y_pred_classes = clf_tree.predict(X_test)
    accuracy_ = (y_pred_classes == y_test).mean()
    logger.info(f'DecisionTree - Accuracy: {accuracy_:.4f}')


if __name__ == '__main__':
    main()
