import numpy as np
import multiprocessing

n_jobs = max(1, int(multiprocessing.cpu_count()*0.75)-2)

param_grid = {
    # "linear regression": {  # model id -> 0
    #     'alpha': [1.0],  # default -> 1.0
    #     'fit_intercept': [True],  # default -> True
    #     'normalize': [False],  # default -> False
    #     'precompute': [False, True],  # default -> False
    #     'copy_X': [True, False],  # default -> True
    #     'max_iter': [1000],  # default -> 1000
    #     'tol': [1e-4],  # default -> 1e-4
    #     'warm_start': [False, True],  # default -> False
    #     'positive': [False, True],  # default -> False
    #     'selection': ['cyclic', 'random']  # default -> cyclic
    # },

    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    "logistic regression": {  # model id -> 1

        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # default -> 1e-4
        'C': [1e5, 1e4, 1e3, 100, 10, 1.0, 0.1, 0.01, 0.001],  # default -> 1
        # 'max_iter': [10, 50, 100] + list(range(200, 1001, 100)),  # default -> 100
        'max_iter': [10, 50, 100, 500, 1000],  # default -> 100
        # 'intercept_scaling': [1],  # default -> 1

        'fit_intercept': [True, False],  # default -> True
        'warm_start': [True, False],  # default -> None

        'class_weight': ['balanced', None],  # default -> None
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # default -> l2
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'saga'],  # default -> lbfgs
        'multi_class': ['auto', 'ovr', 'multinomial'],  # default -> auto

        'n_jobs': [n_jobs],
        # 'verbose': [True]
    },

    # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    "k-neighborhood": {  # model id -> 2
        'n_neighbors': range(1, 21, 2),  # default -> 5
        'leaf_size': range(1, 20, 2),  # default -> 30
        'p': range(1, 6, 1),  # default -> 2
        'metric': ['minkowski', 'chebyshev', 'manhattan', 'euclidean', 'mahalanobis', 'hamming', 'canberra',
                   'braycurtis'],  # default -> minkowski

        'weights': ['uniform', 'distance'],  # default -> uniform
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],  # default -> auto. 'brute' -> unfeasible for long datasets

        'n_jobs': [n_jobs],
        # 'verbose': [True]
    },

    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    "decision tree classifier": {  # model id -> 3
        'max_depth': list(range(1, 10)) + list([None, 50, 100]),  # default -> None
        'max_leaf_nodes': list(range(1, 10)) + list([None, 50, 100]),  # default -> None
        'min_samples_split': list(range(1, 10)) + list([50, 100]),  # default -> 2
        'min_samples_leaf': list(range(1, 10)) + list([50, 100]),  # default -> 1

        'ccp_alpha': list(np.arange(0, 0.3, 0.05)),  # default -> 0.0

        # 'min_weight_fraction_leaf': [0.0],  # default -> 0.0
        # 'min_impurity_decrease': [0.0],  # default -> 0.0
        # 'min_impurity_split': [0],  # default -> 0

        'class_weight': [None, 'balanced'],  # default -> None
        'max_features': ['auto', 'sqrt', 'log2'],  # default -> auto
        'criterion': ['gini', 'entropy'],  # default -> gini
        'splitter': ['best', 'random'],  # default -> best
        # 'verbose': [True]

    },

    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    "random forest": {  # model id -> 4
        'n_estimators': [10, 50, 100, 500],  # default -> 100
        'max_samples': list(range(1, 10)) + list([None, 50, 100]),  # default -> None
        'max_depth': list(range(1, 10)) + list([None, 50, 100]),  # default -> None
        'min_samples_split': list(range(1, 10)) + list([50, 100]),  # default -> 2
        'min_samples_leaf': list(range(1, 10)) + list([50, 100]),  # default -> 1
        'max_leaf_nodes': list(range(1, 10)) + list([None, 50, 100]),  # default -> None

        'ccp_alpha': list(np.arange(0, 0.3, 0.05)),  # default -> 0.0

        # 'min_weight_fraction_leaf': [0.0],  # default -> 0.0
        # 'min_impurity_split': [None],  # default -> None
        # 'min_impurity_decrease': [0.0],  # default -> 0.0

        'bootstrap': [True, False],  # default -> True
        'oob_score': [False, True],  # default -> False
        'warm_start': [False, True],  # default -> False

        'class_weight': [None, 'balanced', 'balanced_subsample'],  # default -> None
        'max_features': ['auto', 'sqrt', 'log2'],  # default -> auto
        'criterion': ['gini', 'entropy'],  # default -> gini

        'n_jobs': [n_jobs],
        # 'verbose': [True]
    },

    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    "support vector": {  # model id -> 5
        'C': [1e5, 1e4, 1e3, 100, 10, 1.0, 0.1, 0.01, 0.001],  # default -> 1.0
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],  # default -> 1e-3
        'max_iter': [10, 50, 100, 500, 1000],  # default -> -1
        'degree': range(1, 10),  # default -> 3 #degree of the polynomial function

        'coef0': list(np.arange(-5, 5, 0.5
                                )),  # default -> 0.0

        'shrinking': [True, False],  # default -> True
        'probability': [False, True],  # default -> False
        'break_ties': [False, True],  # default -> False

        'class_weight': ['balanced', None],  # default -> None
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # default -> rbf # 'precomputed' -> square matrix must be pass
        'gamma': ['scale', 'auto'],  # default -> scale
        'decision_function_shape': ['ovo', 'ovr'],  # default -> ovr

        # 'verbose': [True]

    }
}
