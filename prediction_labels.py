import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from scipy.stats import pearsonr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm


def prediction_labels(X_train, Y_train, X_test, Y_test,n_components_LDA, best_result, weights):

    current_results = []

    # 0. LOGISTIC REGRESSION (Classification)---------------------------------------------------------------------------
    # Train the model
    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train, np.round(Y_train))

    # Make predictions using the testing set
    pred_class = clf.predict(X_test)
    # print('Pearson Correlation using Logistic Regression:')
    current_results.append(pearsonr(pred_class, Y_test)[0])
    # print(pearsonr(pred_class, Y_test)[0])

    # 1. LOGISTIC REGRESSION WITH LDA (Classification)------------------------------------------------------------------
    # Train the model
    clf2 = LinearDiscriminantAnalysis(n_components=n_components_LDA)
    clf2.fit(X_train, np.round(Y_train))
    X_train_projected = clf2.transform(X_train)
    X_test_projected = clf2.transform(X_test)

    clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_train_projected, np.round(Y_train))

    # Make predictions using the testing set
    pred_class_LDA = clf.predict(X_test_projected)
    # print('\nPearson Correlation using Logistic Regression an LDA:')
    current_results.append(pearsonr(pred_class_LDA, Y_test)[0])
    # print(pearsonr(pred_class_LDA, Y_test)[0])


    # 2. LINEAR REGRESSION ---------------------------------------------------------------------------------------------
    regr_1 = linear_model.LinearRegression()

    # Train the model using
    regr_1.fit(X_train, Y_train)

    # Make predictions using the testing set
    pred_1 = regr_1.predict(X_test)
    # print('\nPearson Correlation using Linear Regression:')
    current_results.append(pearsonr(pred_1, Y_test)[0])
    # print(pearsonr(pred_1, Y_test)[0])

    # 3. RIDGE REGRESSION ----------------------------------------------------------------------------------------------
    regr_2 = linear_model.Ridge(alpha=.5)

    # Train the model using
    regr_2.fit(X_train, Y_train)

    # Make predictions using the testing set
    pred_2 = regr_2.predict(X_test)
    # print('\nPearson Correlation using Ridge Regression:')
    current_results.append(pearsonr(pred_2, Y_test)[0])
    # print(pearsonr(pred_2, Y_test)[0])

    # 4. SVM (Clasification)--------------------------------------------------------------------------------------------
    regr_3 = svm.SVC()

    # Train the model using
    regr_3.fit(X_train, np.round(Y_train))

    # Make predictions using the testing set
    pred_3 = regr_3.predict(X_test)
    # print('\nPearson Correlation using SVM:')
    current_results.append(pearsonr(pred_3, Y_test)[0])
    # print(pearsonr(pred_3, Y_test)[0])

    # 5. SVR (Regression)-----------------------------------------------------------------------------------------------
    regr_4 = svm.SVR()

    # Train the model using
    regr_4.fit(X_train, Y_train)

    # Make predictions using the testing set
    pred_4 = regr_4.predict(X_test)
    # print('\nPearson Correlation using SVR:')
    current_results.append(pearsonr(pred_4, Y_test)[0])
    # print(pearsonr(pred_4, Y_test)[0])

    #-------------------------------------------------------------------------------------------------------------------
    if max(current_results)>best_result:
        best_result = max(current_results)
        print(best_result)
        print(weights)
        print(current_results.index(max(current_results)))
        print('--------------------------')

    return best_result
