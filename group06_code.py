'''
FINAL PROJECT
TEAM 06
_____________________
LAST REVIEW: Dec 2021
_____________________
THE MNIST DATABASE of handwritten digits
http://yann.lecun.com/exdb/mnist/

Yann LeCun, Courant Institute, NYU
Corinna Cortes, Google Labs, New York
Christopher J.C. Burges, Microsoft Research, Redmond

'''



# In[0]: Imports Libraries.
# Standard libraries.
import os
import joblib
import numpy as np
import pandas as pd # data processing

# Visualization libraries.
import matplotlib.pyplot as plt
import seaborn as sns

# Modeling and machine learning.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from hyperopt import Trials
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from hyperopt import fmin
from scipy.ndimage.interpolation import shift
from hyperopt import hp, tpe, STATUS_OK
from scipy.stats import reciprocal, uniform
from scipy.ndimage.interpolation import shift

# Specify paths for easy dataloading.
# Warning: Change BASE_PATH depends where you put file programming.
BASE_PATH = 'D:/HCMC University of Technology and Education/2nd Year_Semester 1/Machine Learning/Final_Project/dataset/'
TRAIN_PATH = BASE_PATH + 'mnist_train.csv'
TEST_PATH = BASE_PATH + 'mnist_test.csv'

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

# Define new_run.
new_run = False

# To make this notebook's output stable across runs
np.random.seed(42)



# In[1]: MNIST dataset.
# 1.1. Load the data.
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
# Info the data
# Printing the size of the dataframe.
print ('Training dataset has %i observations and %i variables' %(train.shape[0], train.shape[1]))
print ('Testing dataset has %i observations and %i variables' %(test.shape[0], test.shape[1]))
# Seperate the target and independant variables.
train_x = train.iloc[:,1:]
train_y = train.iloc[:,0]
# Print example image.
def print_image(row, df):
    temp = df.iloc[row,:].values
    temp = temp.reshape(28,28).astype('uint8')
    plt.imshow(temp)  
print_image(0, train_x)

#%% 1.2. Visualize target distribution.
train['label'].value_counts().sort_index().plot(kind='bar', figsize=(15,9), rot=0)
plt.title('Visualization of class distribution for the MNIST Dataset', fontsize=20, weight='bold')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Class', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
save_fig('class_distribution_plot')
# Check the frequency of each number.
train_y.value_counts().sort_index()

#%% 1.3. Split the data.
# We put test set aside when we are exploring dataset, to prevent our brain to mislead us.
# Trying to create a solution that generalizes and not memorizes.
# Test set should only be used for final evaluation.
# Create X_train_split and y_train_split variables.
X_train_split = train.drop(['label'], axis=1).copy()
y_train_split = train['label'].copy()
# Create X_test and y_test variables.
X_test = test.drop(['label'], axis=1).copy()
y_test = test['label'].copy()
# Split X_train_split and y_train_split data.
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_train_split, y_train_split, test_size=0.05, random_state=42)
# Info the data.
print('Training features:', X_train.shape)
print('Training labels:', y_train.shape)
print('Validation features:', X_val.shape)
print('Validation labels:', y_val.shape)
print('Test features:', X_test.shape)
print('Test labels:', y_test.shape)

#%% 1.4. Scale the data.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_val_scaled = scaler.transform(X_val.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))




# In[2]: Train model with Decision Tree Classifier, Bagging Classifier, Ada Boost Classifier.
# 2.1. Use grid search with cross-validation.
# Parameters to be tested - Iterate several times to find better ranges.
# Try various values for max_leaf_nodes, max_depth.
param_tree_clf = {
    'max_depth': list(range(10, 30)),
    'criterion': ['gini', 'entropy']
    }
# Instantiate Decision Tree Classifier.
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(random_state=42)
# Instantiate Grid Search CV
from sklearn.model_selection import GridSearchCV
tree_clf_grid = GridSearchCV(tree_clf, 
                             param_tree_clf,
                             cv=3, verbose=3)
# Find good hyperparameter values for Decision Tree Classifier.
# Warning: takes time for new run!
if new_run == True:
    tree_clf_grid.fit(X_train_scaled, y_train)
    joblib.dump(tree_clf_grid, 'saved_var/tree_clf_grid')
else:
    tree_clf_grid = joblib.load('saved_var/tree_clf_grid')
# Best hyperparameters values for Decision Tree Classifier.
print(tree_clf_grid.best_estimator_)
# Decision Tree Classifier score
print('Decision Tree Classifier score:', tree_clf_grid.best_score_)

# 2.2. Evaluate trained Decision Tree Classifier on validation set.
tree_clf = DecisionTreeClassifier(**tree_clf_grid.best_params_, 
                                  random_state=42)
# Warning: takes time for new run!
from sklearn.metrics import accuracy_score
if new_run == True:
    y_pred = tree_clf.predict(X_val_scaled)
    tree_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(tree_clf_acc, 'saved_var/tree_clf_acc')
else:
    tree_clf_acc = joblib.load('saved_var/tree_clf_acc')
# Accuracy on validation set
print('Accuracy on validation set:', tree_clf_acc)


#%% 2.3. Train model with Bagging Classifier based trained Decision Tree Classifier.
# Instantiate Bagging Classifier.
from sklearn.ensemble import BaggingClassifier
bag_clf = BaggingClassifier(tree_clf, n_estimators=500, 
                            max_features=1.0, max_samples=1.0,
                            bootstrap=True, oob_score=True, 
                            n_jobs=-1, random_state=42)
# Warning: takes time for new run!
if new_run == True:
    bag_clf.fit(X_train_scaled, y_train)
    joblib.dump(bag_clf, 'saved_var/bag_clf')
else:
    bag_clf = joblib.load('saved_var/bag_clf')
# Bagging Classifier score.
print('Bagging Classifier score:', bag_clf.oob_score_)

# 2.4. Evaluate trained Bagging Classifier on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    y_pred = bag_clf.predict(X_val_scaled)
    bag_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(bag_clf_acc, 'saved_var/bag_clf_acc')
else:
    bag_clf_acc = joblib.load('saved_var/bag_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', bag_clf_acc)


#%% 2.5. Using Ada Boost Classifier based trained Decision Tree Classifier.
# Instantiate Ada Boost Classifier.
from sklearn.ensemble import AdaBoostClassifier
ada_clf = AdaBoostClassifier(tree_clf, 
                             n_estimators=500, 
                             algorithm='SAMME.R', 
                             learning_rate=0.1, 
                             random_state=42)
# Warning: takes time for new run!
if new_run == True:
    ada_clf.fit(X_train_scaled, y_train)
    joblib.dump(ada_clf, 'saved_var/ada_clf')
else:
    ada_clf = joblib.load('saved_var/ada_clf')
# Ada Boost Classifier score.
if new_run == True:
    ada_clf_score = ada_clf.score(X_train_scaled, y_train)
    joblib.dump(ada_clf_score, 'saved_var/ada_clf_score')
else:
    ada_clf_score = joblib.load('saved_var/ada_clf_score')
print('Ada Boost Classifier score:', ada_clf_score)

# 2.6. Evaluate trained Ada Boost Classifier on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    y_pred = ada_clf.predict(X_val_scaled)
    ada_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(ada_clf_acc, 'saved_var/ada_clf_acc')
else:
    ada_clf_acc = joblib.load('saved_var/ada_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', ada_clf_acc)




# In[3]: Train model with K-Neighbors Classifier.
# 3.1. Use grid search with cross-validation.
# Parameters to be tested - Iterate several times to find better ranges.
param_knn_clf = [{'weights': ["uniform", "distance"], 
                  'n_neighbors': [3, 4, 5]}]
# Instantiate K-Neighbors Classifier.
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
# Instantiate Grid Search CV.
from sklearn.model_selection import GridSearchCV
knn_clf_grid = GridSearchCV(knn_clf, 
                            param_knn_clf,
                            cv=3, verbose=3)
# Warning: takes time for new run!
if new_run == True:
    knn_clf_grid.fit(X_train_scaled, y_train)
    joblib.dump(knn_clf_grid, 'saved_var/knn_clf_grid')
else:
    knn_clf_grid = joblib.load('saved_var/knn_clf_grid')
# Best hyperparameters values for K-Neighbors Classifier.
print(knn_clf_grid.best_estimator_)
# K-Neighbors Classifier score.
print('K-Neighbors Classifier score:', knn_clf_grid.best_score_)

# 3.2. Evaluate trained K-Neighbors Classifier on validation set.
knn_clf = KNeighborsClassifier(**knn_clf_grid.best_params_)
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    y_pred = knn_clf_grid.predict(X_val_scaled)
    knn_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(knn_clf_acc, 'saved_var/knn_clf_acc')
else:
    knn_clf_acc = joblib.load('saved_var/knn_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', knn_clf_acc)




# In[4]: Train model with Support Vector Classifier.
# 4.1. Using randomized search with cross-validation.
# Tune the hyperparameters by doing Randomized Search CV.
from scipy.stats import reciprocal, uniform
param_svc_clf = {"gamma": reciprocal(0.001, 0.1),
                 "C": uniform(1, 10)}
# Instantiate Support Vector Classifier.
# Try an SVC with an RBF kernel (the default).
from sklearn.svm import SVC
svc_clf = SVC(gamma="scale")
# Instantiate Randomized Search CV.
from sklearn.model_selection import RandomizedSearchCV
svc_clf_rnd = RandomizedSearchCV(svc_clf, param_svc_clf, 
                                 n_iter=10, cv=3, verbose=2)
# Warning: takes time for new run!
if new_run == True:
    svc_clf_rnd.fit(X_train_scaled, y_train)
    joblib.dump(svc_clf_rnd, 'saved_var/svc_clf_rnd')
else:
    svc_clf_rnd = joblib.load('saved_var/svc_clf_rnd')
# Best hyperparameters values for Support Vector Classifier.
print(svc_clf_rnd.best_estimator_)
# Support Vector Classifier score.
print('Support Vector Classifier score:', svc_clf_rnd.best_score_)

# 4.2. Evaluate trained Support Vector Classifier on validation set.
svc_clf = SVC(**svc_clf_rnd.best_params_, probability=True)
from sklearn.metrics import accuracy_score
# Warning: takes time for new run! 
if new_run == True:
    y_pred = svc_clf_rnd.predict(X_val_scaled)
    svc_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(svc_clf_acc, 'saved_var/svc_clf_acc')
else:
    svc_clf_acc = joblib.load('saved_var/svc_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', svc_clf_acc)




# In[5]: Train model with Random Forest Classifier.
# 5.1. Bayesian optimization using hyperopt.
# A model-based method for finding the minimum of a function.
from hyperopt import Trials, fmin, hp, tpe, STATUS_OK
# Defines the range of input values to train.
param_forest_clf = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']), 
    'max_depth': hp.randint('max_depth', 10, 30), 
    'max_features': hp.randint('max_features', 10, 100), 
    'n_estimators': hp.randint('n_estimators', 100, 300)
    }
# Instantiate Random Forest Classifier.
from sklearn.ensemble import RandomForestClassifier
# Instantiate cross_val_score.
from sklearn.model_selection import cross_val_score 
# This space creates a probability distribution for each of the used hyperparameters.
def objective(space):
    forest_clf = RandomForestClassifier(criterion = space['criterion'], 
                                        max_depth = space['max_depth'], 
                                        max_features = space['max_features'], 
                                        n_estimators = space['n_estimators'], 
                                        random_state=42)
    # Warning: takes time for new run!
    if new_run == True:
        forest_clf_score = cross_val_score(forest_clf, 
                                           X_train_scaled, y_train, 
                                           cv=3).mean()
        joblib.dump(forest_clf_score, 'saved_var/forest_clf')
    else:
        forest_clf_score = joblib.load('saved_var/forest_clf')
    # To maximize accuracy, return it as a negative value.
    return {'loss':-forest_clf_score, 'status':STATUS_OK}
trials = Trials()
best_forest_clf = fmin(fn=objective, space=param_forest_clf,
                       algo=tpe.suggest, max_evals=20,
                       trials=trials)
# Save trained Random Forest Classifier model.
crit = {0: 'entropy', 1: 'gini'}
forest_clf = RandomForestClassifier(criterion=crit[best_forest_clf['criterion']], 
                                    max_depth=best_forest_clf['max_depth'], 
                                    max_features=best_forest_clf['max_features'], 
                                    n_estimators=best_forest_clf['n_estimators'], 
                                    random_state=42)
print(forest_clf)

# 5.2. Evaluate trained Random Forest Classifier on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    forest_clf.fit(X_train_scaled, y_train)
    y_pred = forest_clf.predict(X_val_scaled)
    forest_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(forest_clf_acc, 'saved_var/forest_clf_acc')
else:
    forest_clf_acc = joblib.load('saved_var/forest_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', forest_clf_acc)




# In[6]: Train model with Extra Trees Classifier.
# 6.1. Bayesian optimization using hyperopt.
# A model-based method for finding the minimum of a function.
from hyperopt import Trials, fmin, hp, tpe, STATUS_OK
# Defines the range of input values to train.
param_extra_clf = {
    'criterion': hp.choice('criterion', ['entropy', 'gini']), 
    'max_depth': hp.randint('max_depth', 10, 30), 
    'max_features': hp.randint('max_features', 10, 100), 
    'n_estimators': hp.randint('n_estimators', 100, 300) 
    }
# Instantiate Extra Trees Classifier.
from sklearn.ensemble import ExtraTreesClassifier
# Instantiate cross_val_score.
from sklearn.model_selection import cross_val_score 
# This space creates a probability distribution for each of the used hyperparameters.
def objective(space):
    extra_clf = ExtraTreesClassifier(criterion=space['criterion'],  
                                     max_depth=space['max_depth'],  
                                     max_features=space['max_features'], 
                                     n_estimators=space['n_estimators'], 
                                     random_state=42)
    # Warning: takes time for new run!
    if new_run == True:
        extra_clf_score = cross_val_score(extra_clf, 
                                          X_train_scaled, y_train, 
                                          cv=3).mean()
        joblib.dump(extra_clf_score, 'saved_var/extra_clf')
    else:
        extra_clf_score = joblib.load('saved_var/extra_clf')
    # To maximize accuracy, return it as a negative value.
    return {'loss':-extra_clf_score, 'status':STATUS_OK}
trials = Trials()
best_extra_clf = fmin(fn=objective, space=param_extra_clf, 
                         algo=tpe.suggest, max_evals=20, 
                         trials=trials)
# Save trained Extra Trees Classifier model.
crit = {0: 'entropy', 1: 'gini'}
extra_clf = ExtraTreesClassifier(criterion=crit[best_extra_clf['criterion']], 
                                 max_depth=best_extra_clf['max_depth'], 
                                 max_features=best_extra_clf['max_features'], 
                                 n_estimators=best_extra_clf['n_estimators'], 
                                 random_state=42)
print(extra_clf)

# 6.2. Evaluate trained Extra Trees Classifier on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    extra_clf.fit(X_train_scaled, y_train)
    y_pred = extra_clf.predict(X_val_scaled)
    extra_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(extra_clf_acc, 'saved_var/extra_clf_acc')
else:
    extra_clf_acc = joblib.load('saved_var/extra_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', extra_clf_acc)




# In[7]: Train model with SGD Classifier.
# 7.1. Bayesian optimization using hyperopt.
# A model-based method for finding the minimum of a function.
from hyperopt import Trials, fmin, hp, tpe, STATUS_OK
# Defines the range of input values to train.
param_sgd_clf = {
    'alpha': hp.uniform('alpha', 0.0001, 1), 
    'tol': hp.uniform('tol', 0.001, 1),  
    'max_iter': hp.randint('max_iter', 1000, 5000), 
    'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling', 'adaptive']),
    'eta0': hp.uniform('eta0', 0.0001, 1)
    }
# Instantiate SGD Classifier.
from sklearn.linear_model import SGDClassifier
# Instantiate cross_val_score.
from sklearn.model_selection import cross_val_score 
# This space creates a probability distribution for each of the used hyperparameters.
def objective(space):
    sgd_clf = SGDClassifier(alpha=space['alpha'], 
                            tol=space['tol'], 
                            max_iter=space['max_iter'], 
                            learning_rate=space['learning_rate'], 
                            eta0=space['eta0'], 
                            random_state=42)
    # Warning: takes time for new run!
    if new_run == True:
        sgd_clf_score = cross_val_score(sgd_clf,
                                        X_train_scaled, y_train,
                                        n_jobs=-1, cv=3).mean()
        joblib.dump(sgd_clf_score, 'saved_var/sgd_clf')
    else:
        sgd_clf_score = joblib.load('saved_var/sgd_clf')
    # To maximize accuracy, return it as a negative value.
    return {'loss':-sgd_clf_score, 'status':STATUS_OK}
trials = Trials()
best_sgd_clf = fmin(fn=objective, space=param_sgd_clf, 
                    algo=tpe.suggest, max_evals=200, 
                    trials=trials)
# Save trained SGD Classifier model.
learn = {0: 'constant', 1: 'optimal', 2: 'invscaling', 3: 'adaptive'}
sgd_clf = SGDClassifier(alpha=best_sgd_clf['alpha'], 
                        tol=best_sgd_clf['tol'], 
                        max_iter=best_sgd_clf['max_iter'], 
                        learning_rate=learn[best_sgd_clf['learning_rate']], 
                        eta0=best_sgd_clf['eta0'], 
                        random_state=42)
print(sgd_clf)

# 7.2. Evaluate trained SGD Classifier on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    sgd_clf.fit(X_train_scaled, y_train)
    y_pred = sgd_clf.predict(X_val_scaled)
    sgd_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(sgd_clf_acc, 'saved_var/sgd_clf_acc')
else:
    sgd_clf_acc = joblib.load('saved_var/sgd_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', sgd_clf_acc)




# In[8]: Train model with Logistic Regression.
# 8.1. Bayesian optimization using hyperopt.
# A model-based method for finding the minimum of a function.
from hyperopt import Trials, fmin, hp, tpe, STATUS_OK
# Defines the range of input values to train.
param_log_clf = {
    'tol': hp.uniform('tol', 0.0001, 1), 
    'C': hp.uniform('C', 0.01, 1), 
    'max_iter': hp.randint('max_iter', 1000, 5000)
    }
# Instantiate Logistic Regression.
from sklearn.linear_model import LogisticRegression
# Instantiate cross_val_score.
from sklearn.model_selection import cross_val_score 
# This space creates a probability distribution for each of the used hyperparameters.
def objective(space):
    log_clf = LogisticRegression(tol=space['tol'], 
                                 C=space['C'], 
                                 max_iter=space['max_iter'], 
                                 random_state=42)
    # Warning: takes time for new run!
    if new_run == True:
        log_clf_score = cross_val_score(log_clf,
                                        X_train_scaled, y_train,
                                        n_jobs=-1, cv=3).mean()
        joblib.dump(log_clf_score, 'saved_var/log_clf')
    else:
        log_clf_score = joblib.load('saved_var/log_clf')
    # To maximize accuracy, return it as a negative value.
    return {'loss':-log_clf_score, 'status':STATUS_OK}
trials = Trials()
best_log_clf = fmin(fn=objective, space=param_log_clf, 
                    algo=tpe.suggest, max_evals=100, 
                    trials=trials)
# Save trained Logistic Regression model.
log_clf = LogisticRegression(tol=best_log_clf['tol'], 
                             C=best_log_clf['C'], 
                             max_iter=best_log_clf['max_iter'], 
                             random_state=42)
print(log_clf)

# 8.2. Evaluate trained Logistic Regression on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    log_clf.fit(X_train_scaled, y_train)
    y_pred = log_clf.predict(X_val_scaled)
    log_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(log_clf_acc, 'saved_var/log_clf_acc')
else:
    log_clf_acc = joblib.load('saved_var/log_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', log_clf_acc)


#%% 8.3. Using Gradient Boosting Classifier based trained Logistic Regression.
# Instantiate Gradient Boosting Classifier.
from sklearn.ensemble import GradientBoostingClassifier
gra_clf = GradientBoostingClassifier(init=log_clf, 
                                     n_estimators=500, 
                                     loss='deviance', 
                                     learning_rate=0.1,  
                                     random_state=42)
# Warning: takes time for new run!
if new_run == True:
    gra_clf.fit(X_train_scaled, y_train)
    joblib.dump(gra_clf, 'saved_var/gra_clf')
else:
    gra_clf = joblib.load('saved_var/gra_clf')
# Gradient Boosting Classifier score.
if new_run == True:
    gra_clf_score = gra_clf.score(X_train_scaled, y_train)
    joblib.dump(gra_clf_score, 'saved_var/gra_clf_score')
else:
    gra_clf_score = joblib.load('saved_var/gra_clf_score')
print('Gradient Boosting Classifier score:', gra_clf_score)

# 8.4. Evaluate trained Gradient Boosting Classifier on validation set.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    y_pred = gra_clf.predict(X_val_scaled)
    gra_clf_acc = accuracy_score(y_val, y_pred)
    joblib.dump(gra_clf_acc, 'saved_var/gra_clf_acc')
else:
    gra_clf_acc = joblib.load('saved_var/gra_clf_acc')
# Accuracy on validation set.
print('Accuracy on validation set:', gra_clf_acc)




# In[9]: Ensemble learning using Voting Classifier.
# 9.1. Try to combine them into an ensemble that outperforms them all on the validation set.
# Using a soft or hard Voting Classifier.
named_estimators = [
    ('bag_clf', bag_clf), 
    ('knn_clf', knn_clf), 
    ('svc_clf', svc_clf), 
    ('forest_clf', forest_clf), 
    ('extra_clf', extra_clf)
    ]
# Instantiate Voting Classifier.
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(named_estimators)
# Warning: takes time for new run!
if new_run == True:
    voting_clf.fit(X_train_scaled, y_train)
    joblib.dump(voting_clf, 'saved_var/voting_clf')
else:
    voting_clf = joblib.load('saved_var/voting_clf')
print(voting_clf.estimators)
# Voting Classifier score.
# Warning: takes time for new run!
if new_run == True:
    voting_score = voting_clf.score(X_train_scaled, y_train)
    joblib.dump(voting_score, 'saved_var/voting_score')
else:
    voting_score = joblib.load('saved_var/voting_score')
print('Voting Classifier score:', voting_score)

# 9.2. Evaluate trained Voting Classifier on validation set.
# Hard Voting accuracy.
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    y_pred = voting_clf.predict(X_val_scaled)
    voting_hard_acc = accuracy_score(y_val, y_pred)
    joblib.dump(voting_hard_acc, 'saved_var/voting_hard_acc')
else:
    voting_hard_acc = joblib.load('saved_var/voting_hard_acc')
# Hard Voting accuracy on validation set.
print("Hard Voting accuracy on validation set: ", voting_hard_acc)
# Soft Voting accuracy.
voting_clf.voting = "soft"
from sklearn.metrics import accuracy_score
# Warning: takes time for new run!
if new_run == True:
    y_pred = voting_clf.predict(X_val_scaled)
    voting_soft_acc = accuracy_score(y_val, y_pred)
    joblib.dump(voting_soft_acc, 'saved_var/voting_soft_acc')
else:
    voting_soft_acc = joblib.load('saved_var/voting_soft_acc')
# Soft Vsoting accuracy on validation set.
print('Soft Voting accuracy on validation set:', voting_soft_acc)




# In[10]: Final evaluate trained model on test set.
# 10.1. Evaluate accuracy on test set.
# Warning: takes time for new run!
if new_run == True:
    y_pred = voting_clf.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)
    joblib.dump(test_acc, 'saved_var/test_acc')
else:
    test_acc = joblib.load('saved_var/test_acc')
# Accuracy on test set.
print('Accuracy on test set:', test_acc)

# 10.2. Classification report & confusion matrix.
# Classification report for Voting Classifier.
from sklearn.metrics import classification_report
# Warning: takes time for new run!
if new_run == True:
    y_pred = voting_clf.predict(X_test_scaled)
    test_report = classification_report(y_test, y_pred)
    joblib.dump(test_report, 'saved_var/test_report')
else:
    test_report = joblib.load('saved_var/test_report')
print(
    f"Classification report for Voting Classifier:\n"
    f"{test_report}\n"
    )
# Confusion matrix for Voting Classifier.
from sklearn.metrics import confusion_matrix
# Warning: takes time for new run!
if new_run == True:
    y_pred = voting_clf.predict(X_test_scaled)
    test_matrix = confusion_matrix(y_test, y_pred)
    joblib.dump(test_matrix, 'saved_var/test_matrix')
else:
    test_matrix = joblib.load('saved_var/test_matrix')
# Plot confusion matrix.
f,ax = plt.subplots(figsize=(9, 9))
sns.heatmap(test_matrix, annot=True, 
            linewidths=0.01, cmap="Blues", 
            linecolor="gray", fmt= '.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix", fontsize=20, weight='bold')
save_fig('confusion_matrix_plot')
plt.show()




# In[11]: Data Augmentation.
# 11.1. Using shift function to shift image.
from scipy.ndimage.interpolation import shift
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])
# Shifted down 5 pixel & Shifted left 5 pixel.
image = X_train_scaled[0]
shifted_image_down = shift_image(image, 0, 5)
shifted_image_left = shift_image(image, -5, 0)
# Plot a augmented image.
plt.figure(figsize=(12,3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show()

# 11.2. Transform training set to data augmentation.
X_train_augmented = [image for image in X_train_scaled]
y_train_augmented = [label for label in y_train]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    for image, label in zip(X_train_scaled, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)

X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

# 11.3. Train model on augmented data and evaluate again on validation set.
# Warning: takes time for new run!
if new_run == True:
    extra_clf.fit(X_train_augmented, y_train_augmented)
    y_pred = extra_clf.predict(X_val_scaled)
    aug_acc = accuracy_score(y_val, y_pred)
    joblib.dump(aug_acc, 'saved_var/aug_acc')
else:
    aug_acc = joblib.load('saved_var/aug_acc')
print('Accuracy after training on augmented data:', aug_acc)




# In[12]: Using t-SNE to boost
# 12.1. Perform Truncated Singular Value Decomposition (TSVD) on all features
# This will reduce the amount of features to 50 and will simplify t-SNE
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=50).fit_transform(X_train_split)
# Fit t-SNE on the Truncated SVD reduced data (50 features)
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
# Warning: takes time for new run!
if new_run == True:
    transformed = tsne.fit_transform(tsvd)
    joblib.dump(transformed, 'saved_var/tsne_trans')
else:
    transformed = joblib.load('saved_var/tsne_trans')
# Split up the t-SNE results in training and testing data
train_tsne = pd.DataFrame(transformed[:len(train)], columns=['component1', 'component2'])
# Visualize the results for t-SNE on MNIST
plt.figure(figsize=(12, 12))
plt.title(f"Visualization of t-SNE results on the MNIST Dataset\n\
Amount of datapoints: {len(train_tsne)}", fontsize=20, weight='bold')
sns.scatterplot("component1", "component2", 
                data=train_tsne, hue=train['label'], 
                palette="deep", legend="full")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel("Component 1", fontsize=14)
plt.ylabel("Component 2", fontsize=14)
plt.legend(fontsize=16);
save_fig('visualization_of_tSNE_plot')

# 12.2. Perform another split for t-sne feature validation.
X_train_tsne, X_val_tsne, y_train_tsne, y_val_tsne = train_test_split(train_tsne, 
                                                                      train['label'], 
                                                                      test_size=0.05, 
                                                                      random_state=42)

# 12.3. Evaluate again by using Extra Trees Classifier.
# Warning: takes time for new run!
if new_run == True:
    tree_clf.fit(X_train_tsne, y_train_tsne)
    y_pred = tree_clf.predict(X_val_tsne)
    tsne_acc = accuracy_score(y_val_tsne, y_pred)
    joblib.dump(tsne_acc, 'saved_var/tsne_acc')
else:
    tsne_acc = joblib.load('saved_var/tsne_acc')
print('Accuracy of Decision Tree Classifier with t-SNE:', tsne_acc)




# %%
# Done