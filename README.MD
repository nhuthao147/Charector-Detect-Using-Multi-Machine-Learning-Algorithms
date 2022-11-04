# Charector-Detect-Using-Multi-Machine-Learning-Algorithms
LEARN ABOUT PLAGIARISM
	Learn about plagiarism
	What is Plagiarism?
According to APA Ethic Code Standard 8.11, Plagiarism, Plagiarism is the act of copying another person's words, ideas and images and claiming it as one's own without the consent of the principal employer. Whether a person plagiarizes words or ideas intentionally or unintentionally, plagiarism violates academic ethical standards.

	What should, and shouldn't we do?
	Candlestick
	Avoid copy-patse is recommended as works work through copying without any effort to understand it makes such work stolen.
	Adding citations is necessary because this will help spread the knowledge to others and also show your respect for the author.
	Get permission from the owner of the idea before using it, because the work is not yours, that's why you must get permission from the owner to use it.

	Should not
Do not include other people's work in your work without any citation and understanding of that work.

	Commitment
According to the requirements of the final report form of the Machine Learning course:
We hereby declare that this project is done by our team members. We do not copy, use any documents, source code... of other people without crediting the source. We take full responsibility for plagiarism violations.

 
WEEK 01
	What is machine learning?
Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy.
The first machine learning application: optical character recognition (OCR). However, still not widely used by many users, instead, the spam filter (1990s) was integrated into the email developer's tools to support spam filtering. Some applications of machine learning today: traffic warning, facebook, virtual assistant, text-to-speech…=> Shows that machine learning is ubiquitous in life.
Machine learning is a subfield of Computer Science that is capable of self-learning based on input data without having to be specifically programmed.

	Traditional programming techniques
 

	Machine learning techniques
 

	Advantages and disadvantage of Machine learning
 

	Types of machine learning
	Machine learning classification criteria
With human supervision or not? Supervised learning, unsupervised learning, semi supervised learning, and reinforcement learning.
Learn on the fly or not? Online learning, batch learning.
Use a model or not? Instance-based learning, model-based learning.

	With human supervision or not?
 

Supervised learning (SL) is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. ... This requires the learning algorithm to generalize from the training data to unseen situations in a "reasonable" way (see inductive bias).

In short, Supervised learning uses labeled data. 
Classification, eg., spam filter, image classification, … 
Regression, eg., housing prediction, weather forecast, …
Most important algorithms include K-Nearest Neighbors, Linear Regression, Logistic Regression, Support Vector Machines (SVMs), Decision Trees and Random Forests, Neural networks.

 

Unsupervised learning: use unlabeled data.
Clustering: most important algorithms include K-Means, DBSCAN, Hierarchical Cluster Analysis (HCA) 
Anomaly detection: One-class SVM, Isolation Forest
Dimensionality reduction: Principal Component Analysis (PCA), Locally Linear Embedding (LLE), t-distributed Stochastic Neighbor Embedding (t-SNE)
-Association rule learning: Apriori, Eclat.

	Learn on the fly or not?
	Online learning, batch learning.

	Use model or not?
	Instance-based learning, model-based learning.
Use model?
Instance-based learning compares new data to the learned examples using a similarity measure.

 

Model-based learning is the formation and subsequent development of mental models by a learner. Most often used in the context of dynamic phenomena, mental models organize information about how the components of systems interact to produce the dynamic phenomena.

	Reinforcement learning
Agents are taught using rewards.
E, g. robots learn to walk, DeepMind’s AlphaGo.
 

WEEK 02
	Main tasks in Machine Learning
	Select a learning algorithm
	Train it on some data
The two things that can cause problems: “bad data” and “bad algorithm”:
	“Bad data”: the data that is labeled incorrectly, is full of errors, has missing values or is otherwise poor in quality. Or too complex for the algorithm to handle.
	“Bad algorithm”: algorithm that predicts and gives wrong results.
	Bad data
	Insufficient quantity of data
A good size but not enough data will have a bad effect.

 

	This is a diagram analized and displays the performance of 4 algorithms based on their abilities to read and understand texts and words. We can see, from the beginning, experiment around 400 million of words, Winnow (blue line) is the worst algorithm and Memory-Based (red line) is the best algorithm but from time to time, as the number of words increased till it reached 1000 millions of words, the worst performance Winnow algorithm suddenly became the best performance and the opposite happened Memory-Based.

	Non representative data
In order to generalize the model well, it is crucial that the training data be an accurate representation of the population. In other words, each time a new sample is derived from the population, it is crucial that the sample must accurately paint a picture of the population. A training set of data must be representative of the cases you want to generalize to. It is, however, harder than it sounds. If the sample is too small, you will have sampling noise, which is the non representative data as a result of chance, but even large samples can be non representative if the sampling method is flawed. This is called Sampling Bias.
We need training data that are representative of the new cases.
Two types: 
	Noisy data is meaningless data, any data that has been received, stored or changed in such a manner that it cannot be read or used by the program that originally created it can be described as noisy.
	Sampling bias: get 1 branch of data from 1 big data group.

	Missing data
	Instance are missing
	Ignore features: drop feature - only suitable when column has too much data missing.
	Ignore instances: removing instances damages data in some cases
	Filling instances: fill in the blanks to help minimize data loss.

	Irrelevant features
	Feature selection: also known as variable selection, attribute selection or variable subset selection, is the process of selecting a subset of relevant features (variables, predictors) for use in model construction.
	Feature extraction: Feature extraction helps to reduce the amount of redundant data from the data set. In the end, the reduction of the data helps to build the model with less machine's efforts and increases the speed of learning and generalization steps in the machine learning process.
	Creating new features: the process of creating features (also called "attributes") that don't already exist in the dataset.

	Bad Algorithms
	Overfitting
Overfitting refers to a model that models the training data too well.
Overfitting happens when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance of the model on new data. This means that the noise or random fluctuations in the training data is picked up and learned as concepts by the model. The problem is that these concepts do not apply to new data and negatively impact the model’s ability to generalize.
Overfitting is more likely with nonparametric and nonlinear models that have more flexibility when learning a target function. As such, many nonparametric machine learning algorithms also include parameters or techniques to limit and constrain how much detail the model learns.

	Underfitting
Underfitting refers to a model that can neither model the training data nor generalize to new data.
An underfit machine learning model is not a suitable model and will be obvious as it will have poor performance on the training data.
Underfitting is often not discussed as it is easy to detect given a good performance metric. The remedy is to move on and try alternate machine learning algorithms. Nevertheless, it does provide a good contrast to the problem of overfitting.

	Testing and Hyperparameter tuning
	Test set and generalization error
Data before being processed will be divided into 2 types: training set and test set. If the training set is very low but the test set is very high => overfitting.

	Hyperparameter tuning
Hyperparameter: is a parameter that we do not receive through training but must provide before running training. Using too many unnecessary parameters will lead to overfitting. Conversely, too few parameters will lead to underfitting.
The process of finding the best model for the data is called tuning.

	Validation set
To evaluate a good model, need to run on test error, run training error only during training. However, the test error data is not objective during the learning process => Use validation set.
Consists of 3 phases in the ML process: test set, validation set, and training. Rule (similar to training set): the validation set, and the test must be representative of the data you expect to use in production.

 
WEEK 03
Big picture about End-to-end ML project.
	Exploring the data
Main step of ML project:
	Look at the big picture: Overview of the project: what is the end goal of the project, request from the customer, what algorithms to use, ...
	Get the data (raw data): The good algorithms but the data is bad we can not do anything.
	Discover and visualize the data to gain insights: use python or some other tool to visualize the data to gain insights.
	Prepare the data for machine learning algorithms: Some algorithms can not work with text data, or some datasets have missing value, outlier data… so we must prepare the data before training.
	Train and evaluate models: Try some models to find the most optimization. 
	Fine tune your models: After finding out the most suitable model, fine tune to increase the accuracy for that model.
	Analyze and test your solution: Test the model with real data in read life.
	Launch, monitor, and maintain your system.

 

 

Example: Predict the house price project.
Get the data to see in specific rows and columns (loc, iloc).

 

	Discover and visualize the data to gain insights
 

	Histogram: divide the data into multiple ranges and count the total sample in each range (diagonal from top left corner to bottom right corner).

 

	Correlation between features: the similarities or differences of 2 features. If the data in one feature has an increased trend and the others feature have the same trend => the correlation is high (similarities trend).
 

	Create new features if we think it will help to show the trend of the data.

 

	Prepare the data

 

	Remove unused features
	Split train and test data (we can split the validation data if we want more accurate in real data).

	2 ways to split the data:
	Method 1: randomly select 20% of data, 80% of data for training data.
	Method 2: if the data set is not large, we can use stratified sampling to split the data. After split data by this method the histogram for the feature we want to keep will be same in train and test data.

 

	Histogram of training and test set are same

 

Separate labels from data. (Label to prediction).
 

Pipeline: Put all jobs (worker) we want to do into one process pipeline. Data will be done all that jobs when go through pipeline.
Define worker.

 

Pipeline for categorical features:
	Select column are text.
	Process missing value.
	Change text data to number data (one hot vector).

 

Change text data to number data.
 

Add our combination feature we think before to our feature if we think it good.

 

Pipeline for numerical feature:
	Select column are number.
	Process missing value.
	Add new attribute.
	Feature scaling.
 

Feature scaling: change all feature become the range from -1 to 1 (depend on the coder) to handle the outlier data.

 

 

WEEK 04
Main steps of an End-to-end ML project.

	Look at the big picture
	What is the end goal of the project? 
	How do the existing solutions (if any) perform?
	What algorithms to use? 
	What performance measure is relevant?
	What data we need to collect?

	Get data
Some data repositories:
	UCI Machine Learning Repository.
	Kaggle datasets.
	Amazon’s AWS publid datasets.
	http://dataportals.org.
	http://opendatamonitor.eu.
	Wikipedia’s list of Machine Learning datasets.
	Datasets subreddit.

	Discover the data to gain insights
	Look at the top five rows using the DataFrame’s head () method.
 
	The info () method is useful to get a quick description of the data, in particular the total number of rows, each attribute’s type, and the number of nonnull values.
 

	Find out what categories exist and how many districts belong to each category by using the method.
	The method shows a summary of the numerical attributes.

	Prepare the data
	Remove unused features
	Split training-test set and NEVER touch test set until test phase.
	Separate labels from data since we do not process label values.
	Define pipelines for processing data.
	Combine features transformed by two below pipelines.
	Pipeline for categorical features.
	Pipeline for numerical features.
	Run the pipeline to process training data.

	Train and evaluate models
	Training: learn a hypothesis using training data.
	Compute R2 score and root mean squared error.
	Predict labels for some training instances.
	Store models to files, to compare latter.
	Evaluate with K-fold cross validation.

	Fine-tune models
Method: Grid search (try all combinations of hyperparams in param_grid).

	Analyze and test your solution
	Pick the best model - the SOLUTION.
	Analyse the SOLUTION to get more insights about the data.
	Run on test data:
	Compute R2 score and root mean squared error.
	Predict labels for some test instances.

	Launch monitor and maintain
	Write monitoring code: catch system breakage and performance degradation.
	Human evaluation: analyze system’s output regularly (by experts, workers…).
	Monitor system’s input: poor quality input, e.g., blur images from cameras, noisy signal from sensors, can degrade system performance.
	Retrain the models on a regular basis using fresh data.
	Save snapshots of the system regularly: to roll back to a previously working state in case of a crash.
 
WEEK 05
	Classification
Classification in machine learning is of the supervised learning type. This means that the classification correction program will learn from the data provided and create a list of observations, each of which is a list of attributes that can represent a lot of different things such as building height, room number, etc., these properties are usually divided into two categories:  Number and Categories.
Then based on their observations and labels that conduct the classification of data processing we have the observation sets. As a result, when a model without a label is included, we can classify and give its label correctly.
MNIST dataset: A set of 70,000 small images of digits (0 – 9) handwritten by high school students and employees of the US Census Bureau.
Each image (28 x 28 pixel) is labeled with the digit it represents.

	Classification forms
	Binary Classification
Binary classification is a form of classification - the process of predicting classification variables - in which output is limited to two layers.
For example, spam detection (true or false), Predicting stock ups and downs tomorrow (up or down).
Typically, the label involves two classes of the condition: abnormal and normal, with normal being 0 and abnormalities being 1.
For this type of binary classification, it is common to use the Bernoulli distribution prediction model as a discrete distribution that produces a binary result of 0 or 1.
Commonly used algorithms:
	Logistic Regression.
	k-Nearest Neighbors.
	Decision Trees.
	Support Vector Machine.
	Naive Bayes.

	Multi-Class Classification
A multi-category classification is the type of classification where the data has more than 2 types.
Classifying many of these categories does not attribute to abnormal or normal types but categorizes them to the class we already know. The number of layers can be very large depending on the problem that we work on such as the identification of Latin letters with 26 different letters, we have 26 such classes.
For this type of classification, it is common to use the Multinoulli distribution prediction model as a discrete distribution that produces probability-making predictions for each class.
Commonly used algorithms:
	k-Nearest Neighbors.
	Decision Trees.
	Naive Bayes.
	Random Forest.
	Gradient Boosting.

	Multi-Label Classification
Multi-label classification is the type of classification where the data may have more than one type of label.
Classifying multiple labels tells us the list of labels we already know. The number of labels we can have been as many as photo classification, we can have a photo with a lot of objects inside so there will be many labels or belong to many layers.
For this type of classification, it is common to use the Bernoulli distribution prediction model as a discrete distribution that produces probability-making predictions for each class.
Commonly used algorithms:
	Multi-label Decision Trees.
	Multi-label Random Forests.
	Multi-label Gradient Boosting.

	Multi-Output Classification
Multioutput classification is simply a concept of multilabel classification in which each label can be multiclass (i.e., it has more than 2 possible values).
For example, build a perturbation classification system in the image and it will knock out the noise points on the input image by making a series of pixel intensity changes. Its output is multilabel (1 label per pixel) and each label has a multi-label value (the value of pixels ranges from 0 to 255). The classification line between classification and regression is sometimes unclear. It can be said that predicting pixel values is more like regression than classification.
The classification line between classification and regression is sometimes unclear. It can be said that predicting pixel values is more like regression than classification.

	Process
	Step 1: Prepare the dataset and feature extraction. Pick out good features, remove bad features, jamming. Calculate the number of features accordingly to ensure the speed of running and accuracy of the data.
	Step 2: Build a classifier model. This is a learning step or training to find a function f(x) = y with x being input features. This step we use supervised learning algorithms such as Logistic Regression, Naive Bayes...
	Step 3: Check the data with the make prediction. This is the step of bringing in new data to check the accuracy of the model obtained in step 2.
	Step 4: Evaluate the layering model and choose the best model. This step assesses the level of errors in the data of the model that we have found. Thereby changing the parameters to get the best model.

 
WEEK 06
	Multiclass classification is a classification task with more than two classes. Each sample can only be labeled as one class.
	Multiclass classification strategies: OneVsAll and OneVsOne.

	One-versus-all classifier
	Definition
It is implemented in OneVsRestClassifier. If we have n classes => create M classifiers.
	Advantage and Disadvantage
	Advantage: build less classifier
	Disadvantage: can be imbalanced

	One-versus-one classifier

	Definition
It constructs one classifier per pair of classes. If we have n classes => distinguish classifiers.
	Advantage and Disadvantage
Advantage: avoid being imbalance, uses only a fraction of data => Less time

	Classification with MNIST
	MNIST dataset
The MNIST dataset is a collection of 70,000 small images of handwritten digits by high school students and U.S. Census Bureau employees. Each image is labeled with the number it represents. This set has been studied so much that it is often referred to as the "Hello World" set of Machine Learning.
Add the Keras library and at the same time take two training sets and test sets and then put the data to form a dataset where a data set has 784 features because one of its data is a photo with a size of 28x28 pxl in which the training set has 60000 data, and the test set has 10000 data:
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

Draw an image in the MNIST dataset in the first place in the training set
def plot_digit(data, label = 'unspecified', showed=True):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary)
    plt.title("Digit: " + str(label))
    #plt.axis("off")
    if showed:
        plt.show()
sample_id = 0
plot_digit(X_train[sample_id], y_train[sample_id])


 
Here the first data of the training set is the number 5.

	Performing training in a binary classification
First, we create a label array with two values True and False: True with data is 5 and False with data is another digit 5.
y_train_5 = (y_train == 5) 
y_test_5 = (y_test == 5)

Train data using the fit function
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)


Predictive testing using the predict function
sgd_clf.predict([X_train[sample_id]])


	Check the accuracy of the model
Accuracy (with cross-validation) of SGDClassifier
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")


Precision score is a formula-based accuracy rating value:
Precision score=  (True Positive)/(False Positive + True Positive)
Recall score is a formula-based memory rating value
Recall score=  (True Positive)/(False Positive +  False Negative)
F1 Score is a score that consider both accuracy and memory and is based on precision score and recall score. Calculated by formula:
F1 score= 2*  (Precision score* Recall score)/(Precision score+ Recall score)
Precision, recall and f1_score
from sklearn.metrics import precision_score, recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)
from sklearn.metrics import f1_score
f1_score(y_train_5, y_train_pred)


	Perform MNIST dataset training using MultiClass Classification
Train data using the fit function
sgd_clf.fit(X_train, y_train)


Predictive testing using the predict function with data at 0:
sample_id = 0
sgd_clf.predict([X_train[sample_id]])


To see scores from classifiers
sgd_clf.classes_
sample_scores = sgd_clf.decision_function([X_train[sample_id]]) 
class_with_max_score = np.argmax(sample_scores)


Force sklearn to run OvO (OneVsOneClassifier) or OvA (OneVsRestClassifier)
from sklearn.multiclass import OneVsRestClassifier
ova_clf = OneVsRestClassifier(SGDClassifier(random_state=42))
ova_clf.fit(X_train, y_train)
len(ova_clf.estimators_)
ova_clf.classes_
sample_scores = ova_clf.decision_function([X_train[sample_id]]) 

from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, y_train)
len(ovo_clf.estimators_) 
ovo_clf.classes_
sample_scores = ovo_clf.decision_function([X_train[sample_id]]) 


Perform an accuracy measurement when running a classification algorithm
sgd_acc=0;
sgd_acc = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")


Perform data scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))


	Perform MNIST dataset training using MultiLabel Classification
Create a muti-label labels:
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]


Training with KneighborsClassifier:
from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)


Multi-label classifier rating by f1_score:
y_train_knn_pred = 0;
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
f1_score(y_multilabel, y_train_knn_pred, average="macro") 

	Perform MNIST dataset training using MultiOutput Classification
Make noise points on the dataset
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

Train data using the fit function
knn_clf.fit(X_train_mod, y_train_mod)


Try prediction
sample_id = 12
clean_digit = knn_clf.predict([X_test_mod[sample_id]])    
if let_plot:
    plt.figure(figsize=[12,5])
    plt.subplot(131); 
    plot_digit(X_test_mod[sample_id],str(y_test[sample_id])+" (input SAMPLE)",showed=False)
    plt.subplot(132); 
    plot_digit(clean_digit,str(y_test[sample_id])+" (PREDICTION)",showed=False)
    plt.subplot(133); 
    plot_digit(y_test_mod[sample_id],str(y_test[sample_id])+" (LABEL)",showed=True)


 
WEEK 07
In this chapter we will start by looking at the Linear Regression model, one of the simplest models there is.
	Using a direct “closed-form” equation that directly computes the model parameters that best fit the model to the training set (i.e., the model parameters that minimize the cost function over the training set).
	Using an iterative optimization approach called Gradient Descent (GD) that gradually tweaks the model parameters to minimize the cost function over the training set, eventually converging to the same set of parameters as the first method. 
	Look at a few variants of Gradient Descent: Batch GD, Mini-batch GD, and Stochastic GD
	Polynomial Regression is a more complex model that can fit nonlinear datasets. Since this model has more parameters than Linear Regression, it is more prone to overfitting the training data, so we will look at how to detect whether or not this is the case using learning curves, and then we will look at several regularization techniques that can reduce the risk of overfitting the training set.

	Linear regression
Linear Regression is a machine learning algorithm based on supervised learning. It performs a regression task. Regression models a target prediction value based on independent variables. It is mostly used for finding out the relationship between variables and forecasting. Different regression models differ based on – the kind of relationship between dependent and independent variables, they are considering, and the number of independent variables being used.
Linear regression performs the task to predict a dependent variable value (y) based on a given independent variable (x). So, this regression technique finds out a linear relationship between x (input) and y(output). Hence, the name is Linear Regression.

	Hypothesis function
	Definition 
Hypothesis functions is a tool to help Machine Learning programs predicts and find optimal weights through the Cost Function, which will help make these predictions more accurate.
	Equation
Hypothesis function for Linear Regression:
y ̂=θ_0 x_0+ θ_1 x_1+θ_2 x_2+⋯+ θ_n x_n

In this equation:
	y ̂ is the predicted value.
	n is the number of features.
	x is the i-feature value.
	θ_j is the jth model parameter (including the bias term θ_0 and the feature weights θ_1,θ_2,…,θ_n).

This can be written much more concisely using a vectorized form, as shown in
y ̂=h_θ (x)= θ∙x

In this equation:
	θ is the model’s parameter vector, containing the bias term θ0 and the feature weights θ1 to θn.
	x is the instance’s feature vector, containing x0 to xn, with x0 always equal to 1.
	θ∙x is the dot product of the vectors θ and x, which is of course equal to θ_0 x_0+ θ_1 x_1+θ_2 x_2+⋯+ θ_n x_n
	h is the hypothesis function, using the model parameters θ.

	Cost function
	Definition
By achieving the best-fit regression line, the model aims to predict y value such that the error difference between predicted value and true value is minimum. So, it is very important to update the θ1 and θ2 values, to reach the best value that minimize the error between predicted y value (pred) and true y value (y).
Cost function means squared error (MSE) is the most common performance measure for Linear Regression (LR).

	Calculation formula
Cost function of Linear Regression is the Root Mean Squared Error (RMSE) between predicted y value (pred) and true y value (y).

 
In this formula:
	hθ: is the hypothesis function
	x(i): the output 
	y(i): the label
	hθ(x(i)): The prediction
	i: the samples
	(1/m)^1: the average error
	Methods to minimize the cost function MSE
To find parameter set θ that minimize the cost function MSE, we have 2 ways:
Method 1: Normal Equation
To find the value of θ that minimizes the cost function, there is a closed-form solution - in other words, a mathematical equation that gives the result directly. This is called the Normal Equation
θ ̂=〖(X^T X)〗^(-1) X^T y

In this equation:
	θ ̂  is the value of θ that minimizes the cost function.
	y is the vector of target values containing y (1) to y(m).
The pseudoinverse itself is computed using a standard matrix factorization technique called Singular Value Decomposition (SVD) that can decompose the training set matrix X into the matrix multiplication of three matrices U Σ VT The pseudoinverse is computed as X+ = VΣ+UT. To compute the matrix Σ+, the algorithm takes Σ and sets to zero all values smaller than a tiny threshold value, then it replaces all the nonzero values with their inverse, and finally it transposes the resulting matrix.
This approach is more efficient than computing the Normal Equation, plus it handles edge cases nicely: indeed, the Normal Equation may not work if the matrix XTX is not invertible (i.e., singular), such as if m < n or if some features are redundant, but the pseudoinverse is always defined.

Method 2: Gradient descent
Gradient Descent is a generic optimization algorithm capable of finding optimal solutions to a wide range of problems. The general idea of Gradient Descent is to tweak parameters iteratively in order to minimize a cost function.
Gradient Descent: it measures the local gradient of the error function with regard to the parameter vector θ, and it goes in the direction of descending gradient. Once the gradient is zero, you have reached a minimum!
An important parameter in Gradient Descent is the size of the steps, determined by the learning rate hyperparameter. 
Diagram also illustrates the fact that training a model means searching for a combination of model parameters that minimizes a cost function (over the training set). It is a search in the model’s parameter space: the more parameters a model has, the more dimensions this space has, and the harder the search. Fortunately, since the cost function is convex in the case of Linear Regression, the needle is simply at the bottom of the bowl.
Gradient descent variants:
	Batch gradient descent
	Stochastic gradient descent
	Mini-batch gradient descent

	Batch gradient descent (BGD)
Partial derivative of the cost function:
 

Calculate all derivatives reduced by 1 formula:
 

	Compare to Normal Equation and Gradient Descent

Normal Equation	Gradient Descent
Not choosing learning rate (alpha).	Choose learning rate (alpha).
It is analytical approach.	It is an iterative algorithm.
Normal equation works well with small number of features.	Gradient descent works well with large number of features.
No need for feature scaling.	Feature scaling can be used.
If (XTX) is non-invertible, regularization can be used to handle this.	No need to handle non-invertibility case.
Algorithm complexity is O(n3).
n is the number of features.	Algorithm complexity is O(kn2).
n is the number of features.

	Computational Complexity
The Normal Equation computes the inverse of XTX, which is an (n + 1) × (n + 1) matrix (where n is the number of features). The computational complexity of inverting such a matrix is typically about O(n ) to O(n ), depending on the implementation. The SVD approach used by Scikit-Learn’s LinearRegression class is about O(n2). If you double the number of features, you multiply the computation time by roughly 4.
Both the Normal Equation and the SVD approach get very slow when the number of features grows large (e.g., 100,000). On the positive side, both are linear with regard to the number of instances in the training set (they are O(m)), so they handle large training sets efficiently, provided they can fit in memory.
Also, once you have trained your Linear Regression model (using the Normal Equation or any other algorithm), predictions are very fast: the computational complexity is linear with regard to both the number of instances you want to make predictions on and the number of features. In other words, making predictions on twice as many instances (or twice as many features) will take roughly twice as much time.
 
WEEK 08
	Gradient descent
	Problems of Gradient descent
	Local minima

 

We need to choose the right alpha learning speed. The learning rate determines the magnitude of the step per repetition. 
	If α is too small, then convergence will take a long time and the algorithm will need many steps to finish.
	If α is too large then it may not converge and exceed the minimum value, the algorithm may not finish.
When we choose the wrong initialozation value, we will not achieve the global mininum. The convergence point will be at local mininum or plateau.
How to fix? Choose the cost function is MSE, the equation will be parabol then it not has the local minima problem.

	Feature scales
 
If it is on a very different scales, you have to scale the data. If you do not do so, the level curves (contours) would be narrower and taller which means it would take longer time to converge.
 

	STOCHASTIC GRADIENT DESCENT (SGD)
	Description

 

	Much faster than Batch gradient Descent (BGD).
	Uses much less memory than BGD.

	Properties of SGD
 

	Much faster than BGD.
	Used much less memory than BGD.
	Runs “randomly”: not decreases cost function on every iteration => Benefit.
	When at (closed to) the minimum: continue to bounce around.
	To fix it: Specify the number of loops.

	Implementation of SGD

 

If learning rate reduced too fast: may stop at local minimal.
If the learning rate reduced too slowly: when reaching the solution, SGD still doesn't stop but continues to run around the result point.
To decrease learning rate properly, we create a function called learning schedule.
	BGD => use all samples to calculate the derivative  => Slower speed, but the value of the exact derivative.
	SGD => Use only 1 sample to calculate the derivative => Faster speed, but the value of the derivative is incorrect.

	Mini-batch gradient descent
The difference:
	BGD uses all samples to calculate the derivative.
	SGD only uses 1 sample to calculate the derivative.
	MBGD uses a number of samples in the data set to calculate the derivative.

 

Fewer random moves than SGD.
The resulting derivative is more accurate than SGD.

k << m
	If k = m => It’s the BGD.
	If k = 1 => It’s the SGD.
	Comparison of LR training algorithms
 

m: a number of samples.
n: a number of features.
 
WEEK 09
	Polinomial regression
	Hypothesis function
Polynomial regression is a form of linear regression that allows you to predict a single y variable by decomposing the x variable into a n-th order polynomial. Rather than using a straight line, so a linear model to estimate the predictions, it could be for instance a quadratic model or cubic model with a curved line.
When our data is non-linear (greater than degree one) then we must use model have higher degree to demonstrate the data (Polinomial regression).

 

To train Polinomial regression model:
	Add new features (high-order features).
	Run Linear Regression training algorithms (e.g, Normal equation, gradient descent).

	Methods to identify underfitting / overfitting
Use cross-validation (a technique making it possible to compare models without the need for a separate validation set).
	Underfitting models: If a model performs poorly on both.
	Overfitting models: If a model performs well on the training data but generalizes poorly according to the cross-validation metrics.
Use learning curves: plot performance against training set size (or the training iteration).

	Methods to solve underfitting / overfitting
	Learning curve on training data
	With few instances:  The model can fit them perfectly, which is why the curve starts at zero.
	With more data: It becomes impossible for the model to fit the training data perfectly, both because the data is noisy and because it is not linear at all.
	Learning curve on validation data
	Models trained on few instances: It is incapable of generalizing properly, which is why the validation error is initially quite big.
	More training examples: It learns and thus the validation error slowly goes down.
	Compared to the learning curve of linear model
	Both errors on training and validation data: learning curves are typical of an underfitting model.
	Gap between curves: Reached a plateau, they are close and fairly high.
	Compared to the learning curve of polynomial model (degree 10)
	The error on the training data is much lower than with the Linear Regression model.
	There is a gap between curves. The model performs particularly better on training data than on the validation data. If you used a much larger training set, the two curses would continue to get closer.
	To overcome these errors
	Bias error: due to wrong assumptions, such as assuming that the data is linear when it is actually quadratic. A high-bias model is most likely to underfit the training data.
	Variance error: due to the model’s excessive sensitivity to small variations in the training data. A model with many degrees of freedom is likely to have high variance, and thus to overfit the training data.
	Irreducible errors: due to the noisiness of the data itself. The only way to reduce this part of the error is to clean up the data.

	Type of errors in machine learning
Having 3 typical error make model overfitting and underfitting
	Bias errors: wrong assumption (Choose wrong model- Our data in degree 2,3 but we choose model degree 1). High bias => underfitting model.
	Irreducible errors: noisy data (the error in the data not from the model – the data is error or wrong).
	Variance errors: too sensitive model (Model change too easily to fit the data even minor change but not point out the trend of the data). High Variance => Overfitting model.
To overcome these errors (To overcome underfitting):
	Bias error: use another model (usually is the more complex models).
	Variance error (To overcome overfitting):
	Add more data:
	Use the simpler model
	Regularization
	Early stopping (prevent overfitting due to long training)
	Irreducible error: clean up data.

	Regularization
Constrain your model so that it minimizes overfitting.
	Idea
	Use a complex model: to avoid underfitting.
	Constrain a model (if model underfitting): to decrease underfitting.

	Methods
	Ridge regression.
	Lasso regression.
	Elastic net.

	Ridge regression (l2 – norm|distance regression)

Ridge Regression: a regularized version of Linear Regression.
Cost function of Ridge regression:

 

The hyperparameter α controls how much you want to regularize the model. If α = 0 then Ridge Regression is just Linear Regression. If α is very large, then all weights end up very close to zero and the result is a flat line going through the data’s mean.
Notes:
	Cost function for training can be different from the performance measure of the model.
	Feature scaling is important for regularized models to work properly.

Ridge regression: square theta to make the theta have negative value

 

When value of theta are small it will make the model more simple (regularization term decrease the effect to the large theta).

 

Notes:
	Cost function for training can be different from the performance measure of the model.
	Feature scaling is important for regularized models to work properly.

	Lasso regression (l1 – norm|distance regression)
Lasso Regression: another regularized version of Linear Regression: just like Ridge Regression, it adds a regularization term to the cost function, but it uses the ℓ1 norm of the weight vector instead of half the square of the ℓ2 norm.
Cost function of Lasso regression:

 

Lasso regression is stronger than ridge regression because |ɵ| is stronger than ɵ2 ((-0,01)2   vs |-0.01|).
Lasso regression can help us select or remove feature.

	Elastic net
Elastic Net: a middle ground between Ridge Regression and Lasso Regression.
Cost function of Elastic net:

 

	r = 1 => l1 norm (Lasso regression)
	0 < r < 1 => combination l1 and l2
	r = 0 => l2 norm (Ridge regression)
Listed in order of preference (in practice):
	No regulazation (least preferred) 
	Ridge regulazation
	Lasso regulazation
	Elastic net (most preferred) 

 
WEEK 10
	Early stopping
	Theory
Early Stopping is one of the effective ways to overcome Overfitting.
When the model is been Overfitting, we have this chart with RMSE and Epoch (number of times run).

 

Like this chart we can see Learning Curves of Validation set doesn’t decrease when we increase the number of times run. It's because the more times we train a model, the more optimal we can have with the training set and that model becomes to be Overfitting.
To overcome that overfitting, we will train this model while MSE still decreases and stop training when MSE starts increasing. That way is called Early Stopping.
Stop training as soon as the validation error reaches a minimum.

	Coding
	Step 9.2: Create a dataset 2 degrees (4+ x + 2*x2)
	Step 9.3: Pipeline add “poly_features” to use Polynomial Regression
	Step 9.4: Use SGD Regression to train Linear model, but because we have added poly_features before so this model has the same effect with Polynomial Regression.
 

Next, we use”. fit” to train this model. We will use a “for” loop to run fit round by round. Control the number of rounds by Epoch. When the error become bigger than minimum value, we can know that is the minimum, we have to stop right this and save the minimum.

 

	Logistic regression
	Theory
Logistic Regression is a Classifier model with the outcome 0 and 1, it is called binary classifier.
Logistic functions have a regression value from 0 to 1, if this value is smaller than 0,5 Logistic Regression, we return 0, and opposite if this value bigger than 0,5 Logistic Regression will return 1.
The predicted probability:
 
With logistic function:
 

	Hypothesis function of logistic regression
Hypothesis function has a Sigmoid function chart:

 
Value range when drawing Logistic function between 0 and 1
	Cost function of logistic regression
Cost function of logistic regression with 1 sample will be solve by
 
We can see if p ⩾ 0.5 that mean is the models will return the right answer with low error c(𝛉), and opposite.
With many samples we have log loss function like:
 
Gradients of the log loss function:
 
Decision boundaries if the line that splits space into 2 parts, one part is a class.
If hypothesis bigger than 0,5 logistic regression will return 1 and marked by a green dot in below picture.
Opposite, if hypothesis smaller than 0,5 logistic regression will return 0 and marked by a red dot in below picture.
The dot on the wrong side is the model's error.
	Train with Log Loss Cost Function
 

No closed-form solution (normal equation).
Log loss is convex => No local minima so GD always can find the global minima.
	Training using Gradient Descent
 
Decision Boundary 
Split space into 2 parts:
 
 
 

	Softmax Regression
	Theory
Softmax regression (a.k.a. Multinomial Logistic Regression) is a multi-label classifier (labels: binary vectors).
Given a classification of 𝐾 classes, the hypothesis function of class 𝑘:
 
Where 𝑠_𝑘 (𝐱)=𝐱𝛉^((𝑘)) is called the score of class 𝑘 for the sample 𝐱.
Loss function of Softmax Regression is cross entropy
 
To solve gradient descent of the cross-entropy cost function with regards to parameters 𝛉^((𝑘)) of class k
 
Use Softmax Regression in the same way with logistic regression.
To use Softmax Regression we have to set “multi_classs” to “multinomial”.

 

The same way with solve Decision boundaries in logistic regression, we have the chart split class like below picture

 

	Code
Step 11.2: Like Logistic regression we use 2 features: petal length, petal width in line 754.
We still use LogisticRegression function but set multi_class = “multinomial” in line 756.
 

To predict 1 sample in Softmax Regression we have to solve 3 parameters. In the picture below, that is Hypothesis of class 1. And to plot a chart with 3 parameters we will use a contour plot (from high look down) and to show the height in this case we use class 1 with the green color, so the sample is nearer green area, the Hypothesis is closer 1.

 

 
WEEK 11
	Support vector machine (SVM)
Introduction
A Support Vector Machine (SVM) is a very powerful and versatile ML model. Sometimes it is better than deep learning, because it is faster. SVM is very useful:
	Classification: linear or nonlinear.
	Regression.
	Outlier detection.
SVM is the most popular ML model in ML. 
 

	Classification with linear SVM models
Linear SVM is a binary classifier, with 0 or 1 is the output.
->Coding.
In [1]: import datasets (iris)
In line 27: Choose petal length and petal width are 2 features for this classifier.
In line 29: We chose 2 classes for this classifier: setosa and versicolor.
In part 1.2: create 2 Decision boundaries of arbitrary models.
 

In part 1.3: Create The third Decision boundaries by SVC (SVM Classifier)
Parameter C in line 42 for control overfitting. It is larger, overfitting has less regularization. In this case, linear SVM is hard to be overfitting, so we set C as infinite.
In part 1.4: Plot Decision boundaries.
Parameter margin in line 57 is the broken line at fig 3 in the next picture.

 

After plot Decision boundaries, we can see fig 1 and 2 is the random Decision boundaries we create in part 1.2 and fig 3 is the Decision boundaries that we create by SVC.
 
We can see in fig 2 and Decision boundaries both can classify 2 classes. So how can we choose the better Decision boundaries? That is the larger margin. A decision boundary with a large margin will be better if we add more samples in our dataset.

	Large Margin Classification
The margin is the nearest sample’s distance with Decision boundaries.
Support vectors are samples determined on margin.
 
In this picture, the red dots are Support vectors.
We have 2 group of margins:
	Hard margin: all samples are on 1 side of the margin. But there are 2 problems we can see here. First problem, when we add an outlier sample, the model will have a big change. Second problem, if the dataset is a nonlinear model, we can’t do SVM anyway.
	Soft margin: allow some samples on 2 sides of the margin. By that way, we will have a larger margin, but more errors will appear. We can control the Margin by parameter C in part 1.3 in our code. The lower C is the larger margin (soft margin) and less overfitting. Larger margin is the better generalization. The below picture shows us How the parameter C can affect SVC.
 

	Support vector
 
Support vectors are sample determine (“support”) margin.
	Hard margin
All sample of one class -> one side of margin. If 
Problem:
	If data is non-linear, we can not find the result.
	Outlier data will make big affect to the model. 
 
	Soft margin
Allow some sample in another side => to get the large margin
 
 
	Classification with non-linear SVM
	Method1: add polynomial features
 
 

The larger degree the more complex model we can learn but the number of feature increase very large.
Kernel tricks help us to get the result as the polynomial but we do not need to add feature in explicit ways.
If our model overfitting we can use regularization.
If our model underfitting we can use kernel trick.

 

Add new feature is the distance from every point to landmark => We can find the linear to classified it.
 

Eucldean distance: Add new feature is the distance from every point to landmark => We can find the linear to classified it.
String kernel: only popular on text data. To calculate the distance between two different strings.

 
 

X: sample
l: landmark
Gamma: larger gamma more concentrate around landmark.

 
 

 

 

How to select landmark?

 

 

The larger the gamma, the more overfitting (try to go through all the data points).

 

 

Kernel trick: how to make kernel import data in implicit ways. 
Kernel: To learn nonlinear data.

	Method 2: add similarity features.
Choose a landmark and measure the distance from sample to landmark.
The farer sample with landmark, the higher feature x2 (x2 (similarity feature) is distance from data to landmark) of this sample. So, we can find a linear boundary to split datas.
 

Use Gaussian (RBF) kernel function to solve distance:
 
 
WEEK 12
	SVM regression
	Idea
In SVM Classification we find a model (decision boundary) with large margin and less violations.
With SVM Regression we find a model (hypothesis) across datasets that fit more samples within narrow margins.

 

	Implementation
SVM Regression tries to fit more samples within narrow margins.
We have a parameter that is epsilon. The smaller epsilon, the more narrow the margin.

 

	The bigger epsilon will have larger margin.
 

 

Code:
In part 6.1: create a nonlinear data.
In part 6.2: use LinearSVR to train with this data. It will be underfitting, but we can see how epsilon affects the model. 
 

SVM tries to fit more samples within narrow margins.
We can see that the smaller the epsilon, the more narrow the margin.

 

However, they are both not good because they are both underfitting.
A good model should fit more samples in narrow margins.
SVR will support us to train a nonlinear model. Set “kernel=”poly”” like you add a polynomial feature. We still have epsilon and C parameters.
 

The smaller epsilon, the less data fitted. The smaller C, the less overfitting.

 

	Hypothesis function
 
Where ℎ(𝐱)=𝐱𝐰+𝑏=0 is the decision function (score).

	Hard margin
 
With a hard margin we will have a 100% score.

	Soft margin
  
s(i) if slack variable.
Slack variable allows h(x(i)) to have a wrong result.
Parameter C can help us to control how wrong the results model allows.
If C is large, s will be small so the violation will be less.

	Solve constraint problems
Primal problem: original
Dual problem: closely related with primal problem (lower brunel - solution for primal problem)
Under some condition: solution for Dual problem is the same solution for Primal problem.
So we can find solutions for The dual of SVM problems to solve SVM problems.

SVM primal problem with new notation:
 ’
SVM dual problem:
 

SVM dual problems cost less time than SVM primal problems but with the same solution.
Closed-form solution:
 
 
SVM dual problem enables kernel trick.

	Kernel trick
Kernel trick gives the same solution without explicitly adding features.
We can solve the multiplication of 2 vectors (2 samples) in new space by using multiplication of 2 vectors in old space. (Don’t need to create a new vector).
 
 

	Cost function (for linear SVM)
 
For large-scale non-linear problems: neural networks.
 
WEEK 13
	Decision Trees
Decision Trees (DTs): are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features. A tree can be seen as a piecewise constant approximation.
For instance, in the example below, decision trees learn from data to approximate a sine curve with a set of if-then-else decision rules. The deeper the tree, the more complex the decision rules and the fitter the model.
 
Make Predictions
 

The condition tree will always start at node #0, and the remaining nodes at the last node are the leaf node where the decision tree is executed, and the "PETAL LENGTH" line is the line that asks the question to decide if the next step is correct. or wrong.
Explain:
	#0, #1, #3, #4: node order
	PETAL LENGTH <=2.45: picture
	class = SETOSE, class= VERSICOLOR, class = VIRGINICA: is predict.

	Estimating class probabilities
First, we put in a set of 150 data. Continuing reading the program will read the data at "Value" at 3 classes, respectively, SETOSA is 100%, VERSICO is about 95%, and continues to decrease after that.
Now suppose you find a flower, and the petal length is greater than 2.45 cm. You must move down to the root’s right child node (depth 1, right), which is not a leaf node, so it asks another question: is the petal width smaller than 1.75 cm? If it is, then your flower is most likely an Iris-Versicolor (depth 2, left). If not, it is likely an Iris-Virginica (depth 2, right). It’s really that simple.
NOTE: One of many qualities of Decision Trees is that they require very little data preparation. They don't require feature scaling or centering at all.
Explanation of components in a node:
	A node's samples attribute counts how many training instances it applies to.
	A node's value attribute counts how many training instances of each class this node applies to.
	A node's gini attribute measures its impurity: a node is "pure" (gini=0) if all training instances it applies to belong to the same class.

 

For example, the depth-2 left node has a gini score equal to
 .

NOTE: Scikit-Learn uses the CART algorithm, which produces only binary trees: non-leaf nodes always have two children. However other algorithms such as ID3 can produce Decision Trees with nodes that have more than 2 children.
 
	Regularised Hyper-parameters
Decision Trees make very few assumptions about the training data (as opposed to linear models, which obviously assume that the data is linear, for example). If left unconstrained, the tree structure will adapt itself to the training data, fitting it very closely, and mostly overfitting it. Such a model is often called a nonparametric model, not because it does not have any parameters but because the number of parameters is not determined prior to training, so the model structure is free to stick closely to the data.
In contrast, a parametric model such as a linear model has a predetermined number of parameters, so its degree of freedom is limited, reducing the risk of overfitting.
To avoid overfitting the training data, you need to restrict the Decision Tree's freedom during training. (regularisation)
	min_samples_split: the minimum number of samples a node must have before it can be split
	min_samples_leaf: the minimum number of samples a leaf node must have
	min_weight_fraction_lead: same as min_samples_leaf but expressed as a fraction of the total number of weighted instances
	max_leaf_nodes: maximum number of leaf nodes
	max_features: maximum number of features that are evaluated for splitting at each node
	*Increasing min_* hyper-parameters or reducing max_* hyper-parameters will regularise the model. *
 

	Estimating Class Probabilities
A Decision Tree can also estimate the probability that an instance that an instance belongs to a particular class k: first it traverses the tree to find the leaf node for this instance, and then it returns the ratio of training instances of class k in this node.
The corresponding leaf node is the depth-2 left node, so the Decision Tree should output the following probabilities: 0% for Iris-Setosa (0/54), 90.7% for Iris-Versicolor (49/54), and 9.3% for Iris-Virginica (5/54). And of course, if you ask it to predict the class, it should output Iris-Versicolor (class 1) since it has the highest probability.

>>> tree_clf.predict_proba([[5, 1.5]])
array([[ 0. ,  0.90740741,  0.09259259]])
>>> tree_clf.predict([[5, 1.5]])
array([1])

Notice that the estimated probabilities would be identical anywhere else in the bottom-right rectangle.

	The CART Training Algorithm
Scikit-Learn uses the Classification and Regression Tree (CART) algorithm to train Decision trees. The idea is quite simple: the algorithm first splits the training set in two subsets using a single feature k and a threshold Tk. How does it choose k and Tk? It searches for the pair (K, Tk) that produces the purest subsets (weighted by their size). The cost function:
 

Once it has successfully split the training set in two, it splits the subsets using the same logic, then the subsets and so on, recursively. It stops recursing once it reaches the maximum depth (defined by the max_depth hyper-parameter), or if it cannot find a split that will reduce impurity. A few other hyper-parameters control additional stopping conditions (min_samples_split,min_samples_leaf,min_weight_fraction_leaf, andmax_leaf_nodes).

WARNING:
The CART algorithm is a greedy algorithm: it greedily searches for an optimum split at the top level, then repeats the process at each level. It does not check whether or not the split will lead to the lowest possible impurity several levels down. A greedy algorithm often produces a reasonably good solution, but it's not guaranteed to be the optimal.

	Regression
Decision Trees are also capable of performing regression tasks.

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)
 
The tree looks very similar to the classification tree built earlier. The main difference is that instead of predicting a class in each node, it predicts a value. The prediction is simply the average target value of the 110 training instances associated with this leaf node.

 

Notice how the predicted value for each region is always the average target value of the instances in that region. The algorithm splits each region in a way that makes most training instances as close as possible to that predicted value.
The CART algorithm works mostly the same way as earlier, except that instead of trying to split the training set in a way that minimises impurity, it now tries to split the training set in a way that minimises the MSE.

 

Just like for classification tasks, Decision Trees are prone to overfitting when dealing with regression tasks.
 Instability
Decision Trees are simple to understand and interpret, easy to use versatile and powerful. But Decision Trees also have some disadvantages.
Firstly, Decision Trees love orthogonal decision boundaries (all splits are perpendicular to an axis), which makes them sensitive to training set rotation. Although both Decision Trees fit the training set perfectly, it is very likely that the model on the right side will nit generalise well. One way to limit this problem is to use PCA, which often results in a better orientation of the training set.

 
More generally, the main issue with Decision Trees is that they are very sensitive to small variations in the training data. Since the training algorithm used by Scikit-Learn is stochastic you may get very different models even on the same training data (unless you set the random_state hyper-parameter).
 
Random Forests can limit this instability by averaging predictions over many trees.

	Advantages and disadvantages of Decision Trees
	Advantages
The decision tree is a simple and popular algorithm. This algorithm is widely used because of its benefits:
	The model produces easy-to-understand rules for the reader, creating a code with each branch of leaves being a law of the tree.
	Input data can be missing data, without standardizing or creating false variables.
	Can work with both digital and classification data.
	Can authenticate the model using statistical tests.
	It's likely to be about big data.

	Disadvantages
	Decision trees make splits: perpendicular to a feature axis. Solution: to use PCA algorithm.
	In training, features are randomly selected to evaluate at each node. Output models changed from one train to another.
 

Solution: to use Random Forest.
Decision trees: usually overfitting problems.

 
WEEK 14
	Ensemble learning
Ensemble learning is the process by which multiple models, such as classifiers or experts, are strategically generated and combined to solve a particular computational intelligence problem. Ensemble learning is primarily used to improve the (classification, prediction, function approximation, etc.).
Ensemble methods are learning models that achieve performance by combining the opinions of multiple learners. Typically, an ensemble model is a supervised learning technique for combining multiple weak learners or models to produce a strong learner with the concept of Bagging and Boosting for data sampling.

	Pros of ensemble learning
Generally, ensembles have higher predictive accuracy. Test results improve with the size of the ensemble. That is why, ensembles are often challenge winners. Each technique has its own characteristics. For example, in data wrangling and tuning options. Tweaking makes models fit better.

	Cons of ensemble learning
However, model ensembles are not always better. New observations can still confuse. That is, ensembles cannot help unknown differences between sample and population. Ensembles should be used carefully.
Is it understood? Ensembles can be more difficult to interpret. Sometimes, even the very best ideas cannot be sold to decision makers. Sometimes, the best ideas are not accepted by the final users.

Finally, ensembles cost more to create, train, and deploy. The ROI of an ensemble approach should be considered carefully. Generally, more complexity is not good in of itself. KISS. We have found that a full one-third of IS systems failure is due to complexity.

	Voting methods
	Idea
	Train many multiple predictors.
	Prediction most voted result of predictor.

 

	Hard voting classifiers
Hard voting classifier classifies input data based on the mode of all the predictions made by different classifiers. The majority voting is considered differently when weights associated with the different classifiers are equal or otherwise.
Majority Voting based on equal weights: When majority voting is taken based equal weights, mode of the predicted label is taken. Let’s say there are 3 classifiers, clf1, clf2, clf3. For a particular data, the prediction is [1, 1, 0]. In case, the weights assigned to the classifiers are equal, the mode of the prediction is taken. Thus, the mode of [1, 1, 0] is 1 and hence the predicted class to the record becomes class 1. For equal weights, the equation in fig 1 gets simplified to the following:

 

Diagrammatically, this is how the hard voting classifier with equal weights will look like:
 

	Soft Voting Classifiers
Soft voting used when all the parts used in the set can predict the class's probabilities. the predictions of a category division are averaged to predict the class with the highest probability (weighted vote).
With equal weights, the probabilities will get calculated as the following:
Prob of Class 0 = 0.33*0.2 + 0.33*0.1 + 0.33*0.8 = 0.363
Prob of Class 1 = 0.33*0.8 + 0.33*0.9 + 0.33*0.2 = 0.627

 

As in class with very predictive books the highest average is defined as the last class of prediction.

	Bagging method
Bagging is a powerful ensemble method that was proposed by Leo Breiman in 1994 to prevent overfitting. The concept behind bagging is to combine the predictions of several base learners to create a more accurate output. Bagging is the application of the Bootstrap procedure to a high-variance machine learning algorithm, typically decision trees.
	Suppose there are N observations and M features. A sample from observation is selected randomly with replacement (Bootstrapping). 
	A subset of features are selected to create a model with sample of observations and subset of features. 
	Feature from the subset is selected which gives the best split on the training data. 
	This is repeated to create many models and every model is trained in parallel 
	Prediction is given based on the aggregation of predictions from all the models.

	Diversity is the key
To get diverse classifiers:

	Method 1: Use different types of classifiers.
Examples: SVM, Random Forest Classifier, Logistic Regression, …
	Method 2: Train classifiers (of 1 type) using.
Use only one algorithm but train on different training sets.

	Training using different training sets
	Step 1: Create different training sets using random sampling.
	Step 2: Train predictors and aggregate them to predict.
This method is called “Bagging”, it means bootstrap aggregating.

	Get different training sets
Random sampling methods:
	Bootstrapping: we take part of the training sets and are replaced by a copy of them.
	Non-bootstrapping: we take part of the training sets and are not replaced by a copy of them

	Computation of predictors: can be in parallel
This method can be performed on the training sets (random sampling) at the same time.
This method is great for scale.

 

	Sampling features
How to get different training sets?
	Method 1: Random sampling of instances.  
	Method 2: Random sampling of features.

	Boosting methods
Boosting is a general ensemble method that creates a strong classifier from several weak classifiers. This is done by building a model from the training data, then creating a second model that attempts to correct the errors from the first model.

 

	AdaBoost
	Idea: pay more attention to mistaken samples.
	Method: increase weights (probability selected) of mistaken examples.
	Drawback of boosting:

 

 

⇒ The biggest disadvantage of this algorithm is that it takes time because the implementation is serial. This is also a common sloiwj of most boot algorithms.

	Gradient boosting
Gradient boosting is a machine learning technique for regression, classification and other tasks, which produces a prediction model in the form of an ensemble of weak prediction models, typically decision trees
 

Like AdaBoost, Gradient Boosting sequentially adds predictors to correct its predecessor.
 
Combine predictions by adding: 
True label y = h1() + error of h1() (h2)

 

 

More predictors, more accurate: true label y = h1+h2+h3.

 
 
REFERENCES
[1.]	Aurélien Géron (2019, September 05). Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow (2nd edition). O’Reilly Media, Inc.
[2.]	Kevin P. Murphy (2012, June). Machine Learning: A Probabilistic Perspective. The Massachusetts Institute of Technology.
[3.]	Vu Huu Tiep (2019, June 15). Machine Learning co ban. Retrieved form https://machinelearningcoban.com/
 
