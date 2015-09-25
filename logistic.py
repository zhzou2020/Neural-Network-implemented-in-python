import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib

matplotlib.rcParams['figure.figsize'] = (10.0,8.0)

#generate a dataset
np.random.seed(0)
X,y = sklearn.datasets.make_moons(200,noise = 0.20)
#plt.scatter(X[:,0],X[:,1],s = 40,c = y,cmap = plt.cm.Spectral)
#plt.show()

#train the logistic classifier
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X,y)

#draw boundary
def plot_decision_boundary(pred_func):
	x_min,x_max = X[:,0].min()-.5,X[:,0].max()+.5
	y_min,y_max = X[:,1].min()-.5,X[:,1].max()+.5
	
	# Generate a grid of points with distance h between them
	h=0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole gid
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# Plot the decision boundary
plot_decision_boundary(lambda x: clf.predict(x))
plt.title("Logistic Regression")
plt.show()
