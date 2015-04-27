# Important Data Visualization Demo in Data Science


#### Bar Charts, Histograms, Line Charts
- curve fitting, with MOM and MLE

```python
# Function to plot the MOM and MLE on top of the data
def plot_mom_mle(df, col, ax):
    data = df[col]

    sample_mean = data.mean()
    sample_var = np.sum(np.square(data - sample_mean)) /  (data.count() - 1)

    alpha = sample_mean**2 / sample_var
    beta = sample_var / sample_mean

    # Use MLE to fit a gamma distribution
    ahat, loc, bhat = scs.gamma.fit(df[month], floc=0)
    alpha_mle, beta_mle = ahat, 1./bhat

    gamma_rv = scs.gamma(alpha, beta)
    mle_gamma_rv = scs.gamma(alpha_mle, beta_mle)

    x_vals = np.linspace(data.min(), data.max())

    gamma_p = gamma_rv.pdf(x_vals)
    mle_gamma_p = mle_gamma_rv.pdf(x_vals)

    ax.plot(x_vals, gamma_p, color='r', alpha=0.4, linestyle='--', label='MOM')
    ax.plot(x_vals, mle_gamma_p, color='g', alpha=0.4, label='MLE')
    ax.plot(x_vals, kde_p, color='b', alpha=0.4, label='KDE', linestyle='--' )

    ax.set_xlabel('Rainfall')
    ax.set_ylabel('Probability Density')
    ax.set_title(col)

    ax.set_xlim(0, 12)
    ax.set_ylim(0., .35)

    label = 'alpha = %.2f\nbeta = %.2f' % (alpha, beta)
    ax.annotate(label, xy=(8, 0.3))


months = df.columns - ['Year']
months_df = df[months]

# Use pandas to get the histogram, the axes as tuples are returned
axes = months_df.hist(bins=20, normed=1,
                    grid=0, edgecolor='none',
                    figsize=(15, 10))

# Iterate through the axes and plot the line on each of the histogram
for month, ax in zip(months, axes.flatten()):
    plot_estimate(months_df, month, ax)

```

![Bar plot and line plot](Images/mom_mle.png)

- curve fitting, with KDE in Scipy

```python
from scipy.stats import kde

x1 = np.random.normal(0, 2, 500)
x2 = np.random.normal(4, 1, 500)

# Append by row
x = np.r_[x1, x2]

density = kde.gaussian_kde(x)
xgrid = np.linspace(x.min(), x.max(), 100)
plt.hist(x, bins=50, normed=True)
plt.plot(xgrid, density(xgrid), 'r-')
plt.ylabel('Probability Density')
plt.xlabel('Value')
plt.title("KDE of Bimodal Normal")
```
![kde](Images/kde.png)


#### Scatterplots


#### Feature Importance
- Relative Feature Importance

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
%matplotlib inline

# Load data
boston = datasets.load_boston()
X, y = shuffle(boston.data, boston.target, random_state=13)
X = X.astype(np.float32)
offset = int(X.shape[0] * 0.9)
X_train, y_train = X[:offset], y[:offset]
X_test, y_test = X[offset:], y[offset:]

# Fit regression model
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
          'learning_rate': 0.01, 'loss': 'ls'}
clf = ensemble.GradientBoostingRegressor(**params)

clf.fit(X_train, y_train)
mse = mean_squared_error(y_test, clf.predict(X_test))
print("MSE: %.4f" % mse)

# Plot training deviance

# compute test set deviance
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')

# Plot feature importance
feature_importance = clf.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, boston.feature_names[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
```

![feature importance](Images/feature_importance.png)


#### Principal Component Analysis
- Variance Explained

```python
from sklearn import (cluster, datasets, decomposition, ensemble, lda, manifold, random_projection, preprocessing)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
%matplotlib inline

digits = datasets.load_digits(n_class=6)

# Populating the interactive namespace from numpy and matplotlib

def scree_plot(num_components, pca):
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35, 
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                 ])

    ax.annotate(r"%d%%" % (int(vals[0]*100)), (ind[0]+0.2, vals[0]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%d%%" % (int(vals[1]*100)), (ind[1]+0.2, vals[1]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%d%%" % (int(vals[2]*100)), (ind[2]+0.2, vals[2]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%d%%" % (int(vals[3]*100)), (ind[3]+0.2, vals[3]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%d%%" % (int(vals[4]*100)), (ind[4]+0.2, vals[4]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%d%%" % (int(vals[5]*100)), (ind[5]+0.2, vals[5]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%s%%" % ((str(vals[6]*100)[:4 + (0-1)])), (ind[6]+0.2, vals[6]), va="bottom", ha="center", fontsize=12)
    ax.annotate(r"%s%%" % ((str(vals[7]*100)[:4 + (0-1)])), (ind[7]+0.2, vals[7]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind, 
                       fontsize=12)
    ax.set_yticklabels(('0.00', '0.05', '0.10', '0.15', '0.20', '0.25'), fontsize=12)
    ax.set_ylim(0, .25)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel("Principal Component", fontsize=12)
    ax.set_ylabel("Variance Explained (%)", fontsize=12)

    plt.title("Scree Plot for the Digits Dataset", fontsize=16)
    plt.savefig("scree.png", dpi= 100)

X_centered = preprocessing.scale(digits.data)

# you can also use the StandardScaler
ss = preprocessing.StandardScaler()
ss.fit_transform(digits.data)

pca = decomposition.PCA(n_components=10)
X_pca = pca.fit_transform(X_centered)

scree_plot(10, pca)
```

![PCA Variance Explained](Images/pca_vairance_explained.png)


#### Correlation

```python
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="darkgrid")

rs = np.random.RandomState(33)
d = rs.normal(size=(100, 30))

f, ax = plt.subplots(figsize=(9, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.corrplot(d, annot=False, sig_stars=False,
             diag_names=False, cmap=cmap, ax=ax)
f.tight_layout()
```

![correlation](Images/many_pairwise_correlations.png)

#### Decision Boundary
Example:
- Nearest Neighbors Classification

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
import seaborn as sns
%matplotlib inline
sns.set(style="darkgrid")

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features. We could
                      # avoid this ugly slicing by using a two-dim dataset
y = iris.target

h = .01  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
```

![Decision Boundary pic 1](Images/NB-decision-boundary-1.png)
![Decision Boundary pic 2](Images/NB-decision-boundary-2.png)


#### Network and Graphs

Tutorials:
- [Facebook Friend](http://www.obviousleaks.org/step-by-step-gephi-tutorial/)
- [Introduction to Network Visualization with GEPHI](http://www.martingrandjean.ch/introduction-to-network-visualization-gephi/)

Patent Citation Network
![Patent Citation Network](Images/patent_citation_network.png)

Facebook Friend Network
![Facebook Friend Network](Images/gephi_facebook_friend_network.png)


#### Confusion Matrix


#### ROC Curve


#### Time Series

