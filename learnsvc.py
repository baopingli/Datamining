from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification



if __name__ == '__main__':

 X, y = make_classification(n_features=4, random_state=0)
 print(X)
 print(y)
 clf = LinearSVC(random_state=0, tol=1e-5)
 clf.fit(X, y)
