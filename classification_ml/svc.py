from sklearn import datasets 
from sklearn import svm
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)
digits =datasets.load_digits()
#print(digits.data)
clf=svm.SVC(gamma=0.001,C=1000)
x,y= digits.data[:-1],digits.target[:-1]
clf.fit(x,y)
#print(digits.data)
#print(digits.target[-2])
print(clf.predict(digits.data[[-2]]))

plt.imshow(digits.images[-2],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()