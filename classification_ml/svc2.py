import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

#x=[1,5,3,8,1,9]
#y=[2,8,1,4,4,9]

#plt.scatter(x,y)
#plt.show()

X=np.array([[1,2],[5,8],[3,6],[8,4],[1,7],[9,9 ]])

Y=[0,1,0,1,0,1]

clf=svm.SVC(kernel='linear',C=1.0)

clf.fit(X,Y)
print(clf.predict([[5,7]]))

w=clf.coef_[0]
#print(w)

a = -w[0]/w[1]
xx=np.linspace(0,10)
yy=a*xx-clf.intercept_[0]/w[1]

f=plt.plot(xx,yy,'k-',label='non weighted div')
plt.scatter(X[:,0],X[:,1],c=Y)
plt.legend()
plt.show()




