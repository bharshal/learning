#from flask import Flask, flash, redirect, render_template, request, session, abort
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy import genfromtxt
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split  
from sklearn.naive_bayes import GaussianNB
from tkinter.ttk import *
import tkinter as tk
from tkinter import *
import itertools
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

dataset = pd.read_csv("./pima-indians-diabetes.csv");
r=genfromtxt("./pima-indians-diabetes.csv",delimiter=',')
# plt.plot(r)
# plt.legend()
# plt.show()
#dataset.plot(kind= 'box' , figsize = (20, 10))

#dataset.loc[dataset['serum_insulin'] == 0, 'serum_insulin'] = dataset['serum_insulin'].mean()

x = dataset.iloc[:, 0:-1] #contains  columns i.e attributes
y = dataset.iloc[:, -1] #contains labels

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)  

def knn(X_train, X_test, y_train, y_test,choice=0):
	classifier = KNeighborsClassifier(n_neighbors=10)
	classifier.fit(X_train, y_train)  #the model gets train using this 

	#make predictions
	y_pred = classifier.predict(X_test)  

	#compare the accuracy
	accuracy = classifier.score(X_test, y_test)
	conf=confusion_matrix(y_test,y_pred)

	if choice==1:
		plot_confusion_matrix(conf,topic='KNN')
	#print("K nearest neighbour accuracy : " + str(accuracy * 100))
	#print(conf)
	return(accuracy*100,conf)



def nb(X_train, X_test, y_train, y_test,choice=0):
	acc=0
	#print("Executing Naive Bayes")
	model = GaussianNB()
	model.fit(X_train, y_train)
	#Predict Output
	y_pred= model.predict(X_test)
	g=np.array(y_test)
	for i in range(0,(len(y_pred)-1)):
		if y_pred[i]==g[i]:
			acc=acc+1

	acc=(acc/(len(X_test)-1))*100
	#print("Native Bayes accuracy : ",acc)
	conf=confusion_matrix(y_test,y_pred)

	if choice==1:
		plot_confusion_matrix(conf,topic='Naive Bayes')

	#print(conf)
	return(acc,conf)

	#print(predicted)

def dt(X_train, X_test, y_train, y_test,choice=0):
	acc=0
	#print("Executing Decision Tree")
	model = tree.DecisionTreeClassifier(criterion='gini')

	# Train the model using the training sets and check score
	model.fit(X_train, y_train)

	#model.score(X, y)
	#Predict Output

	y_pred = model.predict(X_test)
	g=np.array(y_test)
	for i in range(0,(len(y_pred)-1)):
		if y_pred[i]==g[i]:
			acc=acc+1

	acc=(acc/(len(X_test)-1))*100
	#print("Decision tree accuracy : ",acc)
	conf=confusion_matrix(y_test,y_pred)

	if choice==1:
		plot_confusion_matrix(conf,topic='Decision Tree')

	#print(conf)
	return(acc,conf)

	#print(predicted)

def compare():
	a1=[]
	a2=[]
	a3=[]
	b=[0.1,0.2,0.3,0.4,0.5]
	x = dataset.iloc[:, 0:-1] #contains  columns i.e attributes
	y = dataset.iloc[:, -1] #contains labels

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state = 42)  
	acc,conf=dt(X_train,X_test,y_train,y_test)
	a1.append(acc)
	acc,conf=knn(X_train,X_test,y_train,y_test)
	a2.append(acc)
	acc,conf=nb(X_train,X_test,y_train,y_test)
	a3.append(acc)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state = 42)  
	acc,conf=dt(X_train,X_test,y_train,y_test)
	a1.append(acc)
	acc,conf=knn(X_train,X_test,y_train,y_test)
	a2.append(acc)
	acc,conf=nb(X_train,X_test,y_train,y_test)
	a3.append(acc)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state = 42)  
	acc,conf=dt(X_train,X_test,y_train,y_test)
	a1.append(acc)
	acc,conf=knn(X_train,X_test,y_train,y_test)
	a2.append(acc)
	acc,conf=nb(X_train,X_test,y_train,y_test)
	a3.append(acc)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40, random_state = 42)  
	acc,conf=dt(X_train,X_test,y_train,y_test)
	a1.append(acc)
	acc,conf=knn(X_train,X_test,y_train,y_test)
	a2.append(acc)
	acc,conf=nb(X_train,X_test,y_train,y_test)
	a3.append(acc)

	X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.50, random_state = 42)  
	acc,conf=dt(X_train,X_test,y_train,y_test)
	a1.append(acc)
	acc,conf=knn(X_train,X_test,y_train,y_test)
	a2.append(acc)
	acc,conf=nb(X_train,X_test,y_train,y_test)
	a3.append(acc)

	

	
	plt.plot( b,a1, label = "Decision Tree") 
	plt.plot(b,a2, label = "KNN") 
	plt.plot(b,a3, label = "Naive Bayes") 

	  
	# naming the x axis 
	plt.xlabel('x - axis') 
	# naming the y axis 
	plt.ylabel('y - axis') 
	# giving a title to my graph 
	plt.title('Comparison of Accuracy vs Test/train ratio of various Classification methods') 
	  
	# show a legend on the plot 
	plt.legend() 
	  
	# function to show the plot 
	plt.show() 


def plot_confusion_matrix(cm,topic,classes=['Positive','Negative'],cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix."""
	title='Confusion matrix of '
	title=title+topic


	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	fmt = 'd'
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')
	plt.show()


def quit():
	root.quit()

root=tk.Tk()   
#root.geometry("200x200")                                  
lmain = tk.Label(master=root)
lmain.grid(row=0,column=0, rowspan=10, padx=10, pady=10)
a = Button(master=root, text="KNN",height=3,width=15,command=lambda: knn(X_train, X_test, y_train, y_test,choice=1),activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
a.grid(row=1,column=0, padx=5, pady=5)
b = Button(master=root, text="Naive Bayes",height=3,width=15,command=lambda: nb(X_train, X_test, y_train, y_test,choice=1), activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
b.grid(row=2,column=0, padx=5, pady=5)
c = Button(master=root, text="Decision Tree",height=3,width=15,command=lambda: dt(X_train, X_test, y_train, y_test,choice=1), activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
c.grid(row=3,column=0, padx=5, pady=5)
d = Button(master=root, text="Compare",height=3,width=15,command=compare, activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
d.grid(row=4,column=0, padx=5, pady=5)
e = Button(master=root, text="Exit",height=3,width=15,command=quit, activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
e.grid(row=5,column=0, padx=5, pady=5)
#text6= tk.Label(root, text= ( 'COUNT = '+ str(det_peeps) ) ,font=("Helvetica bold", 15)).grid(row=4,column=0)
root.title("Classification comparison")   
root.mainloop()