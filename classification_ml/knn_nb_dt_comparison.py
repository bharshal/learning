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
import time
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
	#print(X_test[0])
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
#Number of times pregnant,Plasma glucose concentration a 2 hours in an oral glucose tolerance test,Diastolic blood pressure (mm Hg),
#Triceps skin fold thickness (mm),serum_insulin,Body mass index (weight in kg/(height in m)^2),Diabetes pedigree function,Age (years)
def manual():
	man=tk.Tk()   
#root.geometry("200x200")                                  
	lmain = tk.Label(master=man)
	lmain.grid(row=0,column=0, rowspan=10, padx=10, pady=10)

	info1=Label(man, height=1, width=30,text="Enter Plasma glucose concentration",font=("Helvetica bold", 20))
	textBox1=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info2=Label(man, height=1, width=30,text="Number of times pregnant",font=("Helvetica bold", 20))
	textBox2=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info3=Label(man, height=1, width=30,text="Diastolic blood pressure",font=("Helvetica bold", 20))
	textBox3=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info4=Label(man, height=1, width=30,text="Triceps skin folds thickness",font=("Helvetica bold", 20))
	textBox4=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info5=Label(man, height=1, width=30,text="Serum insulin",font=("Helvetica bold", 20))
	textBox5=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info6=Label(man, height=1, width=30,text="Weight (kg)",font=("Helvetica bold", 20))
	textBox6=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info7=Label(man, height=1, width=30,text="Height (cm)",font=("Helvetica bold", 20))
	textBox7=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info8=Label(man, height=1, width=30,text="Diabetes pedigree function",font=("Helvetica bold", 20))
	textBox8=Text(man, height=1, width=5,font=("Helvetica bold", 20))
	info9=Label(man, height=1, width=30,text="Age",font=("Helvetica bold", 20))
	textBox9=Text(man, height=1, width=5,font=("Helvetica bold", 20))

	info1.grid(row=1,column=0, padx=5, pady=5)
	textBox1.grid(row=1,column=1, padx=5, pady=5)

	info2.grid(row=2,column=0, padx=5, pady=5)
	textBox2.grid(row=2,column=1, padx=5, pady=5)

	info3.grid(row=3,column=0, padx=5, pady=5)
	textBox3.grid(row=3,column=1, padx=5, pady=5)

	info4.grid(row=4,column=0, padx=5, pady=5)
	textBox4.grid(row=4,column=1, padx=5, pady=5)

	info5.grid(row=5,column=0, padx=5, pady=5)
	textBox5.grid(row=5,column=1, padx=5, pady=5)

	info6.grid(row=6,column=0, padx=5, pady=5)
	textBox6.grid(row=6,column=1, padx=5, pady=5)

	info7.grid(row=7,column=0, padx=5, pady=5)
	textBox7.grid(row=7,column=1, padx=5, pady=5)

	info8.grid(row=8,column=0, padx=5, pady=5)
	textBox8.grid(row=8,column=1, padx=5, pady=5)

	info9.grid(row=9,column=0, padx=5, pady=5)
	textBox9.grid(row=9,column=1, padx=5, pady=5)


	buttonCommit=tk.Button(man, height=3, width=15, text="Submit",command=lambda:retrieve_input(man,textBox1,textBox2,textBox3,textBox4,textBox5,textBox6,textBox7,textBox8,textBox9),activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 15))
	#command=lambda: retrieve_input() >>> just means do this when i press the button
	
	buttonCommit.grid(row=10,padx=5,pady=5)
	#a = Button(master=man, text="KNN",height=3,width=15,command=lambda: knn(X_train, X_test, y_train, y_test,choice=1),activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
	#a.grid(row=1,column=0, padx=5, pady=5)
	man.title("Manual Input")   
	man.mainloop()

def retrieve_input(man,a,b,c,d,e,f,g,h,i):
	value=[0,0,0,0,0,0,0,0]
	value[1]=int(a.get("1.0","end-1c"))
	value[0]=int(b.get("1.0","end-1c"))
	value[2]=int(c.get("1.0","end-1c"))
	value[3]=int(d.get("1.0","end-1c"))
	value[4]=int(e.get("1.0","end-1c"))
	wt=int(f.get("1.0","end-1c"))
	ht=(int(g.get("1.0","end-1c")))/100
	value[5]=round((wt/pow(ht,2)),3)
	value[6]=round(float(h.get("1.0","end-1c")),3)
	value[7]=int(i.get("1.0","end-1c"))

	#value=
	classifier = KNeighborsClassifier(n_neighbors=15)
	classifier.fit(X_train, y_train)  #the model gets train using this 
	#print(value)
	value=np.array(value)
	value=value.reshape(1,-1)
	#print(value)
	#print(value.shape)
	y_pred = classifier.predict(value) 
	man.destroy()
	man.quit()

	show=tk.Tk()
	if y_pred==1:
		mess="Congratulations you have diabetes!! :D"
	else:
		mess="Sorry you don't have diabetes :( "

	info1=Label(show,text=mess,font=("Helvetica bold", 20))

	info1.pack()
	show.title("Result")
	show.mainloop()
	#time.sleep(1)
		
	#show.destroy()
	#show.quit()

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
e = Button(master=root, text="Manual Input",height=3,width=15,command=manual, activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
e.grid(row=5,column=0, padx=5, pady=5)
f = Button(master=root, text="Exit",height=3,width=15,command=quit, activebackground="black", activeforeground="cyan", bd=4, bg="#123d63", fg="gold", font=("Helvetica bold", 13))
f.grid(row=6,column=0, padx=5, pady=5)
#text6= tk.Label(root, text= ( 'COUNT = '+ str(det_peeps) ) ,font=("Helvetica bold", 15)).grid(row=4,column=0)
root.title("Classification comparison")   
root.mainloop()