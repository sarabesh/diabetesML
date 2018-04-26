
from flask import Flask, flash, redirect, render_template, request, session, abort
import pandas as pd
import numpy as np
from copy import deepcopy
import warnings
import matplotlib.pyplot as plt 
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
import heapq
import os



app=Flask(__name__,static_url_path="/static")

def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)

def predictKM(X_train,y_train,x_test,k):
        
      X=np.array(list(zip(X_train.values[:,:])))
      #initial centroid
      c1=np.random.randint((np.nanmin(X_train.iloc[:, 0].values)),(np.nanmax(X_train.iloc[:, 0].values)),k)
      c2=np.random.randint((np.nanmin(X_train.iloc[:, 1].values)),(np.nanmax(X_train.iloc[:, 1].values)),k)
      c3=np.random.randint((np.nanmin(X_train.iloc[:, 2].values)),(np.nanmax(X_train.iloc[:, 2].values)),k)
      c4=np.random.randint((np.nanmin(X_train.iloc[:, 3].values)),(np.nanmax(X_train.iloc[:, 3].values)),k)
      c5=np.random.randint((np.nanmin(X_train.iloc[:, 4].values)),(np.nanmax(X_train.iloc[:, 4].values)),k)
     

      C=np.array(list(zip(c1,c2,c3,c4,c5)), dtype=np.float32)
      # To store the value of centroids when it updates
      C_old = np.zeros(C.shape)
      # Cluster Lables(0, 1, 2)
      clusters = np.zeros(len(X))
      # Error func. - Distance between new centroids and old centroids
      error = dist(C, C_old, None)
      # Loop will run till the error becomes zero
    
      while error != 0:
    # Assigning each value to its closest cluster
        for i in range(len(X)):
          distances = dist(X[i], C)
          cluster = np.argmin(distances)
          clusters[i] = cluster
        # Storing the old centroid values
        C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
        for i in range(k):
          points = [X[j] for j in range(len(X)) if clusters[j] == i]
          C[i] = np.mean(points, axis=0)
        error = dist(C, C_old, None)
      distances = []
      for i in range(k):
        distance = np.sqrt(np.sum(np.square(x_test - C[i, :])))
        distances.append(distance)

      distances=sorted(distances)
      return predictKNN(X_train,y_train,distances[0],3);



 
def predictKNN(X_train, y_train, x_test, k):
    # create list for distances and targets
      distances = []
      targets = []
      for i in range(len(X_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - X_train.values[i, :])))
        # add it to list of distances
        distances.append([distance, i])
    # sort the list
      distances = sorted(distances)

    # make a list of the k neighbors' targets
      yes=0
      no=0
      i=0
      for i in range(k):
        index=distances[i][1]
        if(y_train[index]==1):
          yes=yes+1
        else: 
          no=no+1

        i=i+1
      
      if yes>no:
        print (1)
        return 1
      else:
        print (0)
        return 0       


#method for home
@app.route("/")
def home():
    return render_template("indexp1.html")


#method to find colleges input:marks
@app.route('/Predict', methods=['POST'])
def Predict():
        #reading data set that contains history of diabetes data
    df = pd.read_csv("PIMA_diabetes.csv")


    #using chi2 to select only 5 features
    array1 = df.values
    X = array1[:,0:8]
    Y = array1[:,8]
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X, Y)
    chires=(fit.scores_)
    print(chires)
    features=heapq.nlargest(5, range(len(chires)), key=chires.__getitem__)
    print(features)


    row_count=df.shape[0]-1
    row_count=row_count/2
    row_count=int(row_count)
    xTrain=df.iloc[:row_count,features]
    yTrain=df.iloc[:row_count,[8]]
    xTest=df.iloc[row_count+1:,features]
    yTest=df.iloc[row_count+1:,[8]]
    print("xTest starts from ")
    print(row_count+1)
    pmean=df.iloc[:,0].mean()
    gmean=df.iloc[:,1].mean()
    bpmean=df.iloc[:,2].mean()
    stmean=df.iloc[:,3].mean()
    imean=df.iloc[:,4].mean()
    bmimean=df.iloc[:,5].mean()
    dpmean=df.iloc[:,6].mean()
    agemean=df.iloc[:,7].mean()
    #pregnanciesdef=float(pregnanciesdef)
    
    if request.method=='POST':
        pregnancies=request.form['pregnancies']
        glucose=request.form['glucose']
        BP=request.form['BP']
        SkinThickness=request.form['ST']
        Insulin=request.form['Insulin']
        BMI=request.form['BMI']
        DP=request.form['DP']
        age=request.form['age']
    
    #if input not given assigning average values from csv
    if not pregnancies:
      pregnancies=pmean
    if not glucose:
      glucose=gmean
    if not BP:
      BP=bpmean
    if not SkinThickness:
      SkinThickness=stmean
    if not Insulin:
      Insulin=imean
    if not BMI:
      BMI=bmimean
    if not DP:
      DP=dpmean
    if not age:
      age=agemean              
    
    #converting string to float
    pregnancies=(float(pregnancies))
    glucose=(float(glucose))
    BP=(float(BP))
    SkinThickness=float(SkinThickness)
    Insulin=float(Insulin)
    BMI=float(BMI)
    DP=float(DP)
    age=float(age)


    # X contains independent variables
    x_train=df.iloc[:,features]
    # labels contains dependent variable
    y_train=df.iloc[:,[8]]
  
    labelVal=y_train.values.ravel()
    labelVal=labelVal.astype('int')

    input=[pregnancies,glucose,BP,SkinThickness,Insulin,BMI,DP,age]
    x_test=[]
    print(input)
    for i in features:
      x_test.append(input[i])
    print(x_test)  
    #call userdefined KNN function - fin contains 7 
    fin=predictKNN(x_train,labelVal,x_test,7)

    fin2=predictKM(x_train,labelVal,x_test,2)
    
    
    co=0
    yTrain=yTrain.values.ravel()
    yTrain=yTrain.astype('int')
    correct=0
    correct2=0
    yTest=list(yTest.values.flatten())
    print(yTest)
    # accuracy for knn
    for i in yTest:   
      if(predictKNN(xTrain,yTrain,xTest.iloc[co,:],7)==yTest[co]):
       correct+=1
        print("yes")
      co+=1
    accu=(correct/len(yTest)*100)



    #co=0
    #with warnings.catch_warnings():
    #  warnings.simplefilter("ignore", category=RuntimeWarning)
    #  for i in yTest:   
    #    if(predictKM(xTrain,yTrain,xTest.iloc[co,:],2)==yTest[co]):
    #      correct2+=1
    #      print("yes")
    #    co+=1
    #  print(correct2/len(yTest)*100)
    print("here")
    plt.scatter(x_train.iloc[:,0],x_train.iloc[:,1],c=labelVal)
    plt.plot(x_test[0],x_test[1],'g*')

    plt.xlabel('Insulin')
    plt.ylabel('glucose')
    os.remove('static/inputvscsv.png')
    plt.savefig('static/inputvscsv.png')
    
    

    return render_template("resultknn.html",**locals())




if __name__=='__main__':
    app.run(host='0.0.0.0',port=80,debug=True)
