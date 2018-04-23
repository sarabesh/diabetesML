
from flask import Flask, flash, redirect, render_template, request, session, abort
import pandas as pd
import numpy as np
from copy import deepcopy

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
      c6=np.random.randint((np.nanmin(X_train.iloc[:, 5].values)),(np.nanmax(X_train.iloc[:, 5].values)),k)
      c7=np.random.randint((np.nanmin(X_train.iloc[:, 6].values)),(np.nanmax(X_train.iloc[:, 6].values)),k)
      c8=np.random.randint((np.nanmin(X_train.iloc[:, 7].values)),(np.nanmax(X_train.iloc[:, 7].values)),k)

      C=np.array(list(zip(c1,c2,c3,c4,c5,c6,c7,c8)), dtype=np.float32)
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

      print(C)
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
        print(y_train[index])
        if(y_train[index]==1):
          yes=yes+1
        else: 
          no=no+1

        i=i+1
      
      if yes>no:
        return 1
      else:
        return 0       


#method for home
@app.route("/")
def home():
    return render_template("indexp1.html")


#method to find colleges input:marks
@app.route('/Predict', methods=['POST'])
def Predict():
    if request.method=='POST':
        pregnancies=request.form['pregnancies']
        glucose=request.form['glucose']
        BP=request.form['BP']
        SkinThickness=request.form['ST']
        Insulin=request.form['Insulin']
        BMI=request.form['BMI']
        DP=request.form['DP']
        age=request.form['age']
    c=0
    #converting string to int
    pregnancies=(float(pregnancies))
    glucose=(float(glucose))
    BP=(float(BP))
    SkinThickness=float(SkinThickness)
    Insulin=float(Insulin)
    BMI=float(BMI)
    DP=float(DP)
    age=float(age)

    #reading data set that contains history of diabetes data
    df = pd.read_csv("PIMA_diabetes.csv")
    # X contains independent variables
    x_train=df.iloc[:,[0,1,2,3,4,5,6,7]]
    # labels contains dependent variable
    y_train=df.iloc[:,[8]]
    
    labelVal=y_train.values.ravel()
    labelVal=labelVal.astype('int')

    x_test=([pregnancies,glucose,BP,SkinThickness,Insulin,BMI,DP,age])
    #call userdefined KNN function - fin contains 7 
    fin=predictKNN(x_train,labelVal,x_test,7)

    fin2=predictKM(x_train,labelVal,x_test,2)



    return render_template("resultknn.html",**locals())

if __name__=='__main__':
    app.run(host='0.0.0.0',port=80,debug=True)
