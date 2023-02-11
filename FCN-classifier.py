import numpy as np
from pandas import read_csv
#from scipy.io import savemat
from scipy.linalg import qr as qr
from sklearn.preprocessing import MinMaxScaler
import time
from linear_estimator import LinearEstimator

def compute_eq(w, aAssist, cl):

    wShape = np.amax(w.shape[0])
    aAssist = np.copy(aAssist)
    aPrev = np.copy(aAssist)
    weightedSum = w @ aPrev
    aEq = np.copy(np.divide(1, np.add(1, np.exp(np.multiply(np.negative(cl), weightedSum)))))
    #aEq = np.copy(np.divide(np.subtract(np.exp(2*np.negative(cl)*weightedSum),1) , np.add(np.exp(2*np.negative(cl)*weightedSum),1)))
    while np.linalg.norm(aEq-aPrev)> 1e-4: 
        aPrev = np.copy(aEq)
        weightedSum = np.matmul(w, aPrev)
        cl[cl == 1] = -1  
        aEq = np.divide(1, np.add(1, np.exp(cl * weightedSum)))  
        #aEq = np.copy(np.divide(np.subtract(np.exp(2*np.negative(cl)*weightedSum),1) , np.add(np.exp(2*np.negative(cl)*weightedSum),1)))
        for i in range(wShape):
            a = np.copy(np.where(w[i,:]==0))
            if a.shape[1]==wShape:
               aEq[i,0] = np.copy(aAssist[i,0])
    return aEq[:,0]


def buildModel(order,p):
    if p == 0:   
        raise Warning('p = 0')
    elif order == 0:
        modelTerms = np.zeros((1,p), dtype=int)
    elif p == 1:
        modelTerms = np.arange(order,-1,-1).reshape(order+1,1)
    else:
        modelTerms = np.zeros((0,p),dtype= int)
        for i in range(1,-1,-1):
            t = np.copy(buildModel(order-i, p-1))
            nt = t.shape[0]
            repmat = np.full((nt,1),i, dtype=int)    
            repnt = np.concatenate((repmat,t),axis=-1)            
            modelTerms = np.copy(np.concatenate((modelTerms, repnt), axis=0))       
    return modelTerms


class Model: 
    def __init__(self, nb_classes=10, nb_features=84, order=1):
        self.linear_estimator = LinearEstimator()
        self.classNo = nb_classes
        self.featsNo = nb_features
        self.order = order

        x_train = read_csv('train-adam.csv', header=None , nrows=100) 
        y_train = read_csv('data/train_labels.csv', header=None, nrows=100)
        x_test = read_csv('test-adam.csv', header=None, nrows=100)
        y_test = read_csv('data/test_labels.csv', header=None, nrows=100)

        self.initData = x_train.values 
        self.target = np.empty((self.initData.shape[0], self.classNo)) 
        train_labels = y_train.values 
        train_dataset = np.concatenate((self.initData, train_labels), axis=1) 

        self.initDataTest = x_test.values 
        self.targetTest = np.empty((self.initDataTest.shape[0], self.classNo)) 
        test_labels = y_test.values
        test_dataset = np.concatenate((self.initDataTest, test_labels), axis=1)          

        for i in range(train_dataset.shape[0]):
            if train_dataset[i, -1] == 0:
                self.target[i, :] = np.array([0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 1:
                self.target[i, :] = np.array([0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 2:
                self.target[i, :] = np.array([0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 3:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 4:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 5:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 6:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25])
            elif train_dataset[i, -1] == 7:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25])
            elif train_dataset[i, -1] == 8:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25])
            elif train_dataset[i, -1] == 9:
                self.target[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75])

        for i in range(test_dataset.shape[0]):
            if test_dataset[i, -1] == 0:
                self.targetTest[i, :] = np.array([0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 1:
                self.targetTest[i, :] = np.array([0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 2:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 3:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 4:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 5:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 6:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25])
            elif test_dataset[i, -1] == 7:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25])
            elif test_dataset[i, -1] == 8:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25])
            elif test_dataset[i, -1] == 9:
                self.targetTest[i, :] = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75])
                 
        self.labelZero = np.array([[0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
        self.labelOne = np.array([[0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
        self.labelTwo = np.array([[0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
        self.labelThree = np.array([[0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25]])
        self.labelFour = np.array([[0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.25]])
        self.labelFive = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25, 0.25]])
        self.labelSix = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25, 0.25]])
        self.labelSeven = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25, 0.25]])
        self.labelEight = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75, 0.25]])
        self.labelNine = np.array([[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.75]])


        features = self.initData[:, 0:self.featsNo] 
        normalizer = MinMaxScaler(feature_range=(0.25,0.75)) 
        normalizedFeats = normalizer.fit_transform(features) 
        self.dataTrain = np.empty((self.initData.shape[0], self.featsNo + self.classNo)) 
        self.dataTrain[:, 0:self.featsNo] = np.copy(normalizedFeats)
        self.dataTrain[:, self.featsNo:] = np.copy(self.target) 

        featuresTest = self.initDataTest[:, 0:self.featsNo]
        normalizedFeatsTest = normalizer.fit_transform(featuresTest)
        self.dataTest = np.empty((self.initDataTest.shape[0], self.featsNo + self.classNo))
        self.dataTest[:, 0:self.featsNo] = np.copy(normalizedFeatsTest)
        self.dataTest[:, self.featsNo:] = np.copy(self.targetTest)

        self.proj1 = self.proj2 = 1
        self.keepDiag = 0
        self.wToFis = np.empty((self.featsNo + self.classNo, self.featsNo + self.classNo, self.initData.shape[0]))
        self.aEq = np.empty((self.featsNo + self.classNo, self.initData.shape[0]))
        self.aDesToFis = np.empty((self.dataTrain.shape[0], self.featsNo + self.classNo))
        self.cl = np.ones((self.featsNo + self.classNo, 1))
        self.thenValues = np.empty((self.dataTrain.shape[0], self.classNo * (self.featsNo + 1)))
        self.depVarNo = self.thenValues.shape[1]
        self.regressData = np.empty((self.dataTrain.shape[0], self.featsNo + self.depVarNo))
        self.indepVars = np.empty((self.dataTrain.shape[0], self.featsNo))
        self.indepRows = self.indepVars.shape[0]
        self.indepCols = self.indepVars.shape[1]
        self.coefs = np.empty((1, self.indepCols + 1))
        self.coefficients = np.empty((self.depVarNo, self.indepCols + 1))
        self.udo = np.empty((self.order, self.depVarNo))
        self.uda = np.empty((self.order, self.depVarNo))
        self.suda = np.empty((self.order, self.depVarNo))
        self.selectCoefs = np.empty((1, self.coefficients.shape[1]))
        self.weights = np.empty((self.depVarNo, 1))
        self.testEq = np.empty(((self.classNo+self.featsNo), self.dataTest.shape[0]))
        self.output = np.empty((self.dataTest.shape[0], self.classNo))
        self.finalOut = np.empty((self.dataTest.shape[0], self.classNo))
        self.properClassifyNo = 0
        self.comp = np.empty_like(self.finalOut)
        self.score = 0


    def train(self):

        totalsamples = 1

        for i in range(totalsamples):
            for ii in range(self.order, 0, -1):
                self.modelTerms = buildModel(ii, self.indepCols)
            self.all_errors = []

            start = time.time()
            for i in range(self.dataTrain.shape[0]):
                W0 = np.zeros((self.featsNo,(self.featsNo+self.classNo)))
                W1 = np.ones((self.classNo,self.featsNo))
                W2 = np.identity(self.classNo)
                W3 = np.concatenate((W1,W2.T),axis=1)
                W = np.concatenate((W0,W3))
                W = self.linear_estimator(W, self.dataTrain[i, :].reshape(self.featsNo + self.classNo, 1), self.proj1, self.proj2, self.keepDiag)
                self.all_errors.extend(self.linear_estimator.errors)
            
                self.wToFis[:, :, i] = np.copy(W)
                trainAssist = np.copy(self.dataTrain[i, :]).reshape((self.featsNo+self.classNo), 1)
                trainAssist[self.featsNo:,:] = 0.5 
                self.aEq[:, i] = np.copy(compute_eq(W, trainAssist, self.cl))
                self.aDesToFis[i, :] = np.copy(self.aEq[:, i]) 
            end = time.time()
            print("Time needed for linear estimator: " ,(end-start)) 

            for i in range(self.dataTrain.shape[0]):
                self.thenValues[i,:] = np.copy(self.wToFis[:,:,i][self.wToFis[:,:,0] !=0])                
            self.regressData[:, :self.featsNo] = np.copy(self.aDesToFis[:, :self.featsNo])
            self.regressData[:, self.featsNo:(self.featsNo+self.thenValues.shape[1])] = np.copy(self.thenValues) 
            self.indepVars = np.copy(self.regressData[:, :self.featsNo])


            trainOut = np.empty((self.dataTrain.shape[0], self.classNo))
            trainClassifyNo = 0
            compTrain = np.empty_like(trainOut)
            trainRate = 0 

            for kkk in range(self.dataTrain.shape[0]): 
                trainoutAssist = np.copy(self.aDesToFis[kkk, self.featsNo:].reshape(1, self.classNo))
                trainAssistResult = np.argmax(trainoutAssist[0], axis=0)
                if trainAssistResult == 0:
                     trainOut[kkk, :] = np.copy(self.labelZero)     
                elif trainAssistResult == 1:
                     trainOut[kkk, :] = np.copy(self.labelOne) 
                elif trainAssistResult == 2:
                     trainOut[kkk, :] = np.copy(self.labelTwo)  
                elif trainAssistResult == 3:
                     trainOut[kkk, :] = np.copy(self.labelThree)
                elif trainAssistResult == 4:
                     trainOut[kkk, :] = np.copy(self.labelFour) 
                elif trainAssistResult == 5:
                     trainOut[kkk, :] = np.copy(self.labelFive) 
                elif trainAssistResult == 6:
                     trainOut[kkk, :] = np.copy(self.labelSix) 
                elif trainAssistResult == 7:
                     trainOut[kkk, :] = np.copy(self.labelSeven) 
                elif trainAssistResult == 8:
                     trainOut[kkk, :] = np.copy(self.labelEight) 
                elif trainAssistResult == 9:
                     trainOut[kkk, :] = np.copy(self.labelNine)  
                
                compTrain[kkk, :] = np.copy(np.subtract(trainOut[kkk, :], self.dataTrain[kkk, self.featsNo:]))
                if np.count_nonzero(compTrain[kkk, :]) == 0:
                    trainClassifyNo += 1
            trainRate = trainClassifyNo / self.dataTrain.shape[0]

            print("Training Success Rate",trainRate)
        
        return W

    def evaluate(self, W):

        totalclassify=0
        totalsamples=1
        for i in range(totalsamples):

            start1 = time.time()
            for l in range(self.depVarNo):
                depVars = self.regressData[:, self.indepCols + l].reshape((self.indepRows, 1))
                indepScaler = np.copy(np.sqrt(np.diag(np.cov(self.indepVars, rowvar=False)).reshape(self.indepCols, 1)))
                indepVarsStd = self.indepVars @ np.diagflat(1 / indepScaler)
                nt = self.modelTerms.shape[0]
                scaleFact = np.ones((1, nt))
                M = np.ones((self.indepRows, nt))

                # for jj in range(nt):
                #     for kk in range(self.indepCols):
                #         M[:, jj] = np.copy((M[:, jj].reshape(indepVarsStd.shape[0], 1) * (indepVarsStd[:, kk] ** self.modelTerms[jj, kk]).reshape(indepVarsStd.shape[0], 1)).reshape(indepVarsStd.shape[0]))
                #         scaleFact[:, jj] = np.copy(scaleFact[:, jj] / (indepScaler[kk, :] ** self.modelTerms[jj, kk]))

                for jj in range(0, nt):
                    for kk in range(0, self.indepCols, 2):

                        if self.modelTerms[jj, kk] == 1:
                            M[:, jj] = np.copy((M[:, jj].reshape(indepVarsStd.shape[0], 1) * (indepVarsStd[:, kk] ).reshape(indepVarsStd.shape[0],1)).reshape(indepVarsStd.shape[0]))
                            scaleFact[:, jj] = np.copy(scaleFact[:, jj] / (indepScaler[kk, :] ))
                        else:
                            M[:, jj] = np.copy((M[:, jj].reshape(indepVarsStd.shape[0], 1) * 1).reshape(indepVarsStd.shape[0]))
                            scaleFact[:, jj] = np.copy(scaleFact[:, jj])

                        if self.modelTerms[jj, kk + 1] == 1:
                            M[:, jj] = np.copy((M[:, jj].reshape(indepVarsStd.shape[0], 1) * (indepVarsStd[:, kk + 1] ).reshape(indepVarsStd.shape[0], 1)).reshape(indepVarsStd.shape[0]))
                            scaleFact[:, jj] = np.copy(scaleFact[:, jj] / (indepScaler[kk + 1, :])) 
                        else:
                            M[:, jj] = np.copy((M[:, jj].reshape(indepVarsStd.shape[0], 1) * 1).reshape(indepVarsStd.shape[0]))
                            scaleFact[:, jj] = np.copy(scaleFact[:, jj]) 

                q, r, e = qr(M, mode='economic', pivoting=True)
                self.coefs[0, e] = np.copy(np.linalg.solve(r, (q.T @ depVars)).T.reshape(1, nt))
                yHat = np.copy((M @ self.coefs.reshape(nt, 1)).reshape(self.dataTrain.shape[0], 1))
                self.coefs = np.copy(self.coefs * scaleFact)
                self.coefficients[l, :] = np.copy(self.coefs)
                s = np.linalg.norm(depVars - yHat)
                r2 = max(0, 1 - (s/np.linalg.norm(depVars-np.mean(depVars,0))**2))
                adjustedR2 = 1 - (1 - r2)  *  ((self.indepRows-1) / (self.indepRows - nt))
                rmse = np.sqrt(np.mean((depVars - yHat) ** 2))
                self.udo[0,l] = rmse 
                self.uda[0,l] = r2 
                self.suda[0,l] = adjustedR2
            end1 = time.time()
            print("Time needed for qr and coefs: " ,(end1-start1))  


            for iii in range(self.dataTest.shape[0]): 
                for jjj in range(0,self.depVarNo,5):        

                    calculateThis = np.copy(self.dataTest[iii, :self.featsNo].reshape(1, self.featsNo)) 

                    self.selectCoefs = np.copy(self.coefficients[jjj, :]).reshape(-1,1)
                    finalW1 = np.dot(calculateThis , self.selectCoefs[0:self.featsNo,0]) + self.selectCoefs[self.featsNo,0]    
                    self.weights[jjj, 0] = np.copy(finalW1) 

                    self.selectCoefs = np.copy(self.coefficients[jjj+1, :]).reshape(-1,1)
                    finalW2 = np.dot(calculateThis , self.selectCoefs[0:self.featsNo,0]) + self.selectCoefs[self.featsNo,0]    
                    self.weights[jjj+1, 0] = np.copy(finalW2) 

                    self.selectCoefs = np.copy(self.coefficients[jjj+2, :]).reshape(-1,1)
                    finalW3 = np.dot(calculateThis , self.selectCoefs[0:self.featsNo,0]) + self.selectCoefs[self.featsNo,0]    
                    self.weights[jjj+2, 0] = np.copy(finalW3) 

                    self.selectCoefs = np.copy(self.coefficients[jjj+3, :]).reshape(-1,1)
                    finalW4 = np.dot(calculateThis , self.selectCoefs[0:self.featsNo,0]) + self.selectCoefs[self.featsNo,0]    
                    self.weights[jjj+3, 0] = np.copy(finalW4) 

                    self.selectCoefs = np.copy(self.coefficients[jjj+4, :]).reshape(-1,1)
                    finalW5 = np.dot(calculateThis , self.selectCoefs[0:self.featsNo,0]) + self.selectCoefs[self.featsNo,0]    
                    self.weights[jjj+4, 0] = np.copy(finalW5) 

                self.weights[self.weights < -1] = -1
                self.weights[self.weights > 1] = 1

                wFinal = np.copy(W)
                wvar = self.weights.reshape(self.classNo, self.featsNo+1)
                wvar1 = wvar[:,self.featsNo]
                wvar2 = np.diag(wvar1)
                wvar = wvar[:,0:self.featsNo]
                wvar = np.concatenate((wvar,wvar2.T),axis=1)

                for i in range(self.featsNo,self.dataTest.shape[1]):
                    wFinal[i,:] = wvar[i-self.featsNo]       

                testAssist = np.copy(self.dataTest[iii, :]).reshape((self.classNo+self.featsNo), 1)
                testAssist[self.featsNo:,:] = 0.5  
                self.testEq[:, iii] = np.copy(compute_eq(wFinal, testAssist, self.cl))
                self.output[iii, :] = np.copy(self.testEq[self.featsNo:, iii])

            finalOut = np.empty((self.dataTest.shape[0], self.classNo))
            properClassifyNo = 0
            comp = np.empty_like(finalOut)


            for kkk in range(self.dataTest.shape[0]): 
                outAssist = np.copy(self.output[kkk, :].reshape(1, self.classNo))
                assistResult = np.argmax(outAssist[0], axis=0)
                if assistResult == 0:
                     finalOut[kkk, :] = np.copy(self.labelZero)     
                elif assistResult == 1:
                     finalOut[kkk, :] = np.copy(self.labelOne) 
                elif assistResult == 2:
                     finalOut[kkk, :] = np.copy(self.labelTwo)  
                elif assistResult == 3:
                     finalOut[kkk, :] = np.copy(self.labelThree)
                elif assistResult == 4:
                     finalOut[kkk, :] = np.copy(self.labelFour) 
                elif assistResult == 5:
                     finalOut[kkk, :] = np.copy(self.labelFive) 
                elif assistResult == 6:
                     finalOut[kkk, :] = np.copy(self.labelSix) 
                elif assistResult == 7:
                     finalOut[kkk, :] = np.copy(self.labelSeven) 
                elif assistResult == 8:
                     finalOut[kkk, :] = np.copy(self.labelEight) 
                elif assistResult == 9:
                     finalOut[kkk, :] = np.copy(self.labelNine)  
                
                comp[kkk, :] = np.copy(np.subtract(finalOut[kkk, :], self.dataTest[kkk, self.featsNo:]))
                if np.count_nonzero(comp[kkk, :]) == 0:
                    properClassifyNo += 1
            classificationRate = properClassifyNo / self.dataTest.shape[0]

            print("Current Loop Success Rate",classificationRate)
            totalclassify = (totalclassify + classificationRate)

        self.score = (totalclassify/totalsamples) * 100
        print("Total accuracy : " ,self.score)

if __name__ == '__main__':
    model = Model()
    W = model.train()
    model.evaluate(W)