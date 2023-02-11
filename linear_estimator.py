import numpy as np
import math

class LinearEstimator:
    def __call__(self, w, aDesPrev, proj1, proj2, keepDiag):
        wInit, wPrev = np.copy(w), np.copy(w)
        alpha = 0.9
        c = 1
        self.errors = []
        invAdes = np.copy(np.log(np.divide(aDesPrev, (1-aDesPrev))))
        #invAdes = np.copy(0.5*np.log(np.divide((1+aDesPrev), (1-aDesPrev))))
        normDiv =  np.add(c, (np.dot(aDesPrev.T, aDesPrev))) 
        errorVector = np.copy((invAdes - (np.matmul(w, aDesPrev))) / normDiv.item())
        error = np.linalg.norm(errorVector)
        self.errors.append(error)

        while error > 1e-5: 
            if keepDiag == 1:
                w = np.copy(wInit)
            
            for i in range(np.amax(w.shape[0])): 
                e_i = errorVector.item(i,0) 
                w[i,:] = np.copy(np.add(w[i,:], alpha * e_i * aDesPrev[i].T))  

            w[wInit == 0] = 0
            if proj1 == 1:
                w[w < -1] = -1
                w[w > 1] = 1
            if proj2 == 1:
                wNorm = 0
                for i in range(np.amax(w.shape[0])):
                    wNorm = wNorm + np.linalg.norm(w[i,:])**2 
                wNorm = math.sqrt(wNorm)
                if wNorm > 4:
                #if wNorm > 1:
                    #w = np.copy(w * (0.95/wNorm))
                    w = np.copy(w * (3.95/wNorm))
            if np.linalg.norm(w-wPrev) < 1e-6:
                return wPrev
            wPrev = np.copy(w)
            errorVector = np.copy(np.subtract(invAdes, np.dot(w, aDesPrev)) / normDiv) 
            error = np.linalg.norm(errorVector)
            self.errors.append(error)

        return w
          