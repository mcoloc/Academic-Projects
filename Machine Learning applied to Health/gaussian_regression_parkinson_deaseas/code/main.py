# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
ID=301227
FILENAME="parkinsons_updrs.csv"


class  Regression():
    
    def readFile(self):
        plt.close('all')
        self.xx=pd.read_csv(FILENAME) # read the dataset
        self.z=self.xx.describe().T # gives the statistical description of the content of each column
        #xx.info()
        # features=list(xx.columns)
        self.features=['subject#', 'age', 'sex', 'test_time', 'total_UPDRS',
               'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
               'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
               'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
       
    def generateData(self): 
        # scatter plots
        todrop=['subject#', 'sex', 'test_time',  
               'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
               'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
               'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA']
        self.x1=self.xx.copy(deep=True)
        self.X=self.x1.drop(todrop,axis=1)
        # Generate the shuffled dataframe
        np.random.seed(ID)
        self.Xsh = self.X.sample(frac=1).reset_index(drop=True)
        [self.Np,self.Nc]=self.Xsh.shape
        self.F=self.Nc-1
        # Generate training, validation and testing matrices
        self.Ntr=int(self.Np*0.5)  # number of training points
        self.Nva=int(self.Np*0.25) # number of validation points
        self.Nte=self.Np-self.Ntr-self.Nva   # number of testing points
        self.X_tr=self.Xsh[0:self.Ntr] # training dataset
        # find mean and standard deviations for the features in the training dataset
        self.mm=self.X_tr.mean()
        self.ss=self.X_tr.std()
        self.my=self.mm['total_UPDRS']# get mean for the regressand
        self.sy=self.ss['total_UPDRS']# get std for the regressand
        
    def normalization(self): 
        # normalize data
        self.Xsh_norm=(self.Xsh-self.mm)/self.ss
        self.ysh_norm=self.Xsh_norm['total_UPDRS']
        self.Xsh_norm=self.Xsh_norm.drop('total_UPDRS',axis=1)
        self.Xsh_norm=self.Xsh_norm.values
        self.ysh_norm=self.ysh_norm.values
        # get the training, validation, test normalized data
        self.X_train_norm=self.Xsh_norm[0:self.Ntr]
        self.X_val_norm=self.Xsh_norm[self.Ntr:self.Ntr+self.Nva]
        self.X_test_norm=self.Xsh_norm[self.Ntr+self.Nva:]
        self.y_train_norm=self.ysh_norm[0:self.Ntr]
        self.y_val_norm=self.ysh_norm[self.Ntr:self.Ntr+self.Nva]
        self.y_test_norm=self.ysh_norm[self.Ntr+self.Nva:]
        self.y_train=self.y_train_norm*self.sy+self.my
        self.y_val=self.y_val_norm*self.sy+self.my
        self.y_test=self.y_test_norm*self.sy+self.my
                 
    def plotErrorHistogram(self, title):
        #plots the histogram of errors between y_hat annd y for train and test sets
            e=[self.err_train,self.err_test]
            plt.figure()
            plt.hist(e,bins=50,density=True,range=[-8,17], histtype='bar',label=['Train.','Val.','Test'])
            plt.xlabel('error')
            plt.ylabel('P(error in bin)')
            plt.legend()
            plt.grid()
            plt.title(f'Error histogram - {title}')
            v=plt.axis()
            self.N1=(v[0]+v[1])*0.5
            self.N2=(v[2]+v[3])*0.5
            
    def plotRegressorLine(self, title):
         #plot y_hat(y) showing graphically the distances between the correct values
         plt.figure()
         plt.plot(self.y_test,self.yhat_test,'.b')
         plt.plot(self.y_test,self.y_test,'r')
         plt.grid()
         plt.xlabel('y')
         plt.ylabel('yhat')
         plt.title(title)
         v=plt.axis()
         self.N1=(v[0]+v[1])*0.5
         self.N2=(v[2]+v[3])*0.5

    def plotRegressorLineWithErrors(self, title):
        #plot y_hat(y) showing graphically the distances between the correct values
        #plotting also a vertical bar that indicates the range in which the y_hat is estimated to be with the probability of 99.7%
        plt.figure()
        plt.errorbar(self.y_test,self.yhat_test,yerr=3*self.sigmahat_test*self.sy,fmt='o',ms=2)
        plt.plot(self.y_test,self.y_test,'r')
        plt.grid()
        plt.xlabel('y')
        plt.ylabel('yhat')
        plt.title(f"{title} with errorbars")
        v=plt.axis()
        self.N1=(v[0]+v[1])*0.5
        self.N2=(v[2]+v[3])*0.5
                
    def plotErrorStatistics(self):
        #prints statistics about MSE, mean error, standard deviation and R^2
        print('MSE train',round(np.mean((self.err_train)**2),3))
        print('MSE test',round(np.mean((self.err_test)**2),3))
        #print('MSE valid',round(np.mean((self.err_val)**2),3))
        print('Mean error train',round(np.mean(self.err_train),4))
        print('Mean error test',round(np.mean(self.err_test),4))
        #print('Mean error valid',round(np.mean(self.err_val),4))
        print('St dev error train',round(np.std(self.err_train),3))
        print('St dev error test',round(np.std(self.err_test),3))
        #print('St dev error valid',round(np.std(self.err_val),3))
        print('R^2 train',round(1-np.mean((self.err_train)**2)/np.std(self.y_train**2),4))
        print('R^2 test',round(1-np.mean((self.err_test)**2)/np.std(self.y_test**2),4))
        #print('R^2 val',round(1-np.mean((self.err_val)**2)/np.std(self.y_val**2),4))
                     
            
class GPRclass(Regression):         
        
    def GPR(self,X_train,y_train,X_val,r2,s2):
        #Estimates the output y_val given the input X_val, using the training data and  hyperparameters r2 and s2
        Nva=X_val.shape[0]
        yhat_val=np.zeros((Nva,))
        sigmahat_val=np.zeros((Nva,))
        for k in range(Nva):
            x=X_val[k,:]# k-th point in the validation dataset
            A=X_train-np.ones((self.Ntr,1))*x
            dist2=np.sum(A**2,axis=1)
            ii=np.argsort(dist2)
            ii=ii[0:self.N-1];
            refX=X_train[ii,:]
            Z=np.vstack((refX,x))
            sc=np.dot(Z,Z.T)# dot products
            e=np.diagonal(sc).reshape(self.N,1)# square norms
            D=e+e.T-2*sc# matrix with the square distances 
            R_N=np.exp(-D/2/r2)+s2*np.identity(self.N)#covariance matrix
            R_Nm1=R_N[0:self.N-1,0:self.N-1]#(N-1)x(N-1) submatrix 
            K=R_N[0:self.N-1,self.N-1]# (N-1)x1 column
            d=R_N[self.N-1,self.N-1]# scalar value
            C=np.linalg.inv(R_Nm1)
            refY=y_train[ii]
            mu=K.T@C@refY# estimation of y_val for X_val[k,:]
            sigma2=d-K.T@C@K
            sigmahat_val[k]=np.sqrt(sigma2)
            yhat_val[k]=mu        
        return yhat_val,sigmahat_val

    def GPRimplementation(self):
        # Apply Gaussian Process Regression
        self.N=10
        r2_len=100
        s2_values=[1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5, 1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2, 6e-2, 7e-2, 8e-2, 9e-2, 1e-1]
        s2_len=len(s2_values)
        r2_values=np.zeros((r2_len),float)
        r2=0
        MSE_valid_matrix=np.zeros((r2_len),float)
        self.MSEmin=10
        for j in range (s2_len):
            s2=s2_values[j]
            r2=0
            print(f"{s2}\n")
            for i in range(r2_len):
                r2=r2+0.1
                r2_values[i]=r2
                yhat_train_norm,sigmahat_train=self.GPR(self.X_train_norm,self.y_train_norm,self.X_train_norm,r2,s2)
                #yhat_train=yhat_train_norm*self.sy+self.my
                yhat_test_norm,sigmahat_test=self.GPR(self.X_train_norm,self.y_train_norm,self.X_test_norm,r2,s2)
                #yhat_test=yhat_test_norm*self.sy+self.my
                yhat_val_norm,sigmahat_val=self.GPR(self.X_train_norm,self.y_train_norm,self.X_val_norm,r2,s2)
                yhat_val=yhat_val_norm*self.sy+self.my
                #err_train=self.y_train-yhat_train
                #err_test=y_test-yhat_test
                err_val=self.y_val-yhat_val
                MSE_valid=round(np.mean((err_val)**2),3)
                if MSE_valid<self.MSEmin:
                    self.MSEmin=MSE_valid
                    self.r2top=r2
                    self.s2top=s2
                    print(f"{self.s2top} {self.r2top} {MSE_valid} top\n")
            #the fllowing lines can be used to show all MSE(r2) for validation sets
            #plt.figure()
            #plt.plot(r2_values, MSE_valid_matrix[:,1], label=(f"s2={s2}"))
            #plt.grid()
            #plt.legend()
            #plt.xlabel('r2')
            #plt.ylabel('MSE_val')
            #self.v=plt.axis()
            
        self.yhat_train_norm,self.sigmahat_train=self.GPR(self.X_train_norm,self.y_train_norm,self.X_train_norm,self.r2top,self.s2top)
        self.yhat_train=self.yhat_train_norm*self.sy+self.my
        self.yhat_test_norm,self.sigmahat_test=self.GPR(self.X_train_norm,self.y_train_norm,self.X_test_norm,self.r2top,self.s2top)
        self.yhat_test=self.yhat_test_norm*self.sy+self.my
        self.yhat_val_norm,self.sigmahat_val=self.GPR(self.X_train_norm,self.y_train_norm,self.X_val_norm,self.r2top,self.s2top)
        self.yhat_val=self.yhat_val_norm*self.sy+self.my
        self.err_train=self.y_train-self.yhat_train
        self.err_test=self.y_test-self.yhat_test
        self.err_val=self.y_val-self.yhat_val
        
        return self.r2top, self.s2top

    def plotR2S2Optimization(self, r2top, s2top):
        #plots MSE(r2) for validations set, for some values of s2(the first is the best one, choosen to use the GPR method) 
        r2_values=np.zeros(100)
        r2_values[0]=0.1
        for i in range (len(r2_values)-1):
            r2_values[i+1]=r2_values[i]+0.1
        s2_values=[0.0002,0.002,0.005,0.01,s2top]
        s2_values.sort()
        r2_len=len(r2_values)
        s2_len=len(s2_values)
        MSE_valid_matrix=np.zeros((r2_len,s2_len),float)
        for j in range(s2_len):
            print(f"optimization with {s2_values[j]}\n")
            for i in range(r2_len):
                self.yhat_train_norm,self.sigmahat_train=self.GPR(self.X_train_norm,self.y_train_norm,self.X_train_norm,r2_values[i],s2_values[j])
                self.yhat_train=self.yhat_train_norm*self.sy+self.my
                self.yhat_test_norm,self.sigmahat_test=self.GPR(self.X_train_norm,self.y_train_norm,self.X_test_norm,r2_values[i],s2_values[j])
                self.yhat_test=self.yhat_test_norm*self.sy+self.my
                self.yhat_val_norm,self.sigmahat_val=self.GPR(self.X_train_norm,self.y_train_norm,self.X_val_norm,r2_values[i],s2_values[j])
                self.yhat_val=self.yhat_val_norm*self.sy+self.my
                self.err_train=self.y_train-self.yhat_train
                self.err_test=self.y_test-self.yhat_test
                self.err_val=self.y_val-self.yhat_val
                MSE_valid_matrix[i,j]=np.mean((self.err_val)**2)
        plt.figure()
        for j in range(len(s2_values)):
            plt.plot(r2_values, MSE_valid_matrix[:,j], label=(f"s2={s2_values[j]}"))
        plt.grid()
        plt.legend()
        plt.scatter(self.r2top, self.MSEmin, label='MSE_min')
        plt.xlabel('r2')
        plt.ylabel('MSE_val')
        self.v=plt.axis()
        self.yhat_train_norm,self.sigmahat_train=self.GPR(self.X_train_norm,self.y_train_norm,self.X_train_norm,r2top,s2top)
        self.yhat_train=self.yhat_train_norm*self.sy+self.my
        self.yhat_test_norm,self.sigmahat_test=self.GPR(self.X_train_norm,self.y_train_norm,self.X_test_norm,r2top,s2top)
        self.yhat_test=self.yhat_test_norm*self.sy+self.my
        self.yhat_val_norm,self.sigmahat_val=self.GPR(self.X_train_norm,self.y_train_norm,self.X_val_norm,r2top,s2top)
        self.yhat_val=self.yhat_val_norm*self.sy+self.my
        self.err_train=self.y_train-self.yhat_train
        self.err_test=self.y_test-self.yhat_test
        self.err_val=self.y_val-self.yhat_val
              
    def runGPR():
        r=GPRclass()
        r.readFile()
        r.generateData()   
        r.normalization()
        r2top, s2top =r.GPRimplementation()
        r.plotErrorHistogram('Gaussian Process Regression')
        r.plotRegressorLine('Gaussian Process Regression')
        r.plotRegressorLineWithErrors('Gaussian Process Regression')
        r.plotR2S2Optimization(r2top, s2top)
        r.plotErrorStatistics()


class LLSclass(Regression):
    
    def generateData(self): 
        # scatter plots
        todrop=['subject#','test_time']
        self.x1=self.xx.copy(deep=True)
        self.X=self.x1.drop(todrop,axis=1)
        # Generate the shuffled dataframe
        np.random.seed(ID)
        self.Xsh = self.X.sample(frac=1).reset_index(drop=True)
        [self.Np,self.Nc]=self.Xsh.shape
        self.F=self.Nc-1
        # Generate training, validation and testing matrices
        self.Ntr=int(self.Np*0.5)  # number of training points
        self.Nva=int(self.Np*0.25) # number of validation points
        self.Nte=self.Np-self.Ntr-self.Nva   # number of testing points
        self.X_tr=self.Xsh[0:self.Ntr] # training dataset
        # find mean and standard deviations for the features in the training dataset
        self.mm=self.X_tr.mean()
        self.ss=self.X_tr.std()
        self.my=self.mm['total_UPDRS']# get mean for the regressand
        self.sy=self.ss['total_UPDRS']# get std for the regressand
    
    def LLS(self):
        self.w_hat=np.linalg.inv(self.X_train_norm.T@self.X_train_norm)@(self.X_train_norm.T@self.y_train_norm)
        #self.y_hat_test_norm=self.X_test_norm@self.w_hat
        #y_hat_te=sy*h_hat_te_norm+my
        #MSE=np.mean((y_hat_te-y_te)**2)
        #self.MSE_norm=np.mean((self.y_hat_te_norm-self.y_te_norm)**2)
        #self.MSE=self.sy**2*self.MSE_norm
        self.y_test=self.y_test_norm*self.sy+self.my
        self.X_test=self.X_test_norm*self.sy+self.my
        self.y_train=self.y_train_norm*self.sy+self.my
        self.X_train=self.X_train_norm*self.sy+self.my            
        self.err_train=(self.y_train_norm-self.X_train_norm@self.w_hat)*self.sy # training
        self.err_test=(self.y_test_norm-self.X_test_norm@self.w_hat)*self.sy # test           
        self.err_train_real=self.y_train-self.X_train@self.w_hat # training
        self.err_test_real=self.y_test-self.X_test@self.w_hat # test
        self.e=[self.err_train_real,self.err_test_real]
        self.yhat_train=(self.X_train_norm@self.w_hat)*self.sy+self.my # training
        self.yhat_test=(self.X_test_norm@self.w_hat)*self.sy+self.my #test
    
    def runLLS():
        r=LLSclass()
        r.readFile()
        r.generateData()   
        r.normalization()
        r.LLS()
        r.plotErrorHistogram('LLS')
        r.plotRegressorLine('LLS')
        r.plotErrorStatistics()


if __name__=="__main__":
  LLSclass.runLLS()
  GPRclass.runGPR()



