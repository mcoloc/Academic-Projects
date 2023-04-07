# -*- coding: utf-8 -*-

FILENAME="parkinsons_updrs_data.csv"
ID=301227
RANGE=500
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Regression():
    
    def readFile(self): 
        #read the dataset and gives the statistical description of the content of each column, used to drop unwanted features
        #and saves useful variables such as the nnumber of patients and the number of features+total_UPDRS
        plt.close('all') 
        self.x=pd.read_csv(FILENAME) 
        self.x.describe().T 
        self.x.info()
        self.features=list(self.x.columns)
        print(self.features)
        #features=['subject#', 'age', 'sex', 'test_time', 'motor_UPDRS', 'total_UPDRS',
        #       'Jitter(%)', 'Jitter(Abs)', 'Jitter:RAP', 'Jitter:PPQ5', 'Jitter:DDP',
        #       'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
        #       'Shimmer:APQ11', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'PPE']
        self.X=self.x.drop(['subject#','test_time'],axis=1)
        self.Np,self.Nc=self.X.shape
        self.features=list(self.X.columns)
    
    def correlation(self):
        #normalize data, compute the covariance matrix of the features, the coorelation coefficint among UPDRS and the other features
        #and plot them 
        self.Xnorm=(self.X-self.X.mean())/self.X.std()
        self.c=self.Xnorm.cov()

        plt.figure()
        plt.matshow(np.abs(self.c.values),fignum=0)
        plt.xticks(np.arange(len(self.features)), self.features, rotation=90)
        plt.yticks(np.arange(len(self.features)), self.features, rotation=0)    
        plt.colorbar()
        plt.title('Covariance matrix of the features')
        plt.tight_layout()
        plt.savefig('./corr_coeff.png') 
        plt.show()

        plt.figure()
        self.c.motor_UPDRS.plot()
        plt.grid()
        plt.xticks(np.arange(len(self.features)), self.features, rotation=90)
        plt.title('corr. coeff. among motor UPDRS and the other features')
        plt.tight_layout()
        plt.show()
        
        plt.figure()
        self.c.total_UPDRS.plot()
        plt.grid()
        plt.xticks(np.arange(len(self.features)), self.features, rotation=90) 
        plt.title('corr. coeff. among total UPDRS and the other features')
        plt.tight_layout()
        plt.show()        
          
    def meanAndSd(self):    
        # saves the dataframe that contains only the training data and compute means and standard deviations
        self.X_tr=self.Xsh[0:self.Ntr]
        self.mm=self.X_tr.mean()# mean (series)
        self.ss=self.X_tr.std()# standard deviation (series)
        self.my=self.mm['total_UPDRS']# mean of motor UPDRS
        self.sy=self.ss['total_UPDRS']# st.dev of motor UPDRS    
        
    def normalization(self):  
        #normalize data and divide into training set and test set
        self.Xsh_norm=(self.Xsh-self.mm)/self.ss
        self.ysh_norm=self.Xsh_norm['total_UPDRS']
        self.Xsh_norm=self.Xsh_norm.drop('total_UPDRS',axis=1)
        self.X_tr_norm=self.Xsh_norm[0:self.Ntr]
        self.X_te_norm=self.Xsh_norm[self.Ntr:]
        self.y_tr_norm=self.ysh_norm[0:self.Ntr]
        self.y_te_norm=self.ysh_norm[self.Ntr:]   
   
    def plotOptimumWeightVector(self, w_hat, title):
        #compute and plot the Optimized weights
            regressors=list(self.X_tr_norm.columns)
            Nf=len(w_hat)
            nn=np.arange(Nf)
            plt.figure(figsize=(6,4))
            plt.plot(nn,w_hat,'-o')
            ticks=nn
            plt.xticks(ticks, regressors, rotation=90)#, **kwargs)
            plt.ylabel(r'$\^w(n)$')
            plt.title(f"{title}-Optimized weights")
            plt. ylim ([-10, 10])
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"./ {title}-what.png")
            plt.show()
                 
    def plotErrorHistogram(self, w_hat, title):
        #compute and plot errors between y_hat and y
            self.y_te=self.y_te_norm*self.sy+self.my
            self.X_te=self.X_te_norm*self.sy+self.my
            self.y_tr=self.y_tr_norm*self.sy+self.my
            self.X_tr=self.X_tr_norm*self.sy+self.my            
            self.E_tr=(self.y_tr_norm-self.X_tr_norm@w_hat)*self.sy # training
            self.E_te=(self.y_te_norm-self.X_te_norm@w_hat)*self.sy # test           
            self.E_tr_real=self.y_tr-self.X_tr@w_hat # training
            self.E_te_real=self.y_te-self.X_te@w_hat # test
            self.e=[self.E_tr_real,self.E_te_real]
            
            plt.figure(figsize=(6,4))
            plt.hist(self.e,bins=50,density=True, histtype='bar',
            label=['training','test'])
            plt.xlabel(r'$e=y-\^y$')
            plt.ylabel(r'$P(e$ in bin$)$')
            plt.legend()
            plt.grid()
            plt.title(f"{title}-Error histograms")
            plt.tight_layout()
            plt.savefig(f"./{title}-hist.png")
            plt.show()

    def plotRegressorLine(self, w_hat, title):
        #plot y_hat(y) showing graphically the distances between the correct values 
        self.y_hat_te=(self.X_te_norm@w_hat)*self.sy+self.my
        self.y_te=self.y_te_norm*self.sy+self.my
        plt.figure(figsize=(6,4))
        plt.plot(self.y_te,self.y_hat_te,'.')
        v=plt.axis()
        plt.plot([v[0],v[1]],[v[0],v[1]],'r',linewidth=2)
        plt.xlabel(r'$y$')
        plt.ylabel(r'$\^y$')
        plt.grid()
        plt.title(f"{title}-test")
        plt.tight_layout()
        plt.savefig(f"./{title}-yhat_vs_y.png")
        plt.show()
                
    def plotErrorStatistics(self, w_hat, title):
        #compute and print statistics about mean, standard deviation, MSE and R^2
        E_tr_mu=self.E_tr.mean()
        E_tr_sig=self.E_tr.std()
        E_tr_MSE=np.mean(self.E_tr**2)
        self.y_tr=self.y_tr_norm*self.sy+self.my
        R2_tr=1-E_tr_sig**2/np.mean(self.y_tr**2)
        E_te_mu=self.E_te.mean()
        E_te_sig=self.E_te.std()
        E_te_MSE=np.mean(self.E_te**2)
        self.y_te=self.y_te_norm*self.sy+self.my
        R2_te=1-E_te_sig**2/np.mean(self.y_te**2)
        rows=['Training','test']
        cols=['mean','std','MSE','R^2']
        p=np.array([[E_tr_mu,E_tr_sig,E_tr_MSE,R2_tr],[E_te_mu,E_te_sig,E_te_MSE,R2_te]])
        results=pd.DataFrame(p,columns=cols,index=rows)
        print(results)
        
        
class Minibatches(Regression):
    
    def generateData(self):
        #generate the radon shufflesd set and compute the numer of training set elements ad the number of test elements (75% - 25%)
        np.random.seed(ID) # set the seed for random shuffling
        indexsh=np.arange(self.Np)
        np.random.shuffle(indexsh)
        self.Xsh=self.X.copy(deep=True)
        self.Xsh=self.Xsh.set_axis(indexsh,axis=0,inplace=False)
        self.Xsh=self.Xsh.sort_index(axis=0)
        self.Ntr=int(((self.Np*0.75/64)//1)*64)  # number of training points set as a multiple of 64 to correctly use minibatches algorithm implementation
        self.Nte=self.Np-self.Ntr   # number of test points       
        
    def minibatches(self):
        self.gamma=1e-5 #trying with different values for gamma, 1e-5 gives the best results in an accetable time
        self.num_batches=2 #trying with different sizes and analize the results (4-8-16-32-64)
        self.X_tr_norm_matrix=self.X_tr_norm.values
        self.y_tr_norm_vector=self.y_tr_norm.values
        self.w_hat_results=np.zeros((5,self.Nc-1))
        self.y_te=self.y_te_norm*self.sy+self.my
        self.X_te=self.X_te_norm*self.sy+self.my
        self.y_tr=self.y_tr_norm*self.sy+self.my
        self.X_tr=self.X_tr_norm*self.sy+self.my

        for i in range(0,5):
            self.w_hat=np.random.rand(self.Nc-1)
            self.num_batches=self.num_batches*2
            self.size=int(self.Ntr/self.num_batches)
            error_total=np.random.rand(RANGE)
            for it in range(RANGE):
                self.w_hat_old=self.w_hat
                for j in range (self.num_batches):
                    grad=2*self.X_tr_norm_matrix[j*self.size:(j+1)*self.size-1, :].T@(self.X_tr_norm_matrix[j*self.size:(j+1)*self.size-1, :]@self.w_hat-self.y_tr_norm_vector[j*self.size:(j+1)*self.size-1])
                    self.w_hat=self.w_hat-self.gamma*grad                
                self.error=np.abs(self.y_te-self.X_te@self.w_hat)                   
                self.err=np.mean(self.error)
                error_total[it]=self.err
                #stop condition: find the minimum value of error
                if it>1 and error_total[it]>error_total[it-1] and error_total[it-1]<error_total[it-2]:
                    print(f"number of iterations:{it}, size={self.size}, number of bacthes= {self.num_batches}\n")
                    break                
            self.w_hat_results[i,:]=self.w_hat_old 
            #plot used to analize the trend of the error (use it deleting the break before)
            #plt.figure(figsize=(6,4))
            #plt.plot(range(RANGE),error_total)
            #plt.grid()
            #plt.title(f"MBerror size={self.size}")
            #plt.show()
            
        return self.w_hat_results
            
    def run():
        r=Minibatches()
        r.readFile()
        r.correlation()
        r.generateData()   
        r.meanAndSd()
        r.normalization()
        w_hat_MB=r.minibatches()
        num=2
        for i in range(5):
            num=num*2
            r.plotOptimumWeightVector(w_hat_MB[i,:], f"GAM - {num} minibatches")
            r.plotErrorHistogram(w_hat_MB[i,:], f"GAM - {num} minibatches")
            r.plotRegressorLine(w_hat_MB[i,:], f"GAM - {num} minibatches")
            r.plotErrorStatistics(w_hat_MB[i,:], f"GAM - {num} minibatches")
           
           
class Adam(Regression):    
    
    def generateData(self):
        #generate the radon shufflesd set and compute the numer of training set elements ad the number of test elements (75% - 25%)
        np.random.seed(ID) # set the seed for random shuffling
        indexsh=np.arange(self.Np)
        np.random.shuffle(indexsh)
        self.Xsh=self.X.copy(deep=True)
        self.Xsh=self.Xsh.set_axis(indexsh,axis=0,inplace=False)
        self.Xsh=self.Xsh.sort_index(axis=0)
        self.Ntr=int(self.Np*0.75)  # number of training points
        self.Nte=self.Np-self.Ntr   # number of test points        
    
    def adam(self):
        self.w_hat=np.random.rand(self.Nc-1)
        self.b1=0.9
        self.b2=0.999
        self.gamma=1e-5 #trying with gamma = 1e-2 1e-3 1e-4 it takes less time but giving worst results
        self.eps=1e-9
        self.X_tr_norm_matrix=self.X_tr_norm.values
        self.m=self.ms=0
        
        self.old_error=np.zeros(self.Nte)
        self.y_te=self.y_te_norm*self.sy+self.my
        self.X_te=self.X_te_norm*self.sy+self.my
        self.y_tr=self.y_tr_norm*self.sy+self.my
        self.X_tr=self.X_tr_norm*self.sy+self.my
        error_total=np.random.rand(RANGE)
        mse_te=np.zeros((self.Ntr*340, 2))
        mse_tr=np.zeros((self.Ntr*340, 2))
       
        for it in range(RANGE):
            self.w_hat_old=self.w_hat
            for i in range(self.Ntr):   
                self.grad=2*self.X_tr_norm.values[i]*(self.X_tr_norm.values[i].T@self.w_hat-self.y_tr_norm[i])
                self.m = self.m*self.b1 + (1-self.b1)*self.grad
                self.ms = self.ms*self.b2 + (1-self.b2)*(self.grad*self.grad)         
                self.m_corr = self.m/(1-self.b1**(i+1))
                self.ms_corr = self.ms/(1-self.b2**(i+1))
                self.w_hat=self.w_hat-self.gamma*(self.m_corr/(self.ms_corr+10**(-8))**0.5)
                #self.y_hat_te_norm=(self.X_te_norm@self.w_hat)
                #self.y_hat_tr_norm=(self.X_tr_norm@self.w_hat)
                #mse_tr[it*self.Ntr+i,0]=it*self.Ntr+i
                #mse_te[it*self.Ntr+i,0]=it*self.Ntr+i
                #mse_tr[it*self.Ntr+i,1]=np.mean((self.y_hat_tr_norm-self.y_tr_norm)**2)*self.sy**2
                #mse_te[it*self.Ntr+i,1]=np.mean((self.y_hat_te_norm-self.y_te_norm)**2)*self.sy**2
                #print(f"{it} \n {np.mean((self.y_hat_tr_norm-self.y_tr_norm)**2)*self.sy**2}  {np.mean((self.y_hat_te_norm-self.y_te_norm)**2)*self.sy**2} \ntrain={mse_tr[it*self.Ntr+i,1]} test={mse_te[it*self.Ntr+i,1]}\n")
                #commented lines were used to obtain the MSE plot 3.a
            error=np.abs(self.y_te-self.X_te@self.w_hat)                   
            err=np.mean(error)
            error_total[it]=err
            #stop condition: find the minimum value of error
            if it>1 and error_total[it]>error_total[it-1] and error_total[it-1]<error_total[it-2]:
                print(f"number of iterations:{it}\n")
                #plt.figure()        
                #plt.semilogy(mse_tr[:,0],mse_tr[:,1], label= 'training', color='b')
                #plt.semilogy(mse_te[:,0],mse_te[:,1], label= 'test', color='r') 
                #plt.legend()
                #plt.xlabel('iterations')
                #plt.ylabel('MSE(iterations)')
                #plt.title("Mean Squared Error")
                #plt.margins(0.01,0.1)       
                #plt.grid()
                #plt.show()
                return self.w_hat_old
               
        return self.w_hat

    def run():
        r=Adam()
        r.readFile()
        r.correlation()
        r.generateData()   
        r.meanAndSd()
        r.normalization()
        w_hat=r.adam()
        r.plotOptimumWeightVector(w_hat, 'SG')
        r.plotErrorHistogram(w_hat, 'SG')
        r.plotRegressorLine(w_hat, 'SG')
        r.plotErrorStatistics(w_hat, 'SG')
        
        
class Lls(Regression):    
    
    def generateData(self):
        np.random.seed(ID) # set the seed for random shuffling
        indexsh=np.arange(self.Np)
        np.random.shuffle(indexsh)
        self.Xsh=self.X.copy(deep=True)
        self.Xsh=self.Xsh.set_axis(indexsh,axis=0,inplace=False)
        self.Xsh=self.Xsh.sort_index(axis=0)
        #%% Generate training, validation and test matrices
        self.Ntr=int(self.Np*0.75)  # number of training points
        self.Nte=self.Np-self.Ntr   # number of test points       
                                
    def lls(self):
        self.w_hat=np.linalg.inv(self.X_tr_norm.T@self.X_tr_norm)@(self.X_tr_norm.T@self.y_tr_norm)
        #self.y_hat_te_norm=self.X_te_norm@self.w_hat
        #y_hat_te=sy*h_hat_te_norm+my
        #MSE=np.mean((y_hat_te-y_te)**2)
        #self.MSE_norm=np.mean((self.y_hat_te_norm-self.y_te_norm)**2)
        #self.MSE=self.sy**2*self.MSE_norm
        return self.w_hat

    def plotErrorHistogram(self, w_hat, title):
        #plot y_hat(y) showing graphically the distances between the correct values 
        self.E_tr=(self.y_tr_norm-self.X_tr_norm@self.w_hat)*self.sy# training
        self.E_te=(self.y_te_norm-self.X_te_norm@self.w_hat)*self.sy# test
        self.e=[self.E_tr,self.E_te]
        plt.figure(figsize=(6,4))
        plt.hist(self.e,bins=50,density=True, histtype='bar',
        label=['training','test'])
        plt.xlabel(r'$e=y-\^y$')
        plt.ylabel(r'$P(e$ in bin$)$')
        plt.legend()
        plt.grid()
        plt.title('LLS-Error histograms')
        plt.tight_layout()
        plt.savefig('./LLS-hist.png')
        plt.show()
        
    def run():
        r=Lls()
        r.readFile()
        r.correlation()
        r.generateData()   
        r.meanAndSd()
        r.normalization()
        w_hat=r.lls()
        r.plotOptimumWeightVector(w_hat, 'LLS')
        r.plotErrorHistogram(w_hat, 'LLS')
        r.plotRegressorLine(w_hat, 'LLS')
        r.plotErrorStatistics(w_hat, 'LLS')

if __name__=="__main__":
   Lls.run()
   Minibatches.run()
   Adam.run()