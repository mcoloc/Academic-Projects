import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
FILE="covid_serological_results.csv"

class Covid19Roc:
    
    def __init__(self, thresh):
        self.chosen_thresh=thresh
    
    def readFile(self, file): #read the file andcreate the swab vector, counting positive and negative tests
        plt.close('all')
        xx=pd.read_csv(FILE)
        swab=xx.COVID_swab_res.values# results from swab: 0= no illness, 1 = unclear, 2=illness
        Test1=xx.IgG_Test1_titre.values
        Test2=xx.IgG_Test2_titre.values
        ii=np.argwhere(swab==1).flatten()
        swab=np.delete(swab,ii)
        swab=swab//2
        positive=0
        negative=0
        for i in range(len(swab)):
            if swab[i]==0:
                positive=positive+1
            else:
                negative=negative+1
        print(f"Swab test: {positive} positive and negative {negative}\n")
        Test1=np.delete(Test1,ii)
        Test2=np.delete(Test2,ii)
        return Test1, Test2, swab
    
    
    def probPlot(self, x,y, title): # compute and plot the required probabilities relating H/D and tests results n/p
    
        if x.min()>0:# add a couple of zeros, in order to have the zero threshold
            x=np.insert(x,0,0)# add a zero as the first element of xs
            y=np.insert(y,0,0)# also add a zero in y
    
        ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
        ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
        x0=x[ii0]# test values for healthy patients
        x1=x[ii1]# test values for ill patients
        xs=np.sort(x)# sort test values: they represent all the possible  thresholds
        Np=ii1.size# number of positive cases
        Nn=ii0.size# number of negative cases
        pD=0.02
        pH=0.98
        pD_given_Tn=np.zeros((len(xs),1),float)
        pD_given_Tp=np.zeros((len(xs),1),float)
        pH_given_Tn=np.zeros((len(xs),1),float)
        pH_given_Tp=np.zeros((len(xs),1),float)
        i=0 
    
        for thresh in xs:
            num_Tp_given_D=np.sum(x1>thresh)
            num_Tp_given_H=np.sum(x0>thresh)
            num_Tn_given_D=np.sum(x1<thresh)
            num_Tn_given_H=np.sum(x0<thresh)
        
            pH_given_Tn[i]=(pH*num_Tn_given_H/Nn)/(pD*num_Tn_given_D/Np+pH*num_Tn_given_H/Nn)
            pD_given_Tp[i]=(pD*num_Tp_given_D/Np)/(pD*num_Tp_given_D/Np+pH*num_Tp_given_H/Nn+1e-15)
            pD_given_Tn[i]=1-pH_given_Tn[i]
            pH_given_Tp[i]=1-pD_given_Tp[i]
            
            if thresh==self.chosen_thresh:
                print(f'pH_given_Tn={100*pH_given_Tn[i]}, pD_given_Tp={100*pD_given_Tp[i]}, pD_given_Tn={100*pD_given_Tn[i]}, pH_given_Tp={100*pH_given_Tp[i]}\n') 
            
            i=i+1
            
        plt.figure()
        plt.plot(xs,pD_given_Tp,label='P(D|Tp)')
        plt.plot(xs,pD_given_Tn,label='P(D|Tn)')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'P(D) - {title}')
        plt.grid()
    
        plt.figure()
        plt.plot(xs,pH_given_Tp,label='P(H|Tp)')
        plt.plot(xs,pH_given_Tn,label='P(H|Tn)')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(f'P(H) - {title}')
        plt.grid()
        

    def findROC(self, x,y):# 
        """ findROC(x,y) generates data to plot the ROC curve.
        x and y are two 1D vectors each with length N
        x[k] is the scalar value measured in the test
        y[k] is either 0 (healthy person) or 1 (ill person)
        The output data is a 2D array N rows and three columns
        data[:,0] is the set of thresholds
        data[:,1] is the corresponding false alarm
        data[:,2] is the corresponding sensitivity"""
    
        if x.min()>0:# add a couple of zeros, in order to have the zero threshold
            x=np.insert(x,0,0)# add a zero as the first element of xs
            y=np.insert(y,0,0)# also add a zero in y
    
        ii0=np.argwhere(y==0).flatten()# indexes where y=0, healthy patient
        ii1=np.argwhere(y==1).flatten()# indexes where y=1, ill patient
        x0=x[ii0]# test values for healthy patients
        x1=x[ii1]# test values for ill patients
        xs=np.sort(x)# sort test values: they represent all the possible  thresholds
        # if x> thresh -> test is positive
        # if x <= thresh -> test is negative
        # number of cases for which x0> thresh represent false positives
        # number of cases for which x0<= thresh represent true negatives
        # number of cases for which x1> thresh represent true positives
        # number of cases for which x1<= thresh represent false negatives
        # sensitivity = P(x>thresh|the patient is ill)=
        #             = P(x>thresh, the patient is ill)/P(the patient is ill)
        #             = number of positives in x1/number of positives in y
        # false alarm = P(x>thresh|the patient is healthy)
        #             = number of positives in x0/number of negatives in y
        Np=ii1.size# number of positive cases
        Nn=ii0.size# number of negative cases
        data=np.zeros((Np+Nn,3),dtype=float)
        i=0
        ROCarea=0
        for thresh in xs:
            n1=np.sum(x1>thresh)#true positives
            sens=n1/Np
            n2=np.sum(x0>thresh)#false positives
            falsealarm=n2/Nn
            data[i,0]=thresh
            data[i,1]=falsealarm
            data[i,2]=sens
            if round(1-falsealarm,3)==round(sens,3):
                print(f"\n\nsens=1-false alarm for threshold={thresh} with value {sens}\n")
            if i>0:
                ROCarea=ROCarea+sens*(data[i-1,1]-data[i,1])
            i=i+1
            #if thresh==self.chosen_thresh:
            #    print(f'sens={sens},thresh={thresh},fals alarm={1-falsealarm}\n')
        return data,ROCarea

    def plotCDF(self, test, swab, title): #plot the CDF
        ii0=np.argwhere(swab==0)
        ii1=np.argwhere(swab==1)
        plt.figure()
        plt.hist(test[ii0],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
        plt.hist(test[ii1],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
        plt.grid()
        plt.legend()
        plt.title(title)

    def plotTestValues(self, test, title): #plot the test values to better understand the overall behavoiur
        self.tlen=len(test)
        #plot test1 values
        plt.figure()
        plt.plot(range(self.tlen), test)
        plt.title(title)
        
    def deleteOutliers(self, test, swab, title): #chose the paramters and delete the outliers
        Test1fit=np.zeros((self.tlen,1),float)
        for i in range (self.tlen):
            Test1fit[i]=test[i]
            #choosing best parameters
        min_samples=2
        opt_eps=8
        #deleting outliers        
        outliers=DBSCAN(eps=opt_eps, min_samples=min_samples).fit(Test1fit)
        outliers_indexes=outliers.labels_
        count=0
        for i in range(self.tlen):
            if outliers_indexes[i]==-1:
                count=count+1
        Test_without_outliers=np.zeros((self.tlen-count,1),float)    
        swab_without_outliers=np.zeros((self.tlen-count,1),float)    
        positive_outliers=0
        negative_outliers=0                        
        j=0
        for i in range(self.tlen):
            if outliers_indexes[i]!=-1:
                Test_without_outliers[j]=test[i]
                swab_without_outliers[j]=swab[i]
                j=j+1
            else:
                if swab[i]==0:
                    negative_outliers=negative_outliers+1
                else:
                    positive_outliers=positive_outliers+1
        print(f"\n{positive_outliers} positive outliers deleted, {negative_outliers} negative outliers deleted\n")
        plt.figure()
        Test1_wo_len=len(Test_without_outliers)

        ii0_2=np.argwhere(swab_without_outliers[:,0]==0)
        ii1_2=np.argwhere(swab_without_outliers[:,0]==1)
        plt.figure()
        plt.hist(Test_without_outliers[ii0_2,0],bins=100,density=True,label=r'$f_{r|H}(r|H)$')
        plt.hist(Test_without_outliers[ii1_2,0],bins=100,density=True,label=r'$f_{r|D}(r|D)$')
        plt.grid()
        plt.legend()
        plt.title(title)

        plt.figure()
        plt.plot(range(Test1_wo_len), Test_without_outliers)
        plt.title(title)
        
        return Test_without_outliers, swab_without_outliers

    def plotROCSensSpec(self, test,swab, title): #plot the sensitivity and the specificity
        data_Test, area=self.findROC(test,swab)
        plt.figure()
        plt.plot(data_Test[:,1],data_Test[:,2],'-',label=title)
        plt.xlabel('FA')
        plt.ylabel('Sens')
        plt.grid()
        plt.legend()
        plt.title(f'ROC - {title}')
        plt.figure()
        plt.plot(data_Test[:,0],data_Test[:,1],'.',label='False alarm')
        plt.plot(data_Test[:,0],data_Test[:,2],'.',label='Sensitivity')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(title)
        plt.grid()

        plt.figure()
        plt.plot(data_Test[:,0],1-data_Test[:,1],'-',label='Specificity')
        plt.plot(data_Test[:,0],data_Test[:,2],'-',label='Sensitivity')
        plt.legend()
        plt.xlabel('threshold')
        plt.title(title)
        plt.grid()

        print(f"ROC Area {title} {round(area,3)}\n")


if __name__=='__main__':
    thresh1=7.59
    thresh2=0.3
    o1=Covid19Roc(thresh1)
    o2=Covid19Roc(thresh2)
    test1, test2, swab= o1.readFile(FILE)
    
    print('test 1')
    o1.plotCDF(test1, swab, 'Test 1')
    o1.plotTestValues(test1, 'Test 1 values')
    test1_wo, swab1_wo=o1.deleteOutliers(test1,swab,'Test 1 without outliers')
    o1.plotROCSensSpec(test1_wo, swab1_wo, 'Test 1')
    o1.probPlot(test1_wo, swab1_wo, 'Test 1')

    print('test 2')
    o2.plotCDF(test2, swab,'Test 2')
    o2.plotTestValues(test2, 'Test 2 values')
    o2.plotROCSensSpec(test2, swab, 'Test 2')
    o2.probPlot(test2, swab, 'Test 2')

