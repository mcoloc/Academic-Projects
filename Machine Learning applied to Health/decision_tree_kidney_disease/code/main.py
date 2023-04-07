import pandas as pd
import numpy as np
from sklearn import tree
import graphviz 
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import json
import random

ID=301227

class KidneyTree:
    
    def __init__(self):
        self.accuracy=0
        self.runs=0
        self.features_importance=np.zeros((1,24),float)
        self.sensitivity=0
        self.specificity=0
        self.min_accuracy=1
        self.max_accuracy=0
        self.min_sensitivity=1
        self.max_sensitivity=0
        self.min_specificity=1
        self.max_specificity=0
        self.accuracy_vect=np.zeros((1002,1),float)
        self.specificity_vect=np.zeros((1002,1),float)
        self.sensitivity_vect=np.zeros((1002,1),float)
        self.used_features=np.zeros((1,24),float)
        self.trees=[]
        
    def dataframe(self):
    # define the feature names:
        self.feat_names=['age','bp','sg','al','su','rbc','pc',
                    'pcc','ba','bgr','bu','sc','sod','pot','hemo',
                    'pcv','wbcc','rbcc','htn','dm','cad','appet','pe',
                    'ane','classk']
        self.ff=np.array(self.feat_names)
        self.feat_cat=np.array(['num','num','cat','cat','cat','cat','cat','cat','cat',
                           'num','num','num','num','num','num','num','num','num',
                           'cat','cat','cat','cat','cat','cat','cat'])
        # import the dataframe:
            #xx=pd.read_csv("./data/chronic_kidney_disease.arff",sep=',',
            #               skiprows=29,names=feat_names, 
            #               header=None,na_values=['?','\t?'],
            #               warn_bad_lines=True)
        self.xx=pd.read_csv("./data/chronic_kidney_disease_v2.arff",sep=',',
                       skiprows=29,names=self.feat_names, 
                       header=None,na_values=['?','\t?'],)
        self.Np,self.Nf=self.xx.shape
        # change categorical data into numbers:
        key_list=["normal","abnormal","present","notpresent","yes",
                  "no","poor","good","ckd","notckd","ckd\t","\tno"," yes","\tyes"]
        key_val=[0,1,0,1,0,1,0,1,1,0,1,1,0,0]
        self.xx=self.xx.replace(key_list,key_val)
        print(self.xx.nunique())# show the cardinality of each feature in the dataset; in particular classk should have only two possible values

    def missingDataRegression(self):
        # manage the missing data through regression
        print(self.xx.info())
        x=self.xx.copy()
        # drop rows with less than 19=Nf-6 recorded features:
        x=x.dropna(thresh=19)

        x.reset_index(drop=True, inplace=True)# necessary to have index without "jumps"
        n=x.isnull().sum(axis=1)# check the number of missing values in each row
        print('max number of missing values in the reduced dataset: ',n.max())
        print('number of points in the reduced dataset: ',len(n))
        # take the rows with exctly Nf=25 useful features; this is going to be the training dataset
        # for regression
        Xtrain=x.dropna(thresh=25)
        Xtrain.reset_index(drop=True, inplace=True)# reset the index of the dataframe
        # get the possible values (i.e. alphabet) for the categorical features
        alphabets=[]
        for k in range(len(self.feat_cat)):
            if self.feat_cat[k]=='cat':
                val=Xtrain.iloc[:,k]
                val=val.unique()
                alphabets.append(val)
            else:
                alphabets.append('num')

        # run regression tree on all the missing data
        #normalize the training dataset
        mm=Xtrain.mean(axis=0)
        ss=Xtrain.std(axis=0)
        Xtrain_norm=(Xtrain-mm)/ss
        # get the data subset that contains missing values 
        Xtest=x.drop(x[x.isnull().sum(axis=1)==0].index)
        Xtest.reset_index(drop=True, inplace=True)# reset the index of the dataframe
        Xtest_norm=(Xtest-mm)/ss # nomralization
        Np,Nf=Xtest_norm.shape
        regr=tree.DecisionTreeRegressor() # instantiate the regressor
        for kk in range(Np):
            xrow=Xtest_norm.iloc[kk]#k-th row
            mask=xrow.isna()# columns with nan in row kk
            Data_tr_norm=Xtrain_norm.loc[:,~mask]# remove the columns from the training dataset
            y_tr_norm=Xtrain_norm.loc[:,mask]# columns to be regressed
            regr=regr.fit(Data_tr_norm,y_tr_norm)
            Data_te_norm=Xtest_norm.loc[kk,~mask].values.reshape(1,-1) # row vector
            ytest_norm=regr.predict(Data_te_norm)
            Xtest_norm.iloc[kk][mask]=ytest_norm # substitute nan with regressed values
            Xtest_new=Xtest_norm*ss+mm # denormalize
            # substitute regressed numerical values with the closest element in the alphabet
        index=np.argwhere(self.feat_cat=='cat').flatten()
        for k in index:
            val=alphabets[k].flatten() # possible values for the feature
            c=Xtest_new.iloc[:,k].values # values in the column
            c=c.reshape(-1,1)# column vector
            val=val.reshape(1,-1) # row vector
            d=(val-c)**2 # matrix with all the distances w.r.t. the alphabet values
            ii=d.argmin(axis=1) # find the index of the closest alphabet value
            Xtest_new.iloc[:,k]=val[0,ii]
        print(Xtest_new.nunique())
        print(Xtest_new.describe().T)
        #
        self.X_new= pd.concat([Xtrain, Xtest_new], ignore_index=True, sort=False)

    def shuffleData(self, seedID):
        #shuffle
        np.random.seed(seedID)
        indexsh=np.arange(len(self.X_new))
        np.random.shuffle(indexsh)
        self.X_new=self.X_new.set_axis(indexsh, axis=0, inplace=False)
        self.X_new=self.X_new.sort_index(axis=0)

    def decisionTree(self, seedID):
        ##------------------ Decision tree -------------------
        ## first decision tree, using Xtrain for training and Xtest_new for test
        target_names = ['notckd','ckd']
        labels = self.X_new[0:158].loc[:,'classk']
        data = self.X_new[0:158].drop('classk', axis=1)
        clfXtrain = tree.DecisionTreeClassifier(criterion='entropy',random_state=4)
        clfXtrain = clfXtrain.fit(data,labels)
        test_pred = clfXtrain.predict(self.X_new[159:349].drop('classk', axis=1))
        print('accuracy =', accuracy_score(self.X_new[159:349].loc[:,'classk'],test_pred))
        if (type(seedID)==str and seedID=='NoShuffle') or seedID==ID:
            print('Confusion matrix')
        print(confusion_matrix(self.X_new[159:349].loc[:,'classk'],test_pred))
        #% export to graghviz to draw a grahp
        # dot_data = tree.export_graphviz(clfXtrain, out_file=None,feature_names=feat_names[:24], class_names=target_names, filled=True, rounded=True, special_characters=True) 
        # graph = graphviz.Source(dot_data) 
        # graph.render("Tree_Xtrain") 

        #black and white option
        if (type(seedID)==str and seedID=='NoShuffle') or seedID==ID:
            tree.plot_tree(clfXtrain)
        #text option
        text_representation = tree.export_text(clfXtrain)
        if (type(seedID)==str and seedID=='NoShuffle') or seedID==ID:
            print(text_representation)
        #option with colors
        if (type(seedID)==str and seedID=='NoShuffle') or seedID==ID: #this 'if' was deleted to give a look to the trees
            plt.title(seedID)
            fig = plt.figure(figsize=(100,70))
            tree.plot_tree(clfXtrain,
                                feature_names=self.feat_names[:24],
                                class_names=target_names,
                                filled=True, rounded=True)
            fig = plt.figure(figsize=(100,70))
            plt.show()
        self.features_importance=self.features_importance+clfXtrain.feature_importances_ #represent the importance of each features using Gimi
        for i in range(24):
            if clfXtrain.feature_importances_[i]>0:
                self.used_features[0][i]=self.used_features[0][i]+1
        
        
        report=(classification_report(self.X_new[159:349].loc[:,'classk'], test_pred))
        specificity=float(report[84:88])
        sensitivity=float(report[138:142])
        accuracy=float(accuracy_score(self.X_new[159:349].loc[:,'classk'],test_pred))
        
        if  str(seedID)!='NoShuffle': #excluding the first tree obtained without regressed values
            self.accuracy_vect[self.runs]=accuracy
            self.specificity_vect[self.runs]=specificity
            self.sensitivity_vect[self.runs]=sensitivity
            self.runs=self.runs+1
            print(self.runs)
            self.min_sensitivity=min(self.min_sensitivity,sensitivity)
            self.max_sensitivity=max(self.max_sensitivity,sensitivity)
            self.min_specificity=min(self.min_specificity,specificity)
            self.max_specificity=max(self.max_specificity,specificity)
            self.sensitivity=self.sensitivity+sensitivity
            self.specificity=self.specificity+specificity
            
            self.min_accuracy=min(self.min_accuracy,accuracy)
            self.max_accuracy=max(self.max_accuracy,accuracy)
            self.accuracy=self.accuracy+accuracy
        
        for entry in self.trees: #saving all trees only to understand if there are any tree equal to another 
            if clfXtrain==entry:
                return
        self.trees.append(clfXtrain)
        
    def randomForest(self): #apply random forest algorithm and print the obtained results
        rfc=ensemble.RandomForestClassifier(n_estimators=1000, criterion='entropy', random_state=4)
        labels = self.X_new[0:158].loc[:,'classk']
        data = self.X_new[0:158].drop('classk', axis=1)
        rfc=rfc.fit(data,labels)
        y_pred=rfc.predict(self.X_new[159:349].drop('classk', axis=1))
        test_labels=self.X_new[159:349].loc[:,'classk']
        print(f"Random Forest Classification:{accuracy_score(test_labels, y_pred)}\n")
        print("Confusion matrix using Random Forest Classification:")
        print(confusion_matrix(self.X_new[159:349].loc[:,'classk'],y_pred))
        print("\n\n")
        report=(classification_report(self.X_new[159:349].loc[:,'classk'], y_pred))
        specificity=float(report[84:88])
        sensitivity=float(report[138:142])
        print('specificity=',specificity)
        print('sensitivity=',sensitivity)

    def printStat(self): #print the results obtained trough the usage of decision trees
        print(kt.features_importance.shape)
        print('features importances=',self.features_importance)
        print('max sensitivity=',round(float(self.max_sensitivity),3))
        print('min sensitivity=',round(float(self.min_sensitivity),3))
        print('max specificity=',round(float(self.max_specificity),3))
        print('min specificity=',round(float(self.min_specificity),3))
        print('max accuracy=',round(float(self.max_accuracy),3))
        print('min accuracy=',round(float(self.min_accuracy),3))
        print('mean specificity=', round(float(self.specificity/self.runs),3))
        print('mean sensitivity=', round(float(self.sensitivity/self.runs),3))
        print('mean accuracy=', round(float(self.accuracy/self.runs),3))
        print('accuracy std=', round(float(np.std(self.accuracy_vect, dtype = np.float32)),3))
        print('sensitivity std=', round(float(np.std(self.sensitivity_vect,dtype = np.float32)),3))
        print('specificity std=', round(float(np.std(self.specificity_vect,dtype = np.float32)),3))
        print('used features=', self.used_features)
        plt.figure()
        plt.title('features importance mean')
        imp=[kt.features_importance[0][0],kt.features_importance[0][1],kt.features_importance[0][2],kt.features_importance[0][3],kt.features_importance[0][4],kt.features_importance[0][5],kt.features_importance[0][6],kt.features_importance[0][7],kt.features_importance[0][8],kt.features_importance[0][9],kt.features_importance[0][10],kt.features_importance[0][11],kt.features_importance[0][12],kt.features_importance[0][13],kt.features_importance[0][14],kt.features_importance[0][15],kt.features_importance[0][16],kt.features_importance[0][17],kt.features_importance[0][18],kt.features_importance[0][19],kt.features_importance[0][20],kt.features_importance[0][21],kt.features_importance[0][22],kt.features_importance[0][23]]
        for i in range(len(imp)):
            imp[i]=imp[i]/self.runs
        plt.bar(range(1,25), imp, width=0.5)
        plt.xticks(np.arange(1,25))
        plt.figure()
        plt.title('features usage')
        feat_count=[self.used_features[0][0],self.used_features[0][1],self.used_features[0][2],self.used_features[0][3],self.used_features[0][4],self.used_features[0][5],self.used_features[0][6],self.used_features[0][7],self.used_features[0][8],self.used_features[0][9],self.used_features[0][10],self.used_features[0][11],self.used_features[0][12],self.used_features[0][13],self.used_features[0][14],self.used_features[0][15],self.used_features[0][16],self.used_features[0][17],self.used_features[0][18],self.used_features[0][19],self.used_features[0][20],self.used_features[0][21],self.used_features[0][22],self.used_features[0][23]]
        plt.bar(range(1,25), feat_count, width=0.5)
        plt.xticks(np.arange(1,25))
        print(len(self.trees), 'different trees were obtained')
        print(self.runs)
        
if __name__=="__main__":
    kt=KidneyTree()
    kt.dataframe()
    kt.missingDataRegression()
    IDmodifier=np.zeros((1000,1),int)
    random.seed(17)
    for i in range(1000):
        IDmodifier[i]=random.randint(-ID,ID)   
    kt.decisionTree('NoShuffle')
    kt.shuffleData(ID)
    kt.decisionTree(ID)
    for i in IDmodifier: 
        seedID=ID+i
        kt.shuffleData(seedID)
        report=kt.decisionTree(seedID)
    kt.printStat()
    kt.randomForest()