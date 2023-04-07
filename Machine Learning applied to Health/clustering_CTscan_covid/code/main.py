# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 8:54:25 2021

@author: Marco Colocrese
"""

import numpy   as np
import nibabel as nib # to read NII files
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from itertools import product

class CTscan:

    def read_nii(self, filepath):
        '''
        Reads .nii file and returns pixel array
        '''
        ct_scan = nib.load(filepath)
        array   = ct_scan.get_fdata()
        array   = np.rot90(np.array(array))
        return(array)

    def plotSample(self, array_list, color_map = 'nipy_spectral'):
        '''
        Plots a slice with all available annotations
        '''
        plt.figure(figsize=(18,15))
        
        plt.subplot(1,4,1)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.title('Original Image')

        plt.subplot(1,4,2)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
        plt.title('Lung Mask')

        plt.subplot(1,4,3)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[2], alpha=0.5, cmap=color_map)
        plt.title('Infection Mask')

        plt.subplot(1,4,4)
        plt.imshow(array_list[0], cmap='bone',interpolation="nearest")
        plt.imshow(array_list[3], alpha=0.5, cmap=color_map)
        plt.title('Lung and Infection Mask')

        plt.show()
    

    def filterImage(self,D,NN):
        """D = image (matrix) to be filtered, Nr rows, N columns, scalar values (no RGB color image)
        The image is filtered using a square kernel/impulse response with side 2*NN+1"""
        E=D.copy()
        E[np.isnan(E)]=0
        Df=E*0
        Nr,Nc=D.shape
        rang=np.arange(-NN,NN+1)
        square=np.array([x for x in product(rang, rang)])
        #square=np.array([[1,1],[1,0],[1,-1],[0,1],[0,0],[0,-1],[-1,1],[-1,0],[-1,-1]])
        for kr in range(NN,Nr-NN):
            for kc in range(NN,Nc-NN):
                ir=kr+square[:,0]
                ic=kc+square[:,1]
                Df[kr,kc]=np.sum(E[ir,ic])# Df will have higher values where ones are close to each other in D
        return Df/square.size
    
    def useDBSCAN(self,D,z,epsv,min_samplesv):
        """D is the image to process, z is the list of image coordinates to be
        clustered"""
        Nr,Nc=D.shape
        clusters =DBSCAN(eps=epsv,min_samples=min_samplesv,metric='euclidean').fit(z)
        a,Npoints_per_cluster=np.unique(clusters.labels_,return_counts=True)
        Nclust_DBSCAN=len(a)-1
        Npoints_per_cluster=Npoints_per_cluster[1:]# remove numb. of outliers (-1)
        ii=np.argsort(-Npoints_per_cluster)# from the most to the less populated clusters
        Npoints_per_cluster=Npoints_per_cluster[ii]
        C=np.zeros((Nr,Nc,Nclust_DBSCAN))*np.nan # one image for each cluster
        info=np.zeros((Nclust_DBSCAN,5),dtype=float)
        for k in range(Nclust_DBSCAN):
            i1=ii[k] 
            index=(clusters.labels_==i1)
            jj=z[index,:] # image coordinates of cluster k
            C[jj[:,0],jj[:,1],k]=1 # Ndarray with third coord k stores cluster k
            a=np.mean(jj,axis=0).tolist()
            b=np.var(jj,axis=0).tolist()
            info[k,0:2]=a #  store coordinates of the centroid
            info[k,2:4]=b # store variance
            info[k,4]=Npoints_per_cluster[k] # store points in cluster
        return C,info,clusters

    def readFile(self):
        # Read sample
        plt.close('all')    
        self.plotFlag=True

        fold1='./data/ct_scans'
        fold2='./data/lung_mask'
        fold3='./data/infection_mask'
        fold4='./data/lung_and_infection_mask'
        f1='/coronacases_org_001.nii'
        f2='/coronacases_001.nii'
        self.sample_ct   = self.read_nii(fold1+f1+f1)
        self.sample_lung = self.read_nii(fold2+f2+f2)
        self.sample_infe = self.read_nii(fold3+f2+f2)
        self.sample_all  = self.read_nii(fold4+f2+f2)

        self.Nr,self.Nc,self.Nimages=self.sample_ct.shape# Nr=512,Nc=512,Nimages=301

    def plotHistogram(self):
        index=132
        self.sct=self.sample_ct[...,index]
        self.sl=self.sample_lung[...,index]
        self.si=self.sample_infe[...,index]
        self.sa=self.sample_all[...,index]
        self.plotSample([self.sct,self.sl,self.si,self.sa])

        a=np.histogram(self.sct,200,density=True)
        if self.plotFlag:
            plt.figure()
            plt.plot(a[1][0:200],a[0])
            plt.title('Histogram of CT values in slice '+str(index))
            plt.grid()
            plt.xlabel('value')

    def colorQuantizationAndPlot(self):
        Ncluster=5
        self.kmeans = KMeans(n_clusters=Ncluster,random_state=0)# instantiate Kmeans
        self.A=self.sct.reshape(-1,1)# Ndarray, Nr*Nc rows, 1 column
        self.kmeans.fit(self.A)# run Kmeans on A
        self.kmeans_centroids=self.kmeans.cluster_centers_.flatten()#  centroids/quantized colors
        for k in range(Ncluster):
            ind=(self.kmeans.labels_==k)# indexes for which the label is equal to k
            self.A[ind]=self.kmeans_centroids[k]# set the quantized color
        sctq=self.A.reshape(self.Nr,self.Nc)# quantized image
        self.vm=self.sct.min()
        self.vM=self.sct.max()

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(self.sct, cmap='bone',interpolation="nearest")
        ax1.set_title('Original image')
        ax2.imshow(sctq,vmin=self.vm,vmax=self.vM, cmap='bone',interpolation="nearest")
        ax2.set_title('Quantized image')

        ifind=1# second darkest color
        self.ii=self.kmeans_centroids.argsort()# sort centroids from lowest to highest
        ind_clust=self.ii[ifind]# get the index of the desired cluster 
        ind=(self.kmeans.labels_==ind_clust)# get the indexes of the pixels having the desired color
        self.D=self.A*np.nan
        self.D[ind]=1# set the corresponding values of D  to 1
        self.D=self.D.reshape(self.Nr,self.Nc)# make D an image/matrix through reshaping
        plt.figure()
        plt.imshow(self.D,interpolation="nearest")
        plt.title('Image used to identify lungs')

    def findLungs(self):
        eps=2
        min_samples=5
        C,self.centroids,clust=self.useDBSCAN(self.D,np.argwhere(self.D==1),eps,min_samples)
        # we want left lung first. If the images are already ordered
        # then the center along the y-axis (horizontal axis) of C[:,:,0] is smaller
        if self.centroids[1,1]<self.centroids[0,1]:# swap the two subimages
            print('swap')
            tmp = C[:,:,0]*1
            C[:,:,0] = C[:,:,1]*1
            C[:,:,1] = tmp
            tmp=self.centroids[0,:]*1
            self.centroids[0,:]=self.centroids[1,:]*1
            self.centroids[1,:]=tmp
        LLung = C[:,:,0].copy()  # left lung
        RLung = C[:,:,1].copy()  # right lung
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(LLung,interpolation="nearest")
        ax1.set_title('Left lung mask - initial')
        ax2.imshow(RLung,interpolation="nearest")
        ax2.set_title('Right lung mask - initial')

    def darkestColorsImage(self):
        D=self.A*np.nan
        ii=self.kmeans_centroids.argsort()# sort centroids from lowest to highest
        ind=(self.kmeans.labels_==ii[0])# get the indexes of the pixels with the darkest color
        D[ind]=1# set the corresponding values of D  to 1
        ind=(self.kmeans.labels_==ii[1])# get the indexes of the pixels with the 2nd darkest  color
        D[ind]=1# set the corresponding values of D  to 1
        D=D.reshape(self.Nr,self.Nc)# make D an image/matrix through reshaping
        
        C,centers2,clust=self.useDBSCAN(D,np.argwhere(D==1),2,5)
        ind=np.argwhere(centers2[:,4]<1000) # remove small clusters
        centers2=np.delete(centers2,ind,axis=0)
        distL=np.sum((self.centroids[0,0:2]-centers2[:,0:2])**2,axis=1)    
        distR=np.sum((self.centroids[1,0:2]-centers2[:,0:2])**2,axis=1)    
        iL=distL.argmin()
        iR=distR.argmin() 
        self.LLungMask=C[:,:,iL].copy()
        self.RLungMask=C[:,:,iR].copy()
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(self.LLungMask,interpolation="nearest")
        ax1.set_title('Left lung mask - improvement')
        ax2.imshow(self.RLungMask,interpolation="nearest")
        ax2.set_title('Right lung mask - improvement')

    def findLungMask(self):

        C,centers3,clust=self.useDBSCAN(self.LLungMask,np.argwhere(np.isnan(self.LLungMask)),1,5)
        self.LLungMask=np.ones((self.Nr,self.Nc))
        self.LLungMask[C[:,:,0]==1]=np.nan
        C,centers3,clust=self.useDBSCAN(self.RLungMask,np.argwhere(np.isnan(self.RLungMask)),1,5)
        self.RLungMask=np.ones((self.Nr,self.Nc))
        self.RLungMask[C[:,:,0]==1]=np.nan

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(self.LLungMask,interpolation="nearest")
        ax1.set_title('Left lung mask')
        ax2.imshow(self.RLungMask,interpolation="nearest")
        ax2.set_title('Right lung mask')
        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(self.LLungMask*self.sct,vmin=self.vm,vmax=self.vM, cmap='bone',interpolation="nearest")
        ax1.set_title('Left lung')
        ax2.imshow(self.RLungMask*self.sct,vmin=self.vm,vmax=self.vM, cmap='bone',interpolation="nearest")
        ax2.set_title('Right lung')

        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.imshow(self.LLungMask*self.sct,interpolation="nearest")
        ax1.set_title('Left lung')
        ax2.imshow(self.RLungMask*self.sct,interpolation="nearest")
        ax2.set_title('Right lung')

    def groundGrassOpacities(self):#computes the percentage of infected area and the number and the dimension of clusters of infected pixels
                                   #it plots infection masks and the original imamge with the ground glass opacities on it
        
        self.LLungMask[np.isnan(self.LLungMask)]=0
        self.RLungMask[np.isnan(self.RLungMask)]=0
        self.LungsMask=self.LLungMask+self.RLungMask
        Bmin=-650
        Bmax=-300
        thresh=0.25
        B=self.LungsMask*self.sct
        inf_mask=1*(B>Bmin)&(B<Bmax)
        InfectionMask=self.filterImage(inf_mask,1)
        InfectionMask=1.0*(InfectionMask>thresh)# threshold to declare opacity
        
        #deleting some outliers for the plot
        c=0
        for i in range(512):
            for j in range(512):
                if InfectionMask[i,j]==1:
                    c=c+1
                    
        infMask_db=np.zeros((c,2),float)
        
        c=0
        for i in range(512):
            for j in range(512):
                if InfectionMask[i,j]==1:
                    infMask_db[c]=[i,j] 
                    c=c+1      #write the coordinates of infected pixels into an array to perform DBSCAN
                                  
        clusters=DBSCAN(eps=2,min_samples=5, metric='euclidean').fit(infMask_db)
        a,Npoints_per_cluster=np.unique(clusters.labels_,return_counts=True)
        
        infMask_db_wo=np.zeros((512,512),float)
        for i in range(c):
                if clusters.labels_[i]!=-1:
                    a=int(infMask_db[i][0])
                    b=int(infMask_db[i][1])
                    infMask_db_wo[a][b]=1
        infMask_db_wo[infMask_db_wo==0]=np.nan
        
        InfectionMask[InfectionMask==0]=np.nan
   
        plt.figure()
        plt.imshow(InfectionMask,interpolation="nearest")
        plt.title('infection mask')
        total=np.count_nonzero(self.LungsMask==1) #total pixel of lungs
        selected=np.count_nonzero(InfectionMask==1)  #infected pixels
        percentage=selected/total*100
        print(f"The percentage of infected area is {np.round(percentage,1)}\n")
        
        color_map = 'spring'
        plt.figure()
        plt.imshow(self.sct,alpha=0.8,vmin=self.vm,vmax=self.vM, cmap='bone')
        plt.imshow(InfectionMask*255,alpha=1,vmin=0,vmax=255, cmap=color_map,interpolation="nearest")
        plt.title('Original image with ground glass opacities in yellow')
        color_map = 'spring'
        plt.figure()
        plt.imshow(self.sct,alpha=0.8,vmin=self.vm,vmax=self.vM, cmap='bone')
        plt.imshow(infMask_db_wo*255,alpha=1,vmin=0,vmax=255, cmap=color_map,interpolation="nearest")
        plt.title('without some outliers')
        #analysis on left lung, percentage of infected pixels and decsription of the infection distribution
        B=self.LLungMask*self.sct
        inf_mask=1*(B>Bmin)&(B<Bmax)
        InfectionMask=self.filterImage(inf_mask,1)
        InfectionMask=1.0*(InfectionMask>thresh)# threshold to declare opacity
        InfectionMask[InfectionMask==0]=np.nan
        total=np.count_nonzero(self.LLungMask==1) #total pixel of left lung
        selected=np.count_nonzero(InfectionMask==1)   #infected pixels
        percentage=selected/total*100
        chosen_eps=5      #those parameters are chosenen to find the distribution of points, actually not focusing on 'typical clustering'
        chosen_min_samples=2
        infMask_db=np.zeros((selected,2),float)
        c=0
        for i in range(512):
            for j in range(512):
                if InfectionMask[i,j]==1:
                    infMask_db[c]=[i,j]   #write the coordinates of infected pixels into an array to perform DBSCAN
                    c=c+1                
        clusters=DBSCAN(eps=chosen_eps,min_samples=chosen_min_samples, metric='euclidean').fit(infMask_db)
        a,Npoints_per_cluster=np.unique(clusters.labels_,return_counts=True)
        #deleting outliers
        count_big=0
        count_small=0
        conc=0.03*total
        small=0.005*total
        for i in range(1,len(Npoints_per_cluster)):
            if Npoints_per_cluster[i]>=conc:
                count_big=count_big+1
            elif Npoints_per_cluster[i]<=small:
                count_small=count_small+1
        
        print(f"The percentage of infected area in the left lung is {np.round(percentage,1)}")
        print(f"In the left lung there are:\n-{count_big} zones with a large concentration (at least the 3% of the lung)\n-{len(a)-count_small-count_big-1} medium zones (0.5-3%)\n-{count_small+Npoints_per_cluster[0]} small spread zones (<0.5%)\n")
       
        #analysis on right lung, percentage of infected pixels and decsription of the infection distribution
        B=self.RLungMask*self.sct
        inf_mask=1*(B>Bmin)&(B<Bmax)
        InfectionMask=self.filterImage(inf_mask,1)
        InfectionMask=1.0*(InfectionMask>thresh)# threshold to declare opacity
        InfectionMask[InfectionMask==0]=np.nan
        total=np.count_nonzero(self.RLungMask==1)  #total pixel of right lung
        selected=np.count_nonzero(InfectionMask==1)   #infected pixels
        percentage=selected/total*100
        infMask_db=np.zeros((selected,2),float)
        c=0
        for i in range(512):
            for j in range(512):
                if InfectionMask[i,j]==1:
                    infMask_db[c]=[i,j]     #write the coordinates of infected pixels into an array to perform DBSCAN
                    c=c+1                
        clusters=DBSCAN(eps=chosen_eps,min_samples=chosen_min_samples, metric='euclidean').fit(infMask_db)
        a,Npoints_per_cluster=np.unique(clusters.labels_,return_counts=True)
        count_big=0
        count_small=0
        conc=0.03*total
        small=0.005*total
        for i in range(1,len(Npoints_per_cluster)):
            if Npoints_per_cluster[i]>=conc:
                count_big=count_big+1
            elif Npoints_per_cluster[i]<=small:
                count_small=count_small+1
         
        print(f"The percentage of infected area in the right lung is {np.round(percentage,1)}")
        print(f"In the right lung there are:\n-{count_big} zones with a large concentration (at least the 3% of the lung)\n-{len(a)-count_small-count_big-1} medium zones (0.5-3%)\n-{count_small+Npoints_per_cluster[0]} small spread zones (<0.5%)\n")
       

if __name__=='__main__':
    ct=CTscan()
    ct.readFile()
    ct.plotHistogram()
    ct.colorQuantizationAndPlot()
    ct.findLungs()
    ct.darkestColorsImage()
    ct.findLungMask()
    ct.groundGrassOpacities()
