#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:05:57 2018

@author: sedna
"""
#!/usr/bin/env python3.6
import numpy as np


class MetricBinaryClassifier():
    """Confusion matrix in statsmodels"""
    def __init__(self):
        self._paramaters=None
        
    def count(self,x):
           n=0
           for i in x:
               if i==True:
                   n+=1
           return n
       
    def metrics_binary(truth,predicted,misclassif):
        
        if len (truth) != len (predicted ):
            raise Exception ( " Wrong sizes ... " )
        total = len (truth )
        if total == 0:
            return 0
        
        #truth=np.array(truth).reshape(total,1)
        #predicted=np.array(predicted).reshape(total,1)
        # checking empty set putting array.size()>0
        
        f=lambda x,y: x == y==True
        t=lambda x,y: x==y
        h=lambda x:x==0
        g=lambda x:x==1
        
        #TRUTH :NEGATIVE
        atn=list(map(h,truth))
        #TRUTH :POSITIVE
        atp=list(map(g,truth))
        # TEST:FALSE
        apn=list(map(h,predicted))
        # TEST:TRUE
        app=list(map(g,predicted))
        # P[X=1] TRUTH POSITIVE
        pt1=atp.count(True)/total
        # P[X=0] TRUTH NEGATIVE
        pt0=atn.count(True)/total
        # P[X*=1] TEST TRUE
        pp1=app.count(True)/total
        # P[X*=0] TEST FALSE
        pp0=apn.count(True)/total
       
        BIAS=pp1-pt1
     
        fp_index=[]
        fn_index=[]
        #TN+TP
        a=list(map(t,truth,predicted))
        TNTP=a.count(True) 
        #TN
        a=list(map(f,atn,apn))
        TN=a.count(True)
        #FP
        a=list(map(f,atn,app))
        FP=a.count(True)
        aa=[i for i,x in enumerate(a) if x==True]
        fp_index.append(aa)
        
        
        #FN
        a=list(map(f,atp,apn))
        FN=a.count(True)
        aa=[i for i,x in enumerate(a) if x==True]
        fn_index.extend(aa)
        
        #TP
        a=list(map(f,atp,app))
        TP=a.count(True)
        
        if (TN+TP+FN+FP!=total):
            print("Error in metrics_classification TN+TP+FN+FP!=total")
        try:   
            acc=float(TNTP)/ total
        except:
            print("Error in metrics_classification")
        try:
            tpr=TP/(TP+FN)
        except:
            tpr=0
        try:
            tnr=TN/(TN+FP)
        except:      
            tnr=0
        try:
            ppv=TP/(TP+FP)
        except:
            ppv=0
        try:
            dor=(TP*TN)/(FP*FN)
        except:
            try:
                dor=(TN+FP)/(TP+FN)
            except:
                dor=TN/FP
        try:
            fpr=1-tnr
            fnr=1-tpr   
        except:
            print("Error in metrics_classificiation")  
        
        ff=100*np.array([acc,tpr,tnr,ppv,fpr,fnr,dor,pt1,pp1])
        acc,tpr,tnr,ppv,fpr,fnr,dor,pt1,pp1=[float("%.3f"%(a)) for a in ff]
        fp_index=np.array(fp_index)           
        fn_index=np.array(fn_index)
        
    
        #print(pt0,pp0)
        #print(pt1,pp1)
        if misclassif==True:
            return acc,TP,TN,FP,FN,BIAS,pt1,pp1,TNTP,total,fp_index,fn_index
        else:
            return acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1,fp_index,fn_index
       
