#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 14:46:37 2018

@author: sedna
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from biokit.viz import corrplot         
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd
from statsmodels.multivariate.pca import PCA as smPCA
from sklearn.decomposition import PCA as skPCA
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor     
from metrics_classifier_statsmodels import MetricBinaryClassifier as ac
from tools import Tools
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA            
        
colors = [ic for ic in mcolors.BASE_COLORS.values() if ic !=(1,1,1)]

class Data_base:
    
      def data_generation():
          """LOADING THE PIMA DATA SET"""
          data_train={}
          data_test={}
          pima_tr = pd.read_csv('pima.tr.csv', index_col =0)
          pima_te = pd.read_csv ('pima.te.csv', index_col =0)
          # Training aata
          df=pima_tr
          
          #df.dropna(how="all", lace=True) # drops the empty line at file-end
          columns=df.columns
          columns_train=columns[:7]
          columns_test=columns[-1]
          X_train=df[columns_train]
          Y_train=df[columns_test]
          # Testing data
          de=pima_te
          X_test=de[columns_train]
          Y_test=de[columns_test]
          data_train['train']=[columns_train,X_train,Y_train,df]
          data_test['test']=[columns_test,X_test,Y_test,de]
          return data_train,data_test
    
      def data_generation_binary_classification(**kwargs):
          """Generation of dummy variables to Binary Classification Task"""     
         
          _,X_train,Y_train,_=Tools.data_extract_dict_to_list('train',**kwargs)
          _,X_test,Y_test,_=Tools.data_extract_dict_to_list('test',**kwargs)
       
          data_dummy_train={}
          data_dummy_test={}
          # Dummy variables to categorical variable
          V_train=Tools.data_dummy_binary_classification(Y_train,"No",0,1)
          V_test=Tools.data_dummy_binary_classification(Y_test,"No",0,1)
                
          # Mapping 'Training and Testing data to [0,1]".X->U
          U_train=Tools.mapping_zero_one(X_train)
          U_test=Tools.mapping_zero_one(X_test)
         
          data_dummy_train['train']=[U_train,V_train]
          data_dummy_test['test']=[U_test,V_test]
         
          return data_dummy_train,data_dummy_test
         

      def data_head_tail(data):
          head=data.head()
          tail=data.tail()
          return Tools.table(head,'.2f','simple','Data Head Pima Dataset',60),\
          Tools.table(tail,'.2f','simple','Data Tail Pima Dataset',60)

      def data_feature_show(data):
          sns.countplot(x="type",data=data,palette='hls')
          plt.title("Binary Categorical Variable(Yes/No)")
          return plt.show()
      
      def data_features_show(data):
          sns.countplot(x="age",hue="type", data=data[::5], orient='h',palette='Set1')
          plt.title("Behaviour of the Diabetes with the age")
          return plt.show()
      
      def data_describe(data):
          """Descriptive Statistics of data."""
          data_described=data.describe()
          """Grouped Data by the type[yes/no] showing mean values""" 
          data_mean=data.groupby('type').mean()
          return Tools.table(data_described,'.3f','simple','Descriptive Statsistics of Pima Data',60),\
          Tools.table(data_mean,'.3f','simple', ' Data Mean ', 60) 
        
      def data_features_draw_scatter(data):
         """Scatter Plot of data."""
         sns.set(style="ticks",palette=colors[:7])
         sns.pairplot(data)
         plt.title("Scatter Plot of data")
         return plt.show()

      def data_features_draw_hist(data,n_bins):
         """Visualization of statistical distributions.""" 
         np.random.seed(19680801)
         n_bins=10
         df=data
         dfs=df.describe()
         columns=df.columns
         fig, axs = plt.subplots(nrows=4, ncols=2)
         #colors = [ic for ic in mcolors.BASE_COLORS.values() if ic !=(1,1,1)]

         ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7=axs.flatten()
         
         ax=np.array([ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7])
         
         for ii in range(len(columns)):
            data=df.iloc[:,ii]
            count,mean,std,min,Q25,Q50,Q75,max=dfs.iloc[:,ii]
            bins=np.linspace(min,max,100)
            ax[ii].hist(data, n_bins, density=True, histtype='bar', color=colors[ii])
            ax[ii].legend(prop={'size': 10})
            bin_centers = 0.5*(bins[1:] + bins[:-1])
            pdf = stats.norm.pdf(bin_centers,mean,std)
            ax[ii].plot(bin_centers,pdf,color=(0.3,0.5,0.2),lw=2)
    
         #    ax[ii].set_title(df.columns[ii])
    
         for ax in axs.flat[len(ax)-1:]:
             ax.axis('off')
 
         plt.suptitle('Pima Datasets [type~npreg+glu+bp+skin+bmi+ped+age]')
         fig.tight_layout(rect=[0, 0.03, 1, 0.95])
         return plt.show()

class Correlation:
    """Compute correlation of data and inform about its strength."""
    def __init__(self,data,columns,treshold=0.9):
        self.x=data
        self.columns_subset=columns # selection of columns
        self.corr_treshold=treshold
    
    def correlation_training(self):
        """Investigation of training data correlation.""" 
        self.x=pd.DataFrame(self.x)
        dfcorr=self.x.corr()
        Tools.table(dfcorr,'.3f','fancy_grid','Correlation Matrix on Training Data', 60)
        c=corrplot.Corrplot(dfcorr)
        c.plot()
        plt.suptitle("Pair-Correlation Matrix on Training Data")
        return plt.show()
   
    def correlation_level(self):
        """Identify significative correlations"""
        self.x=pd.DataFrame(self.x)
        print("\n","Treshold value , correlation=%.2F"%( self.corr_treshold),sep='\n')
        mcorr=self.x.corr() 
        bb=np.triu(mcorr)-np.diag(mcorr)
        aa=np.where(bb > self.corr_treshold)
        LL=len(aa[0])
        mcorr_low=[]
        mcorr_high=[]
        if (LL>0):
            for ii in range(LL):
                a=self.x.columns[aa[0][ii]]   
                mcorr_high.append(a)
                b=self.x.columns[aa[1][ii]]
                mcorr_high.append(b)
                
            mcorr_low=[names for names in self.x.columns if names not in mcorr_high]
            return print("The following predictors are highly pair-correlationed:"\
                         + ','.join(str(name) for name in mcorr_high),sep='\n'), print("\n",\
                    "The following predictors are midly pair-correlationed "\
                     +','.join(str(name) for name in mcorr_low),sep='\n')
        else:
             return  print("\n\n","The predictors are slightly correlationed",sep="\n")
   
        
class PCA:
    """PCA Analysis."""  
    def __init__(self,x,n_components):
         self.x=x
         self.ncomp=n_components
         
    def pca(self):
         pc=smPCA(self.x,self.ncomp) 
         factors=pc.factors.values[:]
         eigvalues=pc.eigenvals
         eigvectors=pc.eigenvecs.values[:]
         
         # Make a sorted list of (eigenvalue, eigenvector)  
         
         order=[ii for ii,vals in sorted(enumerate(np.abs(eigvalues)), \
                                         key=lambda x:x[1],reverse=True)]
          
         major_factor=factors[:,order[0]]
         minor_factor=factors[:,order[-1]]
         
         
         eigvalues_sorted=[eigvalues[order[ii]] for ii in range(self.ncomp)]
         eigvectors_sorted=[eigvectors[:,order[ii]] for ii in range(self.ncomp) ]
         
         
         tot = sum(eigvalues_sorted)
         var_exp = [(i / tot)*100 for i in eigvalues_sorted[:]]
         cum_var_exp = np.cumsum(var_exp)
         return major_factor,minor_factor,factors,eigvalues_sorted,eigvectors_sorted,var_exp,cum_var_exp


    def pca_draw_major_minor_factors(self,target):     
        """Draw the major and minor PCA factor."""
        """WORKING TO GENERALIZE"""
        
        major_factor,minor_factor,_,_,_,explained_variance_,_=PCA(self.x,self.ncomp).pca()
        mx=max(explained_variance_)
        mean_=np.mean(explained_variance_)/(100*mx)
        
        LENGTHS=[explained_variance_[0]/mx,explained_variance_[-1]/mx]
        
        components=np.array([[major_factor[0],minor_factor[0]],
                             [-minor_factor[0],major_factor[0]]])
    
        plt.figure()
        color=['Darkblue','red']
        ax=plt.subplot(1,1,1)
        ax.scatter(major_factor,minor_factor,c=color, marker='o',edgecolor='none',\
                   alpha=0.8,s=40)
        for length, vector in zip(LENGTHS,components):
            v = mean_+ 4*np.sqrt(length)*vector
            Drawing2d.draw_vector(v[0],v[1],mean_,mean_)
            
        plt.xlabel("PCA factor(major axis]")
        plt.ylabel("PCA factor(minor axis]")
        plt.title("Major PCA Factor vs Minor PCA Factor")
        plt.axis("equal")
        return plt.show()
    
    def pca_show_table(self):
         "Table of eigvalues and eigvectors in reverse order."
         _,_,_,eigvalues,eigvectors,_,cum_var_exp=PCA(self.x,self.ncomp).pca()
    
         #eig_data=[(self.x.columns[i],np.abs(eigvalues[i]), \
         #          cum_var_exp[i],eigvectors[i]) for i in range(self.ncomp)]
         eig_data=[(self.x.columns[i],np.abs(eigvalues[i]), \
                    cum_var_exp[i]) for i in range(self.ncomp)]
             
         eig_data=pd.DataFrame(eig_data,columns=['Predictors','Eig.Abs','cum_var_exp'] )
         #eig_data=pd.DataFrame(eig_data[:,2],columns=['obs_var','Eig.Abs',\
         #                        'cum_var_exp','Eigenvectors'] )
         #return Tools.table(eig_data,'.3f','fancy_grid','Eigenvalues in descending order',60)
         

         return Tools.table(eig_data,'.2f','fancy_grid','Eigenvalues in descending order',60)
             
             
    def pca_draw_by_components(self):   
         """Visualization PCA by components."""
         
         _,_,_,eigvalues,eigvectors, var_exp,cum_var_exp=PCA(self.x,self.ncomp).pca()
        
         plt.Figure()
         ax=plt.subplot(1,1,1)
         width=0.40
         x_draw=[self.x.columns[i] for i in range(self.ncomp)]
         y_draw=var_exp
         z_draw=cum_var_exp
         plt.bar(x_draw,y_draw,width,color=colors[:7])
         plt.scatter(x_draw,z_draw)
         ax.plot(x_draw,z_draw,'y')
         plt.ylabel('Cumulative explained variance')
         plt.xlabel('Observed variables')
         plt.title("Visualization of PCA by components")
         return plt.show()
    
    
    def pca_transformation(self):
        """To apply PCA transformation to Training Data."""
        
        pc=skPCA(n_components=self.ncomp)
        X_pca=pc.fit_transform(self.x)
        X_pca=pd.DataFrame(X_pca,columns=self.x.columns)
        return X_pca
    

class Best_features_filter:
    """ To apply criteria to seek features's subsample."""
    """
     1.z_score
    """
     
    def __init__ (self,x,columns_subset,vif_treshold):
        self.x=x
        self.columns_subset=columns_subset
        self.vif_treshold=vif_treshold

    def variance_influence_factors(self):
        
        
        "Analysis of variance influence factors or colinearity"
        vif = pd.DataFrame()
        vif["features"] = self.columns_subset
        X_vif=np.asarray(self.x[self.columns_subset])
        vif["VIF Factor"] = [variance_inflation_factor(X_vif, i) for i in range(X_vif.shape[1])]
        vif=np.asarray(vif)
        vif =dict(vif)
        vif_inverse={}
        for key,value in vif.items():
            vif_inverse[value]=key
        keys_ordered=[vif_inverse[name] for name in sorted(vif.values(),reverse=True)]
        vif_ordered=pd.DataFrame(index=keys_ordered,columns=['VIF Factor'])

        for key in keys_ordered:
            vif_ordered.loc[key,:]=vif[key]


        key_colinearity= vif_ordered.index[np.where(vif_ordered.iloc[:,0]>self.vif_treshold)]

        return Tools.table(vif_ordered,'.3f','simple',"Variance Influence Factors",0),\
        print("\n","The following features are collinears: "\
                     + ','.join(key for key in key_colinearity),"\n",sep="\n")
   
    
class Best_features_wrap:
    
    def __init__(self):
        self.parameters=None
    
    
    def z_score(self,z_score_table,z_score_treshold):
        """Z_score criteria """
        f=lambda z: z > z_score_treshold
        z_dropped=np.mat(list(map(f,z_score_table.values)))
        Indicator=np.all(z_dropped,axis=1)
        ind_drop=np.where(Indicator==True)
        cols=z_score_table.iloc[ind_drop].index
        columns=[str(name) for name in cols]
        columns_Index=pd.Index(columns)
        columns_list=pd.Index.tolist(cols)
    
        if len(columns) >0:
            print("\n\n","========================================================", \
                      'There are predictors strongly significative to level  0.025 :',\
                      ','.join (str(nn) for nn in columns),
                         "========================================================", sep='\n')
        else:
             print("\n\n","========================================================", \
                      "There aren't predictors strongly significative to level  0.025 ",\
                         "========================================================", sep='\n')
        
        return columns_list,columns_Index

    def cross_validation_binary_classification(self,names,cols_data,cols_base_validation,cols_index,train_exog,\
                              x_test_exog,endog,y_test,family,\
                          method,par_reg,task,Iperfor0,K_fold,mis_K_classif):
        """Cross-validation to Binary Classification"""
        

        x_train=pd.DataFrame(train_exog)
        y_train=pd.DataFrame(endog)
        NN=len(x_train)

        cols_base_base=cols_base_validation.copy()
        cols_index_base=cols_index.copy()
     
        N_features=len(cols_data) - len(cols_base_validation)        
        
        x_train.rename(index=lambda x:x-1,inplace=True)
        
        index_fold=np.ones((NN,1))
        index_fold=np.cumsum(index_fold)-1
        index_fold=pd.DataFrame(index_fold)
        
        PP=np.cumsum(y_train)
        PP1=max(PP.iloc[:,0])
        #PP0=NN-PP1
    
        N_1=PP1/NN

        N_split=int(NN/K_fold) 
        #N_split_last=NN-(K_fold-1)*N_split
        N_1_split=int(N_1*N_split)
        N_0_split=N_split-N_1_split
        
        index_0,index_1=Tools.get_pandas_from_groupby(index_fold,y_train,2)
        index_0=[int(ii) for ii in index_0]
        index_1=[int(ii) for ii in index_1]
        
        index_fold=index_fold.T
        i0=[]
        i1=[]
        index_test_fold=[]
        LENINDEX=0
        LENINDEX0=0
        LENINDEX1=0
        for ii in range(K_fold):
            if ii < K_fold-1:
               i0=index_0[ii*N_0_split:(ii+1)*N_0_split]
               i1=index_1[ii*N_1_split:(ii+1)*N_1_split]
               i01=sorted(np.concatenate((i0,i1))) 
               LENINDEX+=len(i01)
               LENINDEX0+=len(i0)
               LENINDEX1+=len(i1)
               index_test_fold.append(i01)  
            else:
             
               i0=index_0[(K_fold-1)*N_0_split:NN]
               i1=index_1[(K_fold-1)*N_1_split:NN]
               i01=sorted(np.concatenate((i0,i1)))
               index_test_fold.append(i01)   
               LENINDEX+=len(i01)
        if (LENINDEX==index_fold.shape[1]):
            pass
        else:
            ip0=[len(index_0)-LENINDEX0 -len(i0)]
            ip1=[len(index_1)-LENINDEX1 -len(i1)]
            print("Error in computing index_test_fold, LENINDEX:",LENINDEX,ip0,ip1)
            input()
        data_fold={}
        Iperformance={}
        for ii in range(K_fold):
            
            index_train_fold=Tools.list_minus_list(index_fold,index_test_fold[ii])
            
            x_test_fold=x_train.iloc[index_test_fold[ii],:]
            y_test_fold=y_train.iloc[index_test_fold[ii],:]
           
            y_test_fold=np.array([ii for ii in y_test_fold.iloc[:,0]])

            
            x_train_fold=x_train.iloc[index_train_fold,:]
            y_train_fold=y_train.iloc[index_train_fold,:]
            
            y_train_fold=np.array([ii for ii in y_train_fold.iloc[:,0]])
            
            data_fold[ii] =[index_train_fold,x_train_fold,y_train_fold,\
                     x_test_fold,y_test_fold]
            
            """
            Iperfor,model_name_selected,columns_base,y_predicted_table,\
            y_estimated_table,params_table,residuals_table
            """
            Iperformance[ii]=\
            Best_features_wrap().add_feature_selection(names,cols_data,cols_base_base,\
                              cols_index_base,x_train_fold,\
                              x_test_fold,y_train_fold,y_test_fold,family,method,\
                              par_reg,task,Iperfor0,K_fold,mis_K_classif)

            #updating
            cols_base_base=cols_base_validation.copy()
            cols_index_base=cols_index.copy()

        return K_fold,N_features,data_fold,Iperformance
    
  
    def add_feature_selection(self,names,cols_data,cols_base_features,cols_index,train_exog,\
                              test_exog,endog,y_test,family,\
                          method,par_reg,task,Iperfor0,K_fold,mis_A_classif):
            """Feature and method selection based on z_score."""
            """
            Input:
                
            | names-List of models
            | Iperfor0-Percentage of performance desired
            | Iperfor -Percentage of performance obtained
            | data-data from datasets
            | cols_index-Parameter index
            | cols_base_features-cols feature to begin selection
            | U_train_exog,U_test_exog: exog from data
            | V_train,V_test :endog from data
            | Parameters to run simulation_statsmodels
            | family
            | method
            | par_reg
            | task
            |i-fold--index of fold
            |K-fold -number of folds
            
            Output:
            | Iperformance-Dict [Iperfor]=[model_name,cols_features]
                
            """
        
            Iperfor=0
            # Two-columns + 1
            cols_base=cols_base_features
            N=len(cols_base)
            N_base=N
            
            Iperformance_1={}
        
            while ((Iperfor <Iperfor0) or (len(cols_base) < len(cols_data)+1)):
                
                 columns_Index=cols_index.copy()
                 
                 columns=cols_base.copy()
                 
                 
                 columns_diff=Tools.list_minus_list(cols_data,columns)
                 
        
                 Iperformance_2={}
                 
                 N_1=len(cols_data)-len(columns_diff)-2
                 #  Looking the best features on the remainder set
                 
                 for nn in columns_diff:
                
                    columns.append(nn)
                    try: 
                        columns_Index.append(nn)
                    except:
                        columns_Index=Tools.\
                        add_index_to_list_of_indexes(columns_Index,nn,True)
                    
                    exog=train_exog[columns_Index]
                    x_test=test_exog[columns_Index]
                
                    y_predicted_table,y_estimated_table,params_table,residuals_table,\
                    _,_,FPR,TPR,confusion_matrix,\
                    to_model_names=Statsmodels_linear_filter(names,exog,endog,x_test,y_test,family,method,par_reg,task,mis_A_classif).\
                    statsmodels_linear_supervised()
                    
                    
                    #if (Iperfor > 80):
                    # Performance by group of features and algorithms
                       
                    Title='Confusion Matrix '+ ' with ' + str(len(columns))  + ' predictors :'+ columns[0]
                    for name in columns[1:]:
                                Title += ','+ name
                  
                    Table_results(6,confusion_matrix,'.2f','fancy_grid',Title,60).print_table()
                    
                    # Get a maximum performance
                    
                    model_name_selected,model_name_list,Imax,PPV,Iperfor=\
                    Tools.max_in_col_matrix(confusion_matrix,'ACC','PPV',Iperfor)
                    
                    # Checking len(model_name_list) > 1 
                                    
                    Iperformance_2[nn]=[Imax,PPV,model_name_selected,nn,y_predicted_table,\
                                  y_estimated_table,params_table,residuals_table,columns]
    
    
                    if Iperfor > Iperfor0:
                        print("The Best performance is {0:.2f}%".format(Iperfor))
                 
                    columns=cols_base.copy()
                    columns_Index=cols_index.copy()
            
                # Updating
                            
                 N+=1
                 
                 if (len(cols_base)==len(cols_data)):
                     
                     break
                 
                 Idata=pd.DataFrame(index=columns_diff,columns=['ACC','PPV'])
                
                 for nn in columns_diff:
                     Idata.loc[nn,'ACC']=Iperformance_2[nn][0]
                     Idata.loc[nn,'PPV']=Iperformance_2[nn][1]
                     
                 predictor,predictor_list,Imax,PPV,Iperfor=Tools.max_in_col_matrix(Idata,'ACC','PPV',Iperfor)
                   
                 Iperformance_1[N_1]=[Iperformance_2[predictor][0],Iperformance_2[predictor][1],Iperformance_2[predictor][8],\
                                Iperformance_2[predictor][2],\
                                Iperformance_2[predictor][4],Iperformance_2[predictor][5],\
                                Iperformance_2[predictor][6],Iperformance_2[predictor][7]]
    
                 # Checking len(predictor_list) > 1 
            
                 cols_base.append(Iperformance_2[predictor][3])
                 try:
                     cols_index.append(Iperformance_2[predictor][3])
                 except:
                     cols_index=Tools.\
                     add_index_to_list_of_indexes(cols_index,Iperformance_2[predictor][3],True)
            
            if (K_fold==1):     
                
                N_features=min(len(cols_data)-N_base,N_1)
                N_Idata=K_fold*N_features
                # Get the Best performance 
                Idata=pd.DataFrame(index=range(N_Idata),columns=['ACC','PPV'])
               
                for mm in range(N_features):
                        
                        Idata.loc[mm,'ACC']=Iperformance_1[mm][0]
                        Idata.loc[mm,'PPV']=Iperformance_1[mm][1]
                
                predictor,predictor_list,Imax,PPV,Iperfor=\
                Tools.max_in_col_matrix(Idata,'ACC','PPV',Iperfor0)
                
                if len(predictor_list)==0:
                    return print("Error in Tools.max_in_col_matrix")
                
                nn,mm=Tools.get_max_from_multi_index(predictor_list,N_features,1)
                Title=Iperformance_1[mm][3] + ' with ' + str(mm+3) + ' predictors :' 
                Title+=','.join(str(name) for name in Iperformance_1[mm][2])
                                         
                print("The Best performance is {0:.2f}%".format(Imax) + " using", Title ,sep='\n')
            
                model_names=Iperformance_1[mm][3]
                    
                columns_base=Iperformance_1[mm][2]
                
                y_predicted_table_selected=Iperformance_1[mm][4]
                        
                y_estimated_table_selected=Iperformance_1[mm][5]
               
                params_table_selected=Iperformance_1[mm][6]
                
                residuals_table_selected=Iperformance_1[mm][7]  
                         
            
                return Iperfor,model_names,columns_base, y_predicted_table_selected,\
                            y_estimated_table_selected,params_table_selected,residuals_table_selected
            
            else: 
                return Iperformance_1
           
    def draw_K_fold_numerical_results(self,K_fold,N_cols_base,N_features,data_fold,Iperformance):
        """...Draw K_fold performance :ACC and PPV..."""
        import pandas as pd
        from simulation_statsmodels import Draw_numerical_results,Table_results
    
       # Get the Best performance 
       # Checking the number of columns_base :2 for thr future could change
        ACC_data=pd.DataFrame(index=range(K_fold),columns=range(1+N_cols_base,1+N_cols_base+N_features))
        PPV_data=pd.DataFrame(index=range(K_fold),columns=range(1+N_cols_base,1+N_cols_base+N_features))



        for nn in range(K_fold):
            for mm in range(N_features):
                ACC_data.loc[nn,1+N_cols_base+mm]=Iperformance[nn][mm][0]
                PPV_data.loc[nn,1+N_cols_base+mm]=Iperformance[nn][mm][1]    
        
        
        ACC_data_mean=pd.DataFrame(
                      index=['ACC Mean'],columns=range(1+N_cols_base,1+N_cols_base+N_features))
        PPV_data_mean=pd.DataFrame(
                      index=['PPV Mean'],columns=range(1+N_cols_base,1+N_cols_base+N_features))
        ACC_data_mean.loc['ACC Mean',:]=ACC_data.mean()[:]
        PPV_data_mean.loc['PPV Mean',:]=PPV_data.mean()[:]
       
        Table_results(10,ACC_data,'.2f','fancy_grid','ACC K_fold Cross-Validation in Binary Classification',40).print_table()
       
        Table_results(12,ACC_data_mean,'.2f','fancy_grid', ' ACC Data Mean vs Features', 50).print_table()
        
        Title="K_Fold vs Features : Cross-Validation to Binary Classification"
        
        # Text is not garantized inside the box draw..Why???
        Draw_numerical_results.frame_from_dict(ACC_data,"Folds","ACC",Title,'equal',True,"ACC[ K_Fold , Number of Features]",'square')
        

        
        Table_results(11,PPV_data,'.2f','fancy_grid','PPV in K_fold Cross-Validation to Binary Classification',40).print_table()
        Table_results(13,PPV_data_mean,'.2f','fancy_grid', ' PPV Data Mean vs Features', 50).print_table()
        
        Title="K_Fold vs Features : Cross-Validation to Binary Classification"
         
        Draw_numerical_results.frame_from_dict(PPV_data,"Folds","PPV",Title,'equal',True,"PPV [K_Fold, Number of Features]",'square')
    
    
    def K_fold_full_prediction_results(self,K_fold,N_features,data_fold,Iperformance, Iperfor0,x,y,mis_K_classif):
        """Prediction using K_fold Cros-Validation splititng Learning"""
        
        N_Idata=K_fold*N_features
        # Get the Best performance 
        Idata=pd.DataFrame(index=range(N_Idata),columns=['ACC','PPV'])
        
        for nn in range(K_fold):
            for mm in range(N_features):
                ii= N_features*nn + mm
                Idata.loc[ii,'ACC']=Iperformance[nn][mm][0]
                Idata.loc[ii,'PPV']=Iperformance[nn][mm][1]
                
        predictor,predictor_list,Imax,PPV,Iperfor=\
        Tools.max_in_col_matrix(Idata,'ACC','PPV',Iperfor0)
        
        for ii in predictor_list:
            nn,mm=divmod(ii,N_features)
            
            columns_base=Iperformance[nn][mm][2]


            x_train_fold=data_fold[nn][1]
            y_train_fold=data_fold[nn][2]
            
            x_train_fold_exog=x_train_fold[columns_base]
        
            names=['Logit','GenLogit'] 
            exog=x_train_fold_exog
            endog=y_train_fold
            x_test= x[columns_base]
            #Checking 
            y_test=y
            family='Binomial'
            method=''
            par_reg=[]
            task="BinaryClassification"
            
            y_calibrated_table,y_estimated_table,params_table,residuals_table, \
            fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
            to_model_names=Statsmodels_linear_filter(names,exog,endog,x_test,y_test,family,method,par_reg,task,mis_K_classif).\
            statsmodels_linear_supervised()
        
            Table_results(0,params_table,'.3f','fancy_grid','Models Parameters',30).print_table()

            Table_results(6,confusion_matrix,'.2f','fancy_grid','Confusion Matrix ',60).print_table()
        
            kind="Prediction"
            Title="Prediction in Binary Classification using statsmodels"
            
            params=params_table.T
            
            Draw_binary_classification_results(FPR,TPR,names,params,\
                                x_train_fold_exog,y_train_fold,x,\
                                y,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns_base,Title,kind).draw_mis_classification()
            
    def K_fold_numerical_results(self,K_fold,N_cols_data,N_features,data_fold,\
                                 Iperformance,Iperfor0):
        """Get numerical results from K_fold crros-validation...."""
        
        for ii in range(K_fold):
            print ('Fold=',ii)
            print(Iperformance[ii].keys())
            print("========================")
    
        N_Idata=K_fold*N_features
        # Get the Best performance 
        Idata=pd.DataFrame(index=range(N_Idata),columns=['ACC','PPV'])
        
        for nn in range(K_fold):
            for mm in range(N_features):
                ii= N_features*nn + mm
                Idata.loc[ii,'ACC']=Iperformance[nn][mm][0]
                Idata.loc[ii,'PPV']=Iperformance[nn][mm][1]
                
        predictor,predictor_list,Imax,PPV,Iperfor=\
        Tools.max_in_col_matrix(Idata,'ACC','PPV',Iperfor0)
        
        # CHECKING THIS CRITERIA TO BEST SELECTION
        # HERE IS ADOPTED THE CRITERIA TO SELECT THE FEATURES LARGEST SUBSET 
        
        nn,mm=Tools.get_max_from_multi_index(predictor_list,N_features,1)
        
        Title=Iperformance[nn][mm][3] + ' with ' + str(mm+1+N_cols_data) + ' predictors :' 
        Title+=','.join(str(name) for name in Iperformance[nn][mm][2])
        Title+='\n '+'splitting in ' + str(K_fold) + \
        ' folds and testing the fold number: ' + str(nn)
        print("\n","The Best performance is {0:.2f}%".format(Imax) + \
              "  using", Title,"\n" ,sep='\n')
    
        
        model_name_selected=Iperformance[nn][mm][3]
        
        columns_base=Iperformance[nn][mm][2]


        x_train_fold=data_fold[nn][1]
        y_train_fold=data_fold[nn][2]
        
        x_test_fold=data_fold[nn][3]
        y_test_fold=data_fold[nn][4]
        
        x_train_fold_selected=x_train_fold[columns_base]
        x_test_fold_selected=x_test_fold[columns_base]
        
        y_predicted_table=Iperformance[nn][mm][4]
        
        y_estimated_table=Iperformance[nn][mm][5]
        
        params_table=Iperformance[nn][mm][6]
        
        residuals_table=Iperformance[nn][mm][7]
        
        return  Iperfor,model_name_selected,columns_base,x_train_fold_selected,y_train_fold,\
                x_test_fold_selected,y_test_fold,y_predicted_table,\
            y_estimated_table,params_table,residuals_table   
            
#class statsmodels_wrap(statsmodels_filter):
    
class Statsmodels_simula:
    """Simulation with statsmodels."""
    
    """
    x--X_test
    y--Y_test
    family--pdf to use in the model
    """
    def __init__(self,name,exog,endog,x,y,family,method,par_reg,task):
        # to be overridden in subclasses
        self.name=None
        self.model_name=None
        self.model=None
        self.exog= None
        self.endog=None
        self.x=None
        self.y=None
        self.columns=None
        self.family=None
        self.method=None
        self.par_reg=[]
        self.task=None
        self.linear=False
        self.misclassif=False
        self.y_calibrated_new=None
        
    def fit(self):
        
        if self.model_name=='sm.OLS':
             print(self.model_name)
             self.model=sm.OLS(self.endog,self.exog).fit()
        elif self.model_name=='sm.GLM':
             print(self.model_name)
             if self.family=="Binomial":
                 family=sm.families.Binomial()
                 self.model=sm.GLM(self.endog,self.exog,family=family).fit()
        elif self.model_name=='smd.Logit':
                print(self.model_name)
                self.model=smd.Logit(self.endog,self.exog).fit()
        elif self.model_name=='smd.GenLogit':
                print(self.model_name)
                self.model=smd.GenLogit(self.endog,self.exog).fit()
        elif self.model_name=='smd.MNLogit':
                print(self.model_name)
                self.model=smd.GenLogit(self.endog,self.exog).fit()
       
        elif self.model_name=='smd.Probit':
                print(self.model_name)
                self.model=smd.Probit(self.endog,self.exog).fit()
        else:
            print("Error in loop")
        return self.model
     
    def fit_regularized(self):
        
        if  self.method=='l1' and self.model_name=='smd.Logit':
                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
            
                self.model=smd.Logit(self.endog,self.exog).fit_regularized(method='l1',alpha=self.par_reg[0])
        if  self.method=='l1' and self.model_name=='smd.GenLogit':
                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                
                self.model=smd.GenLogit(self.endog,self.exog).fit_regularized(method='l1',alpha=self.par_reg[0])
        if  self.method=='l1' and self.model_name=='smd.Probit':
                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
        if  self.method=='l1' and self.model_name=='smd.MNLogit':
                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))        
                self.model=smd.MNLogit(self.endog,self.exog).fit_regularized(method='l1',alpha=self.par_reg[0])
        if self.method=='elastic_net' and self.model_name=='sm.OLS':
                print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                self.model=sm.OLS(self.endog,self.exog).fit_regularized(method='elastic_net',alpha=self.par_reg[0],L1wt=self.par_reg[1]) 
        if self.method=='elastic_net' and self.model_name=='sm.GLM':
            if self.family=="Binomial":
                 print(self.model_name + " Regularized wwith alpha: %.2f L1wt: %.2f "%(self.par_reg))
                 family=sm.families.Binomial()
                 self.model=sm.GLM(self.endog,self.exog,family=family).fit_regularized(method='elastic_net',alpha=self.par_reg[0],L1wt=self.par_reg[1])     
        return self.model

    def summary_models(self):
        self.z_score=self.model.params[1:]/self.model.bse[1:]
        return self.model_name,self.model.summary(),self.model.params.values,\
        self.model.resid_pearson,self.model.fittedvalues,self.model.bse,self.z_score.values
    
    def summary_LS_models(self):
        from statsmodels.sandbox.regression.predstd import wls_prediction_std
        if self.model_name=='sm.OLS':
            self.prstd, self.iv_l, self.iv_u = wls_prediction_std(self.model)
        if self.model_name=='sm.GLM':
            self.model.ssr=self.model.pearson_chi2
        return self.model.ssr,self.prstd,self.iv_l,self.iv_u
    
    def calibrate(self):
        self.y_calibrated=self.model.predict(self.exog)
        return self.y_calibrated
    
    def predict(self):
        if self.model_name=='sm.GLM':
            self.y_estimated = self.model.predict(self.x,linear=self.linear)
        else:
            self.y_estimated=self.model.predict(self.x)
        return self.y_estimated

    def confusion_matrix(self):

        if (self.misclassif==True) :
            self.y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),self.y_calibrated))
            self.y=np.array(self.endog)
        else:
            self.y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),self.y_estimated))
            self.y=np.array(self.y)

        self.y_estimated_=np.array(self.y_estimated_)
        
        msa=ac.metrics_binary(self.y,self.y_estimated_,self.misclassif)
        if (self.misclassif==True):
            self.msb=msa
        else:  
            self.msb=msa[:7]
        return self.msb
    
    def roc_curve(self):
        
        fpr,tpr,_=metrics.roc_curve(self.y,self.y_estimated)
        self.fpr=fpr
        self.tpr=tpr
        return self.fpr,self.tpr


class Statsmodels_linear_filter(Statsmodels_simula):
    """ Simulation with the original data ."""
    
    def __init__(self,names,exog,endog,x,y,family,method,par_reg,task,mis_endog):
            self.names=names
            self.exog=exog
            self.model=None
            self.endog=endog
            self.x=x
            self.y=y
            self.family=family
            self.method=method
            self.par_reg=par_reg #list with two parameters
            self.task=task
            self.mis_endog=mis_endog
            self.misclassif=False
        
            if len(self.mis_endog)>0:
                 self.misclassif=True
                 
    def  statsmodels_linear_supervised(self):
        """Linear Methods to Supervised prediction."""
        
      
        to_model_names=[]
        to_params=[]
        to_residuals=OrderedDict()
        confusion_matrix=[]

        to_fitted_values=OrderedDict()
        to_iv_l=OrderedDict()
        to_iv_u=OrderedDict()
        to_z_score=[]
        to_y_calibrated=OrderedDict()
        to_y_estimated=OrderedDict()
        to_fpr=OrderedDict()
        to_tpr=OrderedDict()
        to_fp_index=OrderedDict()
        to_fn_index=OrderedDict()
        to_mis_endog=OrderedDict()
        #names= ['OLS','GLM','Logit','GenLogit','Probit'] #
        smd_models=['Logit','GenLogit','Probit','MNLogit']
        sm_models=['OLS','GLM']
        for name in self.names:  
            self.name=name
            if self.name in smd_models:
                self.model_name='smd.'+ str(self.name)
            elif self.name in sm_models:
                self.model_name='sm.'+ str(self.name) 
                self.linear=False
            else:
                print("Error in model_name selection")
                
            if self.name=='OLS' or 'GLM':
                self.method='elastic_net'
            else:
                self.method ='l1'
            self.columns=self.exog.columns
            
            if (len(self.par_reg)==0):
                self.model=Statsmodels_simula.fit(self)
            else:
                self.model=Statsmodels_simula.fit_regularized(self)
                
            model_name,summary,params,resid_pearson,fitted_values,\
            bse,z_score=Statsmodels_simula.summary_models(self)
            
            if (self.task=="LinearRegression"):
                ssr,prstd,iv_l,iv_u=Statsmodels_simula.summary_LS_models(self)
                to_iv_l[str(self.model_name)]=iv_l
                to_iv_u[str(self.model_name)]=iv_u
            
            to_model_names.append(model_name)
            y_calibrated=Statsmodels_simula.calibrate(self)
            y_estimated=Statsmodels_simula.predict(self)
            
            to_y_calibrated[str(self.model_name)]=y_calibrated
            to_y_estimated[str(self.model_name)]=y_estimated
            to_params.append(params)
            to_residuals[str(self.model_name)]=resid_pearson
            to_fitted_values[str(self.model_name)]=fitted_values
            
            to_z_score.append(z_score)
            
            if (self.task=="BinaryClassification"):
                
                    sc=Statsmodels_simula.confusion_matrix(self)
                    
                    if self.misclassif==True:
                        confusion_matrix.append(sc[:8])
                        to_fp_index[str(self.model_name)]=sc[10]
                        to_fn_index[str(self.model_name)]=sc[11]
                        to_mis_endog[str(self.model_name)]\
                        =Statsmodels_simula.simple_corrected_mis_classification(self)  
                        
                    else:
                        confusion_matrix.append(sc)
                        fpr,tpr=Statsmodels_simula.roc_curve(self) 
                        to_fpr[str(self.model_name)]=fpr
                        to_tpr[str(self.model_name)]=tpr
                
                    
                
        if (self.task=="BinaryClassification"):
            
             if (self.misclassif==True):
                 FP_Index=pd.DataFrame.from_dict(to_fp_index,orient='index')
                 FN_Index=pd.DataFrame.from_dict(to_fn_index,orient='index')
                 LEN=[len(names) for names in to_mis_endog.values()]
                
                 mis_endog_table=pd.DataFrame.from_dict(to_mis_endog,orient='index')
                 
                 mis_endog_table=mis_endog_table.T
                 newindex=range(1,LEN[0]+1)
                 mis_endog_table=pd.DataFrame(np.array(mis_endog_table),index=newindex,\
                                               columns=mis_endog_table.columns)
                 confusion_matrix = pd.DataFrame(confusion_matrix, \
                 index=to_model_names,\
                 #acc,TP,TN,FP,FN,BIAS,pt1,pp1,TNTP,total,fp_index,fn_index
                 columns=['ACC','TP','TN','FP','FN','BIAS','P[X=1]','P[X*=1]'])
             else:
                 
                 to_fpr['AUC=0.5']=fpr
                 to_tpr['AUC=0.5']=fpr
                 FPR=pd.DataFrame.from_dict(to_fpr,orient='index')
                 TPR=pd.DataFrame.from_dict(to_tpr,orient='index')
                 confusion_matrix = pd.DataFrame(confusion_matrix, \
                 index=to_model_names,columns=['ACC','TPR','TNR','PPV','FPR','FNR','DOR'])
                 
        elif (self.task=="LinearRegression"):
            iv_l_table = pd.DataFrame.from_dict(to_iv_l)
            iv_u_table = pd.DataFrame.from_dict(to_iv_u)
        else:
            pass
        to_params=np.array(to_params).reshape(len(to_model_names),len(self.columns)).T
        params_table = pd.DataFrame(to_params,index=self.columns, columns=to_model_names)
        
        residuals_table = pd.DataFrame.from_dict(to_residuals)        

        
        
        fitted_values_table = pd.DataFrame.from_dict(to_fitted_values)
        
        
        y_calibrated_table = pd.DataFrame.from_dict(to_y_calibrated)
        
        y_estimated_table = pd.DataFrame.from_dict(to_y_estimated)
        
    
        to_z_score=np.array(to_z_score).reshape(len(to_model_names),len(self.columns)-1).T
        z_score_table = pd.DataFrame(to_z_score,index=self.columns[1:],columns=to_model_names)
        
        
        if (self.task=="BinaryClassification"):
            if (self.misclassif==True):
                return mis_endog_table,y_calibrated_table,y_estimated_table,FP_Index,FN_Index,confusion_matrix,to_model_names
            else:
                return y_calibrated_table,y_estimated_table,params_table,residuals_table, \
                fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,to_model_names
        else:
            return y_calibrated_table,y_estimated_table,params_table,residuals_table,\
            fitted_values_table,z_score_table, iv_l_table, iv_u_table,to_model_names
   
class Table_results:
    """ Show table results."""
    """
      0:params
      1:residuals
      2:fitted_values
      3:y_calibrated
      4:y_estimated
      5:z_score
      6:confusion_matrix
      7:iv_l
      8:iv_u
      9:vif
     10:acc
     11:ppv
     12:acc_mean
     13:ppv_mean
    """       
    
    def __init__(self,var,table,floatfmt,style,title,width):
        
            self.var=var    
            self.table=table
            self.floatfmt=floatfmt
            self.tablefmt=style
            self.title=title
            self.width=width
            
            
    def print_table(self):
        from tools import Tools
        print_table=["params_table","residuals_table","fitted_values_table","y_calibrated_table", \
                "y_estimated_table","z_score_table","confusion_matrix","iv_l_table","iv_u_table",\
                "vif_table","acc_table","ppv_table","acc_mean_table","ppv_mean_table"]       
        self.table_name=print_table[self.var]
        if self.table_name in print_table:
            return Tools.table(self.table,self.floatfmt,self.tablefmt,self.title,self.width)
        else:
            return print("Error in print_table name")
        outer_name ='print_' + str(self.table_name) 
        # Get the method from 'self'. Default to a lambda.
        outer= getattr(self,outer_name,lambda:"Invalid Method selected")
        # call the strategie as we return it
        return outer()
   
class Drawing2d:
    """ Draw 2d figures"""
    
    def __init__ (self):
            self.parameters=None
            
    def plot_matrix_matrix(X,Y,Title,xlabel,ylabel,Labels,\
                           Linestyle,kind,scale,grid,text,boxstyle):
        """Draw matrix vs matrix."""
        """
        kind of graphic
        0-plot
        1-scatter
        2-stem
        scale of plot
        'equal'
        'Log'
        'semilogy'
        'semilogx'
        'Loglog'
        """
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors
        # colors for grahics with matplolib and plotly
        colors = [ic for ic in mcolors.BASE_COLORS.values() if ic !=(1,1,1)]
        keys   = [kc  for kc in mcolors.BASE_COLORS.keys() if mcolors.BASE_COLORS[kc]!=(1,1,1)]
        
        """
        Checking orders of matrices
        """ 
        if (X.shape[0]!=Y.shape[0] or X.shape[1]!=Y.shape[1]):
                    print("These matrices are not equal order")
        
        
        idx=[X.iloc[:,i].argsort() for i in range(X.shape[1])]
        plt.figure()
        ax=plt.subplot(1,1,1)
        for i,linestyle in enumerate(Linestyle):
            if i in range(X.shape[1]):
               label=Labels[i]
               
               if kind==0 :
                   if scale=='equal' or 'Log':
                       ax.plot(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=1.5,label=label,color=colors[i])
                   elif scale=='semilogy':
                       ax.semilogy(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=1.5,label=label,color=colors[i])
                   elif scale =='semilogx':
                       ax.semilogx(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=1.5,label=label,color=colors[i])
                   elif scale =='Loglog':
                       ax.loglog(X.iloc[idx[i],i],Y.iloc[idx[i],i],linestyle=linestyle,linewidth=1.5,label=label,color=colors[i])
                   else:
                       pass
               elif kind==1:
                   ax.scatter(X.iloc[idx[i],i],Y.iloc[idx[i],i],marker=linestyle,label=label,color=colors[i])
               elif kind==2:
                   markerline, stemlines, baseline = ax.stem(X.iloc[idx[i],i],Y.iloc[idx[i],i],'-.',\
                                                             markerfmt=keys[i]+'o',label=label)
                   #ax.setp(baseline, color='r', linewidth=1.5)
               else:
                   pass
            else:
               pass
            
        ax.legend(prop={'size':10})
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(Title)
        plt.tight_layout(rect=[0, 0, 1, 1])    
        plt.grid(grid)
        plt.grid(color=colors[i+1],linestyle='',linewidth='1')
        if len(text)>0:
              plt.text(0.1, 85.0,text,\
                     {'color': 'k', 'fontsize':10, 'ha': 'left', 'va': 'center',\
                      'bbox': dict(boxstyle=str(boxstyle), fc="w", ec="k", pad=0.3)})

        return plt.show()
    def draw_vector(v0, v1,x0,y0, ax=None):
        """Modification to Python Data Science Handbook 
        origin by Jake VanderPlas; Jupyter notebooks"""
        
        ax = ax or plt.gca()
        arrowprops=dict(facecolor='black',arrowstyle='->',
                        linewidth=2,
                        shrinkA=0, shrinkB=0)
        ax.annotate('', xy=(v0, v1),xytext=(x0,y0), arrowprops=arrowprops)
           
class Draw_binary_classification_results:
    """Draw the results."""
    """
     0. Roc_curve
     
    """ 
    def __init__(self,FPR,TPR,model_names,params,exog,endog,x,\
                 y,y_predict,y_estimated,residuals,columns,Title,kind):
        self.fpr=FPR
        self.tpr=TPR
        self.model_names_selected=model_names
        self.params=params
        self.x_train=exog
        self.y_train=endog
        self.x_test=x
        self.y_test=y
        self.y_predict=y_predict
        self.y_estimated=y_estimated
        self.residuals=residuals
        self.columns=columns
        self.Title=Title
        self.kind=kind
        
    def roc_curve(self):
        
        X=self.fpr.T
        Y=self.tpr.T
        Title='Roc Curve '
        xlabel='False Positive Rate'
        ylabel='True Positive Rate'
        Labels=self.model_names_selected
        Linestyle=['-','-.','--',':','--']
        return Drawing2d.plot_matrix_matrix(X,Y,Title,xlabel,ylabel,Labels,Linestyle,0,'equal',False,'','')
    
    
    def fpn_estimated(self):
        """ Draw prediction results from binary classification..."""
    
        m,n=self.y_estimated.shape
        X=np.array(np.ones((m,n)))
        X=np.array(X)
        X=pd.DataFrame(X)
        X.loc[:,0]=0
        X=np.cumsum(X)
        y_test=np.array([self.y_test]).T
        Labels=self.y_estimated.columns.values
        #y=np.concatenate([y_truth,y_predict],axis=1) 
        f=lambda x:np.where(x<0.5,0,1)
        y_estimated=[list(map(f,self.y_estimated.iloc[:,i])) for i in range(n)]
        y_estimated=pd.DataFrame(y_estimated).T
        y=y_test-y_estimated
        y=pd.DataFrame(y)
        Y=y
        #Title='Binary Classification using Statsmodels'
        Title=self.Title
        xlabel='samples'
        ylabel='y_truth - y_estimated'
        Linestyle=['-','-.','--',':','--']
        return Drawing2d.plot_matrix_matrix(X,Y,Title,xlabel,ylabel,Labels,Linestyle,2,'equal',False,'','')
    
    
    def draw_regions_binary_classification(self):
        """Draw regions of binary classification."""
        """
        | x--x_test
        | y--y_test
        | y_estimated =model.predict(x_test)
        | columns --all columns including x and y
        | treshold--vector contains treshold to each models
        | c -shape parameter GenLogit
        """
    
        model_names=self.model_names_selected
        
        self.x_test=pd.DataFrame(self.x_test)
        self.y_test=pd.DataFrame(self.y_test)
        self.y_estimated=pd.DataFrame(self.y_estimated)
        
        MM=len(self.model_names_selected)
        
        z_list=[]
        zz=pd.DataFrame(np.concatenate([self.x_test,self.y_test],axis=1),columns=self.columns)
        z_list.append(zz)
        
        for ii in range(MM):  
            y_estimated_col=self.y_estimated.iloc[:,ii]           
            y_estimated_col=np.where(y_estimated_col<0.5,0,1)
            y_estimated_col=np.array(y_estimated_col).reshape(self.x_test.shape[0],1)
            zz=pd.DataFrame(np.concatenate([self.x_test,y_estimated_col],axis=1),columns=self.columns)
            z_list.append(zz)
       
        # Compute the coeficients of straight line as boundary_decision 
        treshold=[]
        for name in model_names:
            if name=='smd.Logit' or 'sm.GLM' or 'MNLogit':
                treshold.append(0.0)
            elif name=='smd.GenLogit':
                c=Tools.get_value_shape()
                treshold.append(0.5**c)
                
            else:
                treshold.append(0.5)
       
        model_names.reverse()
        model_names.append('Test')
        model_names.reverse()
        
        """N-Number of subplots..."""
        N=len(model_names) 
        if N%2==0:
            nrows=int(N/2)
            ncols=2
            figsize=(15,10)
        elif N%3==0:
            nrows=int(N/3)
            ncols=3
            figsize=(16,4)
        elif N%4==0:
            nrows=int(N/4)
            ncols=4
            figsize=(8,4)
        else:
            nrows=int(N)
            ncols=1
            figsize=(10,15)
            # Don't working plt.sub_plots_adjust
            #   wspace=0.5
            #  hspace=0.5
        
        # Draw scatter plots of "glu" vs "ped"
        
        #Title="Binary Classification Regions (Yes/No) using statsmodels"
        Title=self.Title
        fig,axes=plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
        
        
        for ax,df,name in zip(axes.flatten(),z_list,model_names):
            
            for color, dfe,nlabel in zip(['Darkblue','red'],df.groupby(df.columns[-1]),["No","Yes"]):
                   # x=df.loc[Index,df.columns[0]]
                    x=dfe[1][df.columns[0]]
                    y=dfe[1][df.columns[1]]
                    
                    ax.scatter(x=x,y=y,marker='o',c=color, s=70, label=str(nlabel))
                    ax.set_title(name)
                    ax.set_xlabel(df.columns[0])
                    ax.set_ylabel(df.columns[1])
                    ax.legend()
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.suptitle(Title)
        return plt.show()
       
    def draw_mis_classification(self):
        """Draw regions of binary classification."""
        """
        | x--x_test
        | y--y_test
        | y_estimated =model.predict(x_test)
        | columns --all columns including x and y
        | treshold--vector contains treshold to each models
        | c -shape parameter GenLogit
        """
       
        model_names=self.model_names_selected
        
        if self.kind=="Validation":
            y_train=pd.DataFrame(self.y_train)
            y_predict=pd.DataFrame(self.y_predict)
            residuals_table=self.residuals
            NN=len(y_train)
            x_graph=np.linspace(0,NN-1,NN).astype(float)
            
        elif self.kind=="Prediction":
            y_test=pd.DataFrame(self.y_test)
            y_predict=pd.DataFrame(self.y_estimated)
            NN=len(y_test)      
            x_graph=np.linspace(0,NN-1,NN).astype(float)
        else:
            print("Error on kind selection")
            
        z_list=[]
        for name in model_names:
            if name=='Logit'or'GenLogit'or'Probit':
                    sname='smd.'+str(name)
            if self.kind=="Validation": 
                z_ordinates_ii=np.abs(residuals_table[sname])
            elif self.kind=="Prediction":
                z_ordinates_ii=np.abs(self.y_estimated[sname])
            z_predict_ii=y_predict[sname]
            z_predict=np.where(z_predict_ii<0.5,0,1)
            z_predict=pd.DataFrame(z_predict)
            z_res=np.vstack([x_graph,z_ordinates_ii]).T
            if self.kind=="Validation":
                z_ind=y_train-z_predict
                zz=pd.DataFrame(np.concatenate([z_res,z_ind],axis=1),columns=['obs','residue_pearson','type'])  
            elif self.kind=="Prediction":
                z_ind=y_test-z_predict   
                
                zz=pd.DataFrame(np.concatenate([z_res,z_ind],axis=1),columns=['obs','prediction_values','type']) 
            else:
                print("Error on kind selection")
            z_list.append(zz)
       
            
        N=len(model_names)
        nrows=int(N)
        ncols=1
        figsize=(10,5)
        #wspace=0.5
        #hspace=0.5
      
        
        Title=self.Title
        fig,axes=plt.subplots(nrows=nrows,ncols=ncols, figsize=figsize)
        

        for ax,df,name in zip(axes.flatten(),z_list,model_names):
                ind=sorted(df[df.columns[-1]])[0]
                if ind==0:
                    Color=['red','green']
                    LABEL=["TP+TN","FN"]
                elif ind==1:
                    Color=['green']
                    LABEL=["FN"]
                else:
                    Color=['Darkblue','red','green']
                    LABEL=['FP',"TP+TN","FN"]
                    
                for color, dfe,nlabel in zip(Color,df.groupby(df.columns[-1]),LABEL):
            
                   # x=df.loc[Index,df.columns[0]]
                    x=dfe[1][df.columns[0]]
                    y=dfe[1][df.columns[1]]
                    
                    ax.scatter(x=x,y=y,marker='o',c=color, s=40, label=str(nlabel))

                    ax.set_xlim([0,NN])
                    ax.set_ylim([0,3])
                    ax.set_title(name)
                    ax.set_xlabel(df.columns[0])
                    ax.set_ylabel(df.columns[1])
                    
                    ax.legend(loc='best')
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        
                    
        
        plt.suptitle(Title)   
        return plt.show()

class Draw_numerical_results:
    """Draw numerical properties."""
    @staticmethod
    def frame_from_dict(y,xlabel,ylabel,Title,mapping,grid,text,boxstyle):
        """From dict with orient='index'"""
       
        m,n=y.shape
        X=np.array(np.ones((m,n)))
        X=np.array(X)
        X=pd.DataFrame(X)
    
        X.loc[0,:]=0
        X=np.cumsum(X)
        Labels=y.columns.values
        if mapping=='Log':
            Title='Log20 '+Title
            Y=np.log(np.abs(y))/np.log(20)
        else:
            Y=y
        Title=Title
        xlabel=xlabel
        ylabel=ylabel
        Linestyle=['-','-.','--',':','--']
        return Drawing2d.plot_matrix_matrix(X,Y,Title,xlabel,ylabel,Labels,\
                                            Linestyle,0,mapping,grid,text,boxstyle)
             
        

class Simleng_strategies (Data_base):
        """Dispatch Simulation Strategie with statsmodels."""
        """
        0:"full_features"
        1:"features_selection_z_score"
        2: "K_fold_cross_validation"
        3:"pca"
        4:"additional_test"
        5:"discriminant_methods"
        
        """
        def __init__(self,var,idoc):

            self.var=var
            self.idoc=idoc
            self.data_train,self.data_test=\
            Data_base.data_generation()
            self.data_dummy_train,self.data_dummy_test=\
            Data_base.data_generation_binary_classification(train\
            =self.data_train.values(),\
            test=self.data_test.values())       
 
        def strategies(self):
            
            strategies=["full_features","features_selection_z_score",\
                        "K_fold_cross_validation",\
                        "pca","additional_test","discriminant_methods"]
                    
            argument= strategies[self.var]
            method_name ='simulation_' + str(argument) 
            # Get the strategie from 'self'. Default to a lambda.
            strategie= getattr(self,method_name,lambda:"Invalid Method selected")
            # call the strategie as we return it
            return strategie()
    
    
        def simulation_full_features(self):
            
            if (self.idoc==0):
                    """ To create header of Report..."""
                    pass
            """Working will full data to proof."""
            
            columns_train,X_train,Y_train,df=\
            Tools.data_list_unpack_dict(self.data_train)
            
            columns_test,X_test,Y_test,de=\
            Tools.data_list_unpack_dict(self.data_test)
                   
            U_train,V_train=\
            Tools.data_list_unpack_dict(self.data_dummy_train)
            
            U_test,V_test=\
            Tools.data_list_unpack_dict(self.data_dummy_test)
        
                   
            # addition a constant to compute intercept to apply \
            # statsmodels fitting
            

            U_train_exog=Tools.data_add_constant(U_train)
            U_test_exog=Tools.data_add_constant(U_test)
            
            Data_base.data_head_tail(df)
                      
            Data_base.data_feature_show(df)
            
            Data_base.data_features_show(df)
            
            Data_base.data_describe(df)
        
            Data_base.data_features_draw_hist(U_train,10)
            

            Correlation(U_train,df.columns,0.9).correlation_training()
            
            Correlation(U_train,U_train.columns,0.9).correlation_level()
            
            Best_features_filter(U_train,U_train.columns,10).variance_influence_factors()
            
            names=['GLM','Logit','GenLogit','Probit'] 
            exog=U_train_exog
            endog=V_train
            x=U_test_exog
            y=V_test
            family='Binomial'
            method=''
            par_reg=[]
            task="BinaryClassification"
            mis_classif=''
            y_calibrated_table,y_estimated_table,params_table,residuals_table,\
            fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
            to_model_names=Statsmodels_linear_filter(names,exog,endog,x,y,family,method,par_reg,task,mis_classif).\
            statsmodels_linear_supervised()
            
            Table_results(0,params_table,'.3f','fancy_grid','Models Parameters',60).print_table()
        
            Draw_numerical_results.frame_from_dict(residuals_table,\
            'samples','Residue Pearson',\
            'Residuals of solvers on Training Data','Log',False,'','square') 
            
            Table_results(6,confusion_matrix,'.2f','fancy_grid','Confusion Matrix ',60).print_table()
            
            #FPR,TPR,model_names,params,x_test,y_test,y_predict,columns
            Title='Binary Classification using Statsmodels'
            Draw_binary_classification_results(FPR,TPR,to_model_names,params_table,\
                                U_train_exog,V_train,U_test,V_test,y_calibrated_table, \
                                y_estimated_table, residuals_table,\
                                columns_train,Title,'',).fpn_estimated()

            #Draw_binary_classification_results(FPR,TPR,to_model_names).\
            #fpn_estimated(V_test,y_estimated_table)
            
            to_model_names.append('AUC=0.5')
            Title=" ROC_CURVE "
            Draw_binary_classification_results(FPR,TPR,to_model_names,params_table,\
                            U_train_exog,V_train,U_test,V_test,y_calibrated_table,y_estimated_table,\
                           residuals_table,columns_train,Title,'').roc_curve()
            
            
            
            
        def simulation_features_selection_z_score(self):

            """Working with diferents criterie of features selection."""
        
            
            columns_train,X_train,Y_train,df=\
            Tools.data_list_unpack_dict(self.data_train)
            
            columns_test,X_test,Y_test,de=\
            Tools.data_list_unpack_dict(self.data_test)
                   
            U_train,V_train=\
            Tools.data_list_unpack_dict(self.data_dummy_train)
            
            U_test,V_test=\
            Tools.data_list_unpack_dict(self.data_dummy_test)
         
            # Addition a constant to compute intercept to apply \
            #    regressoin models
            
            U_train_exog=Tools.data_add_constant(U_train)
            U_test_exog=Tools.data_add_constant(U_test)
            
            # Get two predictors based on p_values using scipy.ks_2samp
            z_score={}
            z_score_table={}
            columns=df.columns[:-1]
            
            for ii in range(len(columns)):
                for jj in range(len(columns)):
                   if ii < jj:
                        columns_base=[]
                        columns_base.append(columns[ii])
                        columns_base.append(columns[jj])
                        exog=U_train_exog[columns_base]                       
                        D_stats,p_value=stats.ks_2samp(exog[columns_base[0]],\
                                                      exog[columns_base[1]])                                                                     
                        z_score[p_value]=columns_base
                        
            z_score_table_keys=sorted(z_score.keys(),reverse=False)
            columns_two_table=[]        
            for ii in z_score_table_keys:
                z_score_table[ii]=z_score[ii]
                columns_two_table.append(z_score[ii])
            z_score_table=pd.DataFrame.from_dict(z_score_table).T
            # Show the z_score of all predictors
            Table_results(5,z_score_table,'.3E','fancy_grid','p_values vs predictors',15).print_table()
            
            # Get the best two-predictors based on the p-values
            names=['GLM','Logit','GenLogit','Probit'] 
            endog=V_train
            y=V_test
            family='Binomial'
            method=''
            par_reg=[]
            task="BinaryClassification"
            mis_classif=''
            ACC_TWO=[]
            NN=0
            for name in columns_two_table:
                        NN+=1
                        columns_base=['const']
                        columns_base.append(name[0])
                        columns_base.append(name[1])
                        exog=U_train_exog[columns_base]
                        x=U_test_exog[columns_base]
            
                        y_calibrated_table,y_estimated_table,params_table,residuals_table, \
                        fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
                        to_model_names=Statsmodels_linear_filter(names,exog,endog,x,y,family,method,par_reg,task,mis_classif).\
                        statsmodels_linear_supervised()
                        
                        Table_results(5,confusion_matrix,'.2f','fancy_grid','Confusion Matrix :'+\
                                      columns_base[1] +','+ columns_base[2],40).print_table()

                        ACC_TWO.append([columns_base[1],columns_base[2],confusion_matrix['ACC'].mean()])
            
            ACC_TWO=pd.DataFrame(ACC_TWO,index=range(NN),columns=['predictor_1','predictor_2','ACC MEAN'])
            Table_results(12,ACC_TWO,'.2f','fancy_grid','ACC MEAN for two predictors ',30).print_table()
            
       
            
            MAX_ACC_TWO=max(ACC_TWO['ACC MEAN'])
            Index_ACC_TWO=np.where(ACC_TWO['ACC MEAN']==MAX_ACC_TWO)
            
           
            
            columns_two=[ ]
            for ii in Index_ACC_TWO[0]:
                 name1=ACC_TWO.loc[ii,ACC_TWO.columns[0]]
                 name2=ACC_TWO.loc[ii,ACC_TWO.columns[1]]
                 columns_two.append(name1)
                 columns_two.append(name2)
             
            columns_Index=['const']
            for name in columns_two:
               columns_Index.append(str(name))    
                
            columns_Index_plus=pd.Index(columns_Index)
              
            names=['GLM','Logit','GenLogit','Probit'] 
            endog=V_train
            y=V_test
            family='Binomial'
            method=''
            par_reg=[]
            task="BinaryClassification"   
            exog=U_train_exog[columns_Index_plus]
            x=U_test_exog[columns_Index_plus]
            mis_classif=''
            y_calibrated_table,y_estimated_table,params_table,residuals_table, \
            fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
            to_model_names=Statsmodels_linear_filter(names,exog,endog,x,y,family,method,par_reg,task,mis_classif).\
            statsmodels_linear_supervised()    
            
            Table_results(5,confusion_matrix,'.2f','fancy_grid','Confusion Matrix :'+\
                                     columns_Index_plus[1] +','+ columns_Index_plus[2],40).print_table()
            
            # Draw_binary_classification_results of Best Predictors
            # based on p_values
            x=U_test_exog[columns_two]
            
            columns=columns_two.copy()
            columns.append(df.columns[-1])
            columns_two_copy=columns_two.copy()
        
            Title="Binary Classification Regions (Yes/No) using statsmodels"\
               "over glu-ped plane"
            Draw_binary_classification_results(FPR,TPR,to_model_names,params_table,\
                                U_train_exog,V_train,x,y,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns,Title,'').draw_regions_binary_classification()
            
            
            print("INCREASING ADDING FEATURES AFTER Z-SCORE ANALYSIS")
            
            # Increase Features Method by add-features after Z-score selection
            names=['Logit','GenLogit']
            method=''
            par_reg=[]
            cols_data=df.columns[:-1] 
            mis_classif=''
            
            Iperfor,model_name_selected,columns_base,y_calibrated_table,\
            y_estimated_table,pararesims_table,duals_table=\
            Best_features_wrap().\
            add_feature_selection(names,cols_data,columns_two_copy,\
                              columns_Index_plus,U_train_exog,\
                              U_test_exog,V_train,V_test,'Binomial',method,\
                              par_reg,'BinaryClassification',90,1,mis_classif)

            
            
            # Show mis-classification for the INCREASED BEST FEATURES SELECTION
            params=params_table.T
            kind="Validation"
            Title="Binary Classification using statsmodels"\
                " (Validation Case)"
            
            Draw_binary_classification_results(FPR,TPR,names,params,\
                                U_train_exog,V_train,\
                 U_test_exog,V_test,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns_base,Title,kind).draw_mis_classification()
           
            kind="Prediction"
            Title="Binary Classification using statsmodels"\
                " (Prediction Case)"
            Draw_binary_classification_results(FPR,TPR,names,params,\
                                U_train_exog,V_train,\
                 U_test_exog,V_test,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns_base,Title,kind).draw_mis_classification()         
           
        
        def simulation_K_fold_cross_validation(self): 
            print("K_fold CROSS-VALIDATION PROCESS WITH FITTING AND PREDICT")
            """
            | Best selection of features based on K_fold Cross-Validation    
            | K_fold Cross-Validation + Add-features with Best_Features Selection
            | Fitting with K_Fold splitting 
            """
            columns_train,X_train,Y_train,df=\
            Tools.data_list_unpack_dict(self.data_train)
            
            columns_test,X_test,Y_test,de=\
            Tools.data_list_unpack_dict(self.data_test)
                   
            U_train,V_train=\
            Tools.data_list_unpack_dict(self.data_dummy_train)
            
            U_test,V_test=\
            Tools.data_list_unpack_dict(self.data_dummy_test)
                   
            # Addition a constant to compute intercept to apply \
            # regressoin models
            
            U_train_exog=Tools.data_add_constant(U_train)
            U_test_exog=Tools.data_add_constant(U_test)

            # Fitting to wrap the best predictors
            names=['GLM','Logit','GenLogit','Probit'] 
            exog=U_train_exog
            endog=V_train
            x=U_test_exog
            y=V_test
            family='Binomial'
            method=''
            par_reg=[]
            task="BinaryClassification"
            mis_classif=''
            y_calibrated_table,y_estimated_table,params_table,residuals_table, \
            fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
            to_model_names=Statsmodels_linear_filter(names,exog,endog,x,y,family,method,par_reg,task,mis_classif).\
            statsmodels_linear_supervised()
            
            Table_results(5,z_score_table,'.3g','fancy_grid','Z-score = params/bse',30).print_table()
            
            # Columns_two contain the best predictors 
            columns_two,columns_Index=Best_features_wrap().z_score(z_score_table,1.96)
            
            # Draw_binary_classification_results of Best Predictors
            # based on Z-score
            

            K_fold=10
            names=['Logit','GenLogit']
            method=''
            par_reg=[]
            cols_data=df.columns[:-1] 
            N_cols_base=len(columns_two)
            params =params_table.T
            columns_Index_plus=["const"]
            for name in columns_Index:
                columns_Index_plus.append(name)
            params=params[columns_Index_plus]
            params=params.T
            x=U_test[columns_Index]
            y=V_test
            
            columns=columns_two.copy()
            columns.append(df.columns[-1])
            columns_two_two=columns_two.copy()
            mis_classif=''
                   
            K_fold,N_features,data_fold,Iperformance=\
            Best_features_wrap().cross_validation_binary_classification(names,cols_data,\
                              columns_two_two,columns_Index_plus,U_train_exog,\
                              U_test_exog,endog,y,family,method,par_reg,task,90,K_fold,mis_classif)

            # Draw ACC and PPV from K_fold Cross-Validation fitting numerical results
            Best_features_wrap().draw_K_fold_numerical_results(K_fold,N_cols_base,N_features,data_fold,Iperformance)
            
            # Generation of numerical results to draw mis-classification with K_fold splitting
            Iperfor,model_name_selected,columns_base,x_train_fold_exog,\
            y_train_fold, x_test_fold_exog,y_test_fold, y_calibrated_table,\
            y_estimated_table,params_table,residuals_table=\
            Best_features_wrap().K_fold_numerical_results\
            (K_fold,N_cols_base,N_features,data_fold,Iperformance,90)
            
            # Checking that the best reults are shown
            # Relation between Residues and misclassification in calibration stage
            params=params_table.T
            
            kind="Validation"
            Title="K_fold Cross-Validation in Binary Classification using statsmodels"\
                " (Validation Case)"
            
            Draw_binary_classification_results('','',names,params,\
                                x_train_fold_exog,y_train_fold,\
                 x_test_fold_exog,y_test_fold,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns_base,Title,kind).draw_mis_classification()
           
            kind="Prediction"
            Title="K_fold Cross-Validation in Binary Classification using statsmodels"\
                " (Prediction Case)"
                      
            Draw_binary_classification_results('','',names,params,\
                                x_train_fold_exog,y_train_fold,x_test_fold_exog,\
                                y_test_fold,y_calibrated_table,y_estimated_table,\
                    residuals_table,columns_base,Title,kind).draw_mis_classification()
           
        def simulation_pca(self):
            """Working to explore a spectral mehthod"""
            
            columns_train,X_train,Y_train,df=\
            Tools.data_list_unpack_dict(self.data_train)
            
            columns_test,X_test,Y_test,de=\
            Tools.data_list_unpack_dict(self.data_test)
                   
            U_train,V_train=\
            Tools.data_list_unpack_dict(self.data_dummy_train)
            
            U_test,V_test=\
            Tools.data_list_unpack_dict(self.data_dummy_test)
         
            PCA(U_train,n_components=7).pca()
            PCA(U_train,n_components=7).pca_draw_major_minor_factors(U_train)
            PCA(U_train,n_components=7).pca_show_table()
            PCA(U_train,n_components=7).pca_draw_by_components()
            
            cols=[0,1,4,5,6]
            columns=[]
            for ii in cols:
                columns.append(df.columns[ii])
        
            add=[3,4,5]
            for ii in range(len(add)):
                columns_base=columns[:add[ii]]
                N_base=len(columns_base)
                U_train_reduced=U_train[columns_base]
                U_test_reduced=U_test[columns_base]
                                     
                U_train_pca=PCA(U_train_reduced,n_components=N_base).pca_transformation()
                U_test_pca=PCA(U_test_reduced,n_components=N_base).pca_transformation()
                U_train_pca_exog=Tools.data_add_constant(U_train_pca)
                U_test_pca_exog=Tools.data_add_constant(U_test_pca)
                
                # GET NEW PREDICTION WITH X-TRANSFORMED BY PCA 
                names=['Logit','GenLogit'] 
                endog=V_train
                y=V_test
                family='Binomial'
                method=''
                par_reg=[]
                task="BinaryClassification"  
            
                exog=U_train_pca_exog 
            
                x=U_test_pca_exog
            
                mis_classif=''
                # Observation with resampling based on perturbation
                y_calibrated_table,y_estimated_table,params_table,residuals_table, \
                fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
                to_model_names=Statsmodels_linear_filter(names,exog,endog,x,y,family,method,par_reg,task,mis_classif).\
                statsmodels_linear_supervised()    
               
                Title='Confusion Matrix (PCA: var. in transformed space)'+ ' with ' + str(N_base)  + ' predictors :'+ columns_base[0]
                for name in columns_base[1:]:
                    Title += ','+ name
                  
                Table_results(6,confusion_matrix,'.2f','fancy_grid',Title,60).print_table()
        
        def simulation_additional_test(self):
                """
                TEST ADITIONAL OF features: npreg,glu,bmi,ped without PCA
                """
                columns_train,X_train,Y_train,df=\
                Tools.data_list_unpack_dict(self.data_train)
            
                columns_test,X_test,Y_test,de=\
                Tools.data_list_unpack_dict(self.data_test)
                       
                U_train,V_train=\
                Tools.data_list_unpack_dict(self.data_dummy_train)
            
                U_test,V_test=\
                Tools.data_list_unpack_dict(self.data_dummy_test)
            
                cols=[0,1,4,5]
                columns=[]
                for ii in cols:
                  columns.append(df.columns[ii])
                
                columns_base=columns.copy()
                N_base=len(columns_base)
                U_train_test_exog=Tools.data_add_constant(U_train[columns_base])
                U_test_test_exog=Tools.data_add_constant(U_test[columns_base])  
                  
                names=['Logit','GenLogit'] 
                endog=V_train
                y=V_test
                family='Binomial'
                method=''
                par_reg=[]
                task="BinaryClassification"  
              
                exog=U_train_test_exog 
            
                x=U_test_test_exog
            
                mis_classif=''
                # Observation with resampling based on perturbation
                y_calibrated_table,y_estimated_table,params_table,residuals_table, \
                fitted_values_table,z_score_table,FPR,TPR,confusion_matrix,\
                to_model_names=Statsmodels_linear_filter(names,exog,endog,x,y,family,method,par_reg,task,mis_classif).\
                statsmodels_linear_supervised()    
               
                Title='Confusion Matrix (original var.)'+ ' with ' + str(N_base)  + ' predictors :'+ columns_base[0]
                for name in columns_base[1:]:
                    Title += ','+ name
                  
                Table_results(6,confusion_matrix,'.2f','fancy_grid',Title,60).print_table()
            
            
        def simulation_discriminant_methods(self):

            columns_train,X_train,Y_train,df=\
            Tools.data_list_unpack_dict(self.data_train)
            
            columns_test,X_test,Y_test,de=\
            Tools.data_list_unpack_dict(self.data_test)
                   
            U_train,V_train=\
            Tools.data_list_unpack_dict(self.data_dummy_train)
            
            U_test,V_test=\
            Tools.data_list_unpack_dict(self.data_dummy_test)
            
            # Manual selection of the best results in previous test
            colsA=[0,1,4]
            colsB=[0,1,4,5]
            cols_list=[colsA,colsB]
            for cols in cols_list:
                
                confusion_matrix=[]
            
                columns_base=df.columns[cols]
                #print(columns_base)
                N_base=len(columns_base)
                U_train_reduced=U_train[columns_base]
                U_test_reduced=U_test[columns_base]
                lda = LDA(solver="svd", store_covariance=True)
                qda = QDA(store_covariance=True)
                to_model_names=['LDA','QDA']
                #ORIGINAL SPACE
                print("ORIGINAL VARIABLES")
                y_calibrated = lda.fit(U_train_reduced, V_train).predict(U_train_reduced)
                y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),y_calibrated))
                y_t=np.array(V_train)
                y_est=np.array(V_test)
                acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1,_,_=ac.metrics_binary(y_t,y_estimated_,False)
                print('LDA_CALIBRATED')
                #print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
                y_estimated  = lda.fit(U_test_reduced, V_test).predict(U_test_reduced)
                y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),y_estimated))
                acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1,_,_=ac.metrics_binary(y_est,y_estimated_,False)
                print('LDA_ESTIMATED')
                #print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
                confusion_matrix.append([acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1])
                # Quadratic Discriminant Analysis
            
                y_calibrated = qda.fit(U_train_reduced, V_train).predict(U_train_reduced)
                y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),y_calibrated))
                acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1,_,_=ac.metrics_binary(y_t,y_estimated_,False)
                print('QDA_CALIBRATED')
                #print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
                y_estimated  = qda.fit(U_test_reduced, V_test).predict(U_test_reduced)
                y_estimated_=list(map(lambda x:np.where(x<0.5,0,1),y_estimated))
            
                acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1,_,_=ac.metrics_binary(y_est,y_estimated_,False)
                print('QDA_ESTIMATED')
                #print(acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1)
                confusion_matrix.append([acc,tpr,tnr,ppv,fpr,fnr,dor,BIAS,pt1,pp1])
                #print("===========================================")
                confusion_matrix = pd.DataFrame(confusion_matrix, \
                 index=to_model_names,columns=['ACC','TPR','TNR','PPV','FPR','FNR','DOR','BIAS','P[X=1]','P[X*=1]'])
                Title='Confusion Matrix (original var.)'+ ' with ' + str(N_base)  + ' predictors :'+ columns_base[0]
                for name in columns_base[1:]:
                    Title += ','+ name
                  
                Table_results(6,confusion_matrix,'.2f','fancy_grid',Title,60).print_table()
            
           
            
 