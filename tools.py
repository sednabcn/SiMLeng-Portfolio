#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:03:20 2018

@author: sedna
"""
 
class Tools:  
    def data_unpack_kwargs(**kwargs):
        # data is a list of keys and values
        # kind='list' to list
        #kind='dict' to dict
        keyss=kwargs.keys()
        data={}
        for name in keyss:
            data[name]=kwargs[name]
        return data   
    def data_list_unpack_dict(dict):
            data=[]
            for name in dict.keys():
                data_list=dict[name]
                if len(data_list)==1:
                    data_list=list(dict[name])[0]
                data.extend(data_list)    
            return data
    def data_extract_dict_to_list(kind,**kwargs):
        data=Tools.data_unpack_kwargs(**kwargs)
        data_dict={}                
        for k,name in enumerate(kwargs.keys()):
            if name==kind:
                data_dict[kind]=data[kind]
                data_list=Tools.data_list_unpack_dict(data_dict)
                return data_list
    def add_data(x):
        return x
    def data_0_1_exposure(data):
        """Transformar data to [0,1] by limits"""
        import numpy as np
        f=lambda x:np.where(np.abs(x)>1,1,0)
        return f(data)
    def data_add_exposure(data,loc,var):
        """Add perturbation: u= N(0,1) to X_train...""" 
        from add_exposure import Add_exposure
        import numpy as np
        import pandas as pd
        sadd=Add_exposure().normal_0_1(loc,var,data.shape[0])
        sadd=np.array(sadd,dtype=float).reshape(data.shape[0],1)
        zz=np.zeros((data.shape))
        sadd_table=pd.DataFrame(zz,index=range(1,data.shape[0]+1),\
                                   columns=data.columns) 
        for ii in range(len(sadd_table.columns)):
                sadd_table[sadd_table.columns[ii]]=sadd
        data+=sadd_table
        return data
    
    def add_index_to_list_of_indexes(x,index,kind):
        """Add an index to list of indexes"""
        """
        |x-list of indexes
        | index-to add
        | kind (bool) True:str
        |              False:alphanumeric
        """
        import pandas as pd
        pd_index_list=pd.Index.tolist(x)
        if kind==True:
            index=str(index)
        pd_index_list.append(index)
        pd_index=[str(name) for name in pd_index_list]
        return pd.Index(pd_index)
        
    def diff_series(x,y):
        "Get diff between two series"
        import pandas as pd
        x=pd.Series(x)
        y=pd.Series(y)
        
        index_x=Tools.f_index(x)
        index_y=Tools.g_index(y,index_x)
        index_diff={}
        for name in index_x.keys():
            if name not in index_y.keys():
                index_diff[name]=index_x[name]
        return pd.Series(index_diff)
    def f_index(x):
        """Get index given a prototype..."""
        index={}
        ii=0
        for name in x:
            index[name]=ii
            ii+=1
        return index
    def g_index(y,x):
        index_y={}
        for name in y:
           index_y[name]=x[name]
        return index_y
    def array_sort(x,axis):
        import numpy as np
        x=np.array(x)
        ind=np.argsort(x,axis=axis)
        x=x[ind[0]]
        return x
    def get_max_from_multi_index(x,par,index_col):
        """Get an ordered list from index_colum..."""
        """
        |x-list of multi-index [x1, x2, x3,x4...]"
        |par-divisor of mod
        |index-colum index =0,1,2,....
        """
        import pandas as pd
        
        Idata=pd.DataFrame(index=range(len(x)),\
                           columns=range(2))
        n=0
        for ii in x:
            Idata.loc[n,0],Idata.loc[n,1]=divmod(ii,par)
            n+=1
        LOC=Idata.loc[:,index_col].argsort()
        iloc=LOC.tolist()[-1]
        return Idata.loc[iloc,:]
    def get_pandas_from_groupby(x,y,k):
        import numpy as np
        import pandas as pd
        "k-number of groups"
    
        xy=np.hstack([x,y])
        xy=pd.DataFrame(xy)
        rr=xy.groupby(xy.columns[-1])
        rr=list(rr)
    
        return [np.array(rr[ii][1].iloc[:,0]) for ii in range(k)]
        
    def max_in_col_matrix(X,index_col1,index_col2,max_ref):
         """To identify a location of max in a column of matrix""" 
         """
         | X -pandas DataFrame
         | index_col1,index_col2-index of columns
         | max_ref-level max. of reference
         """
         import numpy as np
         if len(X[index_col1]) > 0:
             imax1=max(X[index_col1])
             sa=X.index[X.loc[:,index_col1]==imax1]
             aa=np.where((X.loc[:,index_col1]==imax1)==True,1,0)
             Na=max(np.cumsum(aa))
         
         if len(X[index_col2])>0:
             imax2=max(X[index_col2])
             sb=X.index[X.loc[:,index_col2]==imax2]
         
         name_list=[]
         if (Na==1):
             name=sa[0]
             name_list.append(name)
             
         else:
             count=0
             for ii in sa:
                 if ii in sb:
                     count+=1
                     name=ii
                     name_list.append(name)
         if len(name_list)==0:
                 name=X.index[sa[-1]] # more number of predictors
                 name_list.append(name)
         max_ref=max(imax1,max_ref)
         return name,name_list,imax1,imax2,max_ref
                            
    def list_plus_list(x,y):
        ladd=[]
        for name in x :
            ladd.append(name)
        ladd+=Tools.list_minus_list(y,ladd)
        return ladd
    
    def list_minus_list(x,y):
        ldiff=[]
        for name in x:
            if name not in y:
                ldiff.append(name)
        return ldiff

    def get_shape_GenLogit():
        c=(input("Shape parameter GenLogit: "))
        return c

    def gensigmoid(X,c):
        f=lambda x,c :(Tools.sigmoid(x))**c
        return f(X,c) 

    def pdf_gensigmoid(X,c):
        import numpy as np
        f=lambda x,c: c*np.exp(-x)/(1+np.exp(-x))**c
        return f(X,c)
    # Mapping training and testing data to [0,1] X->U
    def sigmoid(X):
        import numpy as np
        f=lambda x:1/(1+ np.exp(-x))
        return f(X)

    def pdf_sigmoid(X):
        import numpy as np
        f=lambda x:np.exp(-x)/(1+ np.exp(-x))**2
        return f(X)
    
    
    def data_dummy_binary_classification(y,key,L1,L2):                            
          # Mapping 'type' (Yes/No) to (1/0)
          import numpy as np
          f=lambda x:np.where(x==key,L1,L2)
          V=f(y).astype(int)
          return V
         
    def mapping_zero_one(X):
        import pandas as pd
        """Checking if X matrix is ok!"""
        f=lambda x: (x-x.min())/(x.max()-x.min())
        U=pd.DataFrame(columns=X.columns)
        for ii in X.columns:
            U[ii]=f(X[ii]).astype(float)
        return U
    def gen_col_data(X,n):
        import pandas as pd
        U=pd.DataFrame(index=X.index,columns=range(1))
        for ii in X.columns:
            if X.columns[n]==ii:
                U[:]=X[ii].astype(float)
        return U
    def data_add_constant(x):
             import statsmodels.api as sm
             exog=sm.add_constant(x,prepend=True)
             return exog
    def data_sorted(x,magnitud,key,reverse):
             "Sorted a list,dict,array,tuple.."
             """
             | x : data
             | magnitud :lambda function to apply to x
             | key: lambda function to define order criteria
             | reverse: bool 
             """
             return sorted(enumerate(list(map(magnitud,x))),key=key,reverse=reverse)
   
    def color_text(x,facecolor):
            from termcolor import colored
            return colored(x,facecolor)
    
    def table(x,floatfmt,style,title,width):
            "Print Data Table with tabulate..."
            from tabulate import tabulate
            """
            | x--Data Table 
            | floatfmt--float format number 
            | style="grid","simple","fancy_grid","Latex","html"
            | from tabulate import tabulate
            """
            return print('\n'),print(title.center(width), tabulate(x, headers='keys', tablefmt=style, \
                                  stralign='center', floatfmt=floatfmt),sep='\n'),print('\n\n') 
    
    def solver_eqs_system_2_3(coef_x_y,z_value):
        """Get the solution of equation 05=F(theta*X)"""
        """
        |DataFrame coef_x_y: params= models_params_table contains model.params of each fiiting model
        |z_value: F^(-1)(0.5)
        """
        import numpy as np
        
        #f=lambda alpha,x,y,z: np.where(z!=0,[(alpha-x)/z,-y/z],np.where(y!=0,(alpha-x)/y,0))
        def f(alpha,x,y,z) :
            if np.abs(z)>1e-10:
                return [(alpha-x)/z,-y/z]
            else:
                return [(alpha-x)/y,-1]
            #return np.piecewise(z,[np.abs(z)>1e-10, np.abs(z)< 1e-10 ],[lambda alpha,x,y,z:[(alpha-x)/z,-y/z],\
             #           lambda alpha,x,y,z: [(alpha-x)/y,-1]])
        solution=[]
        for alpha, beta in zip(z_value,coef_x_y.T):
            solution.append(f(alpha,beta[0],beta[1],beta[2]))
        return np.array(solution)        