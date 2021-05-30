#!/usr/bin/env python
# coding: utf-8

# In[460]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import datetime
from scipy import stats
from math import sqrt
import math,os


# In[461]:


#录入path路径下所有的'.npy'结尾的文件
path = 'C:/Users/Administrator/Documents'

#获取所有path下所有文件
dirs = os.listdir(path)
data = {}
#开始录入
for x in dirs:
    #如果以.npy结尾则录入
    if os.path.splitext(x)[1]=='.npy':
        name = os.path.splitext(x)[0]
        file = path+'\\'+x
        value = np.load(file)
        data[name]=value
        
#展示录入的所有文件
data.keys()


# In[462]:


#录入path路径下所有的'.npy'结尾的文件
path = 'C:/Users/Administrator/Documents/COP'

#获取所有path下所有文件
dirs = os.listdir(path)
cop_data = {}
#开始录入
for x in dirs:
    #如果以.npy结尾则录入
    if os.path.splitext(x)[1]=='.xlsx':
        name = os.path.splitext(x)[0]
        file = path+'\\'+x
        value = pd.read_excel(file)
        cop_data[name]=value
        
#展示录入的所有文件
cop_data.keys()


# In[463]:


#原始数据处理
for i in cop_data.keys():
    cop_data[i]['证券代码'] = cop_data[i]['证券代码'].apply(lambda x: x[:6])
    cop_data[i].index = cop_data[i]['证券代码'].values
    cop_data[i].drop(['证券简称','证券代码'],axis=1,inplace=True)
    cop_data[i].columns = list(range(2000,2021))
    cop_data[i] = cop_data[i].T


# In[464]:


#处理得到净值变化
for i in  ['account_payables', 'account_receivables', 'deferred_revenue_current', 'deferred_revenue_noncurrent', 'inventory',
           'other_expense', 'prepaid',]:
    cop_data[i].iloc[:,:] = np.diff(cop_data[i],1,axis=0,prepend=np.nan)
    cop_data[i].fillna(0,inplace=True)
cop_data['R_D_expense'].fillna(0,inplace=True)


# In[465]:


#读取其他数据
tic = data['ticker_names']
dates = pd.to_datetime(data['dates'])
fd_dates = pd.to_datetime(data['fd_dates'])
fd_tick = data['fd_tic']

#统一tic universe
tics = (set(fd_tick)&set(cop_data['operating_income'].columns)&set(tic))

#读取和fama因子模型相关的市值，BP，ROE，资产回报率
ClosePrice = pd.DataFrame(data['ClosePrice'],index = tic,columns=dates).T[tics]
MarketCap = pd.DataFrame(data['CAPQ0_FLOAT_A'],index = tic,columns=dates).T[tics]
BPS = pd.DataFrame(data['FIQ0_S_FA_BPS'].T,index = fd_dates,columns=fd_tick)[tics]
TotalAsset = pd.DataFrame(data['BSQ0_TOT_ASSETS'].T,index = fd_dates,columns=fd_tick)[tics]
ROE = pd.DataFrame(data['FIQ0_S_FA_ROE'].T,index = fd_dates,columns=fd_tick)[tics]


# In[466]:


#对fama5因子相关数据进行数据处理，获得SMB HML RMW CMA的原始对应数据
TotalAsset = TotalAsset.resample('Q').last()
ROE = ROE.resample('Q').last()
BPS = BPS.resample('Q').last()
MC = MarketCap.resample('Q').last()
TotalAsset = TotalAsset.resample('Q').last()

def june_data(df):
    for i in df.index:
        if i.month != 6:
            df.drop(i,axis=0,inplace=True)
    return(df)

TotalAsset = june_data(TotalAsset)
ROE = june_data(ROE)
BPS = june_data(BPS)
MC = june_data(MC)
AR = np.log(TotalAsset)-np.log(TotalAsset.shift(1))
AR[abs(AR)==np.inf]=np.nan


# In[467]:


#获取COP因子值：
non_scaled = (cop_data['operating_income']+cop_data['R_D_expense']-cop_data['account_receivables']-cop_data['inventory']-cop_data['prepaid']+cop_data['deferred_revenue_current']+cop_data['deferred_revenue_noncurrent']+cop_data['account_payables']+cop_data['other_expense'])[tics]
non_scaled.index = TotalAsset.index[9:-1]

#除以年末总资产
COP = non_scaled / TotalAsset.loc[non_scaled.index,tics]
COP[abs(COP)==np.inf] = np.nan

COP.tail()


# In[468]:


#对数据进行中心化和标准化处理，去除极端值，这不改变每期的大小排序
def normalization(data,ax=1):
    def centrilization(x):
        y=abs(x-x.median()).median()*5
        x[x>(x.median()+y)]=x.median()+y
        x[x<(x.median()-y)]=x.median()-y
        return(x)
    data=data.apply(centrilization,axis=ax)
    data=data.apply(lambda x: (x-x.mean())/x.std(),axis=ax)
    return(data)

COP = normalization(COP,ax=0)
COP.tail()


# In[469]:


#描述性统计量(对观察期内全部样本数据)
def statistics(close):
    
    mc = np.mean(close)
    stdc = np.std(close)
    print(pd.Series({'mean':mc,'median':np.median(close),'max':np.max(close),'min':np.min(close),'std':stdc,'skewness':stats.skew(close.T),'kurtosis':stats.kurtosis(close.T)}))
    
    #distribution plot
    x = np.linspace(mc-3*stdc,mc+3*stdc,200)
    y = stats.norm.pdf(x,mc,stdc)
    kde = stats.gaussian_kde(close)
    plt.plot(x,kde(x),label='data')
    plt.plot(x,y,color='black',label='Normal')
    plt.legend()
    plt.show()
    
statistics(np.hstack([COP.loc[k,:].dropna().values.reshape(1,-1) for k in COP.index]))


# In[470]:


#对price取log return
ClosePrice = ClosePrice.resample('Q').last()
ClosePrice = june_data(ClosePrice)
Re = np.log(ClosePrice.shift(-1)) - np.log(ClosePrice)
Re.tail()


# In[471]:


#获取上证指数年度return
market_index = pd.DataFrame(data['index_ClosePrice'],index=data['index_ticker_names'],columns=pd.to_datetime(data['EPS_dates'])).T['000001']
market_index = market_index.resample('Q').last()
market_index = june_data(market_index)
market_index_return = np.log(market_index.shift(-1))-np.log(market_index)
market_index_return.tail()


# In[472]:


#获取SMB HML RMW CMA各组收益
smb = []
hml = []
rmw = []
cma = []

ob_d = COP.index

def fama_groups(df,g=3):
    label = list(range(g))
    fama_sort = pd.qcut(df.dropna(),g,labels=label)

    big = fama_sort[fama_sort==label[-1]].index
    small = fama_sort[fama_sort==label[0]].index
    return(big,small)

for i in ob_d:
    h,l = fama_groups(MC.loc[i,:])
    b,s = fama_groups(BPS.loc[i,:])
    w,r = fama_groups(ROE.loc[i,:])
    c,a = fama_groups(TotalAsset.loc[i,:],2)
    
    hml.append(Re.loc[i,h].mean()-Re.loc[i,l].mean())
    smb.append(Re.loc[i,s].mean()-Re.loc[i,b].mean())
    rmw.append(Re.loc[i,r].mean()-Re.loc[i,w].mean())
    cma.append(Re.loc[i,c].mean()-Re.loc[i,a].mean())
    
hml = pd.Series(hml,index=ob_d)
smb = pd.Series(smb,index=ob_d)
rmw = pd.Series(rmw,index=ob_d)
cma = pd.Series(cma,index=ob_d)


# In[473]:


#去除金融行业的数据 #882115 882116 882117
#获取wind行业分类
ind = pd.DataFrame(data['WIND_IDX_INDCLS2_MEMBER'],index=tic,columns=dates).T[tics]
ind= ind.resample('y').last()
ind


# In[474]:


dropindex = []
for i in ind.columns:
    if (882115 == ind.loc[:,i]).sum()>0 or (882116 == ind.loc[:,i]).sum()>0 or (882117 == ind.loc[:,i]).sum()>0 :
        dropindex.append(i)
len(dropindex)


# In[475]:


drop_finance = set(tics)-set(dropindex)


# In[476]:


#创建分析类
class factor_analysis(object):
    def __init__(self, factors):
        self.factors = factors
    
    #等值加权
    def E_weight(self,ob_t,tt='cvd'):
        ew_re = pd.DataFrame(None,index=ob_t,columns=list(range(10)))
        for i in self.factors.index:
            selected = self.Re.loc[i,:].dropna().index
            groups = pd.qcut(self.factors.loc[i,selected],10,labels=list(range(10)))
            ew_re.loc[i,:] = self.Re.loc[i,groups.index].groupby(by=groups).mean()

        #多空组合
        ew_re['h-l'] = ew_re[9]-ew_re[0]
        
        #cumulative return
        ew_re_cum = ew_re.cumsum()

        #plot
        plt.figure(figsize=(20,24))
        plt.subplot(211)
        for i in range(10):
            plt.plot(ew_re[i],label='group'+str(i))
        plt.plot(ew_re['h-l'],label='group h-l')
        plt.title('equal_weighted_log_return_'+tt)
        plt.legend()

        plt.subplot(212)
        for i in range(10):
            plt.plot(ew_re_cum[i],label='group'+str(i))
        plt.plot(ew_re_cum['h-l'],label='group h-l')
        plt.title('equal_weighted_cumulative_log_return_'+tt)
        plt.legend()
        
        m = np.exp(ew_re_cum.iloc[-1,:].mean())
        def evaluate(df1,rf=0.0384,n=20):
            df1 = df1.astype('float')
            
            #CAPM
            X = np.mat(np.stack([np.ones(market_index_return[ob_t].shape),market_index_return[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            capm_alpha = beta[0][0,0]
            capm_beta = beta[1][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_capm = beta.T/se
            
            #fama 3-factor
            X = np.mat(np.stack([np.ones(market_index_return[ob_t].shape),market_index_return[ob_t],hml[ob_t],smb[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            fama3_alpha = beta[0][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_fama3 = beta.T/se
            
            #fama 5-factor
            
            X = np.mat(np.stack([np.ones(market_index_return[ob_t].shape),market_index_return[ob_t],hml[ob_t],smb[ob_t],rmw[ob_t],cma[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            fama5_alpha = beta[0][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_fama5 = beta.T/se
            
            df = df1.cumsum()
            df = np.exp(df.astype('float'))
            Maxdrawdown = -round(((df-df.expanding().max())/df.expanding().max()).min(),3)
            alpha = (df[-1]-m)/n
            sr = round((df[-1]-1-rf*n)/(df.std()*12**0.5),3)
            return(pd.Series({'TotalReturn':round(df[-1]-1,4),'AnnualReturn':round(df[-1]**(1/n)-1,4),'Sharpe':sr,
                              'CAPM Alpha':[round(capm_alpha,4),round(t_value_capm[0,0],4)],'Beta':[round(capm_beta,4),round(t_value_capm[0,1],4)]
                              ,'fama-3-factor alpha':[round(fama3_alpha,4),round(t_value_fama3[0,0],4)]
                              ,'fama-5-factor alpha':[round(fama5_alpha,4),round(t_value_fama5[0,0],4)],'Maxdrawdown':Maxdrawdown}))
        return(ew_re.apply(evaluate,axis=0).T)
    
    #市值加权
    def MC_weight(self,ob_t,tt='cvd'):
        mw_re = pd.DataFrame(None,index=ob_t,columns=list(range(10)))

        #收益率乘以市值再分组加总后，除以各组市值加总即为各组市值加权平均收益，这一方法在迭代上具有优势
        mw = self.Re*self.MarketCap

        for i in self.factors.index:
            selected = self.Re.loc[i,:].dropna().index
            groups = pd.qcut(self.factors.loc[i,selected],10,labels=list(range(10)))
            mw_re.loc[i,:] = mw.loc[i,groups.index].groupby(by=groups).sum()/self.MarketCap.loc[i,groups.index].groupby(by=groups).sum()

        #多空组合
        mw_re['h-l'] = mw_re[9]-mw_re[0]
        
        #cumulative return
        mw_re_cum = mw_re.cumsum()

        #plot
        plt.figure(figsize=(20,24))
        plt.subplot(211)
        for i in range(10):
            plt.plot(mw_re[i],label='group'+str(i))
        plt.plot(mw_re['h-l'],label='group h-l')
        plt.title('MarketCap_weighted_log_return_'+tt)
        plt.legend()

        plt.subplot(212)
        for i in range(10):
            plt.plot(mw_re_cum[i],label='group'+str(i))
        plt.plot(mw_re_cum['h-l'],label='group h-l')
        plt.title('MarketCap_weighted_cumulative_log_return_'+tt)
        plt.legend()
        
        m = np.exp(mw_re_cum.iloc[-1,:].mean())
        def evaluate(df1,rf=0.0384,n=20):
            df1 = df1.astype('float')
            
            #CAPM
            X = np.mat(np.stack([np.ones(market_index_return[ob_t].shape),market_index_return[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            capm_alpha = beta[0][0,0]
            capm_beta = beta[1][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_capm = beta.T/se
            
            #fama 3-factor
            X = np.mat(np.stack([np.ones(market_index_return[ob_t].shape),market_index_return[ob_t],hml[ob_t],smb[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            fama3_alpha = beta[0][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_fama3 = beta.T/se
            
            #fama 5-factor
            
            X = np.mat(np.stack([np.ones(market_index_return[ob_t].shape),market_index_return[ob_t],hml[ob_t],smb[ob_t],rmw[ob_t],cma[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            fama5_alpha = beta[0][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_fama5 = beta.T/se
            
            df = df1.cumsum()
            df = np.exp(df.astype('float'))
            Maxdrawdown = -round(((df-df.expanding().max())/df.expanding().max()).min(),3)
            alpha = (df[-1]-m)/n
            sr = round((df[-1]-1-rf*n)/(df.std()*12**0.5),3)
            return(pd.Series({'TotalReturn':round(df[-1]-1,4),'AnnualReturn':round(df[-1]**(1/n)-1,4),'Sharpe':sr,
                              'CAPM Alpha':[round(capm_alpha,4),round(t_value_capm[0,0],4)],'Beta':[round(capm_beta,4),round(t_value_capm[0,1],4)]
                              ,'fama-3-factor alpha':[round(fama3_alpha,4),round(t_value_fama3[0,0],4)]
                              ,'fama-5-factor alpha':[round(fama5_alpha,4),round(t_value_fama5[0,0],4)],'Maxdrawdown':Maxdrawdown}))

        return(mw_re.apply(evaluate,axis=0).T)


# In[477]:


#传入参数
cvd_ana = factor_analysis(COP[drop_finance])
cvd_ana.Re = Re[drop_finance]
cvd_ana.MarketCap = MC[drop_finance]

#等值加权
cvd_ana.E_weight(ob_t = COP.index,tt='COP')


# In[478]:


#市值加权
cvd_ana.MC_weight(ob_t = COP.index,tt='COP')


# In[479]:


ind.iloc[10,:].value_counts()


# In[480]:


#我们选取wind行业标码为 882106(耐用消费品与服装) 882114(制药，生命科技与生物科学) 882109（零售业) 行业
ind_882106 = ind.iloc[-1,:][ind.iloc[-1,:]==882106].index & tics
ind_882114 = ind.iloc[-1,:][ind.iloc[-1,:]==882114].index & tics
ind_882109 = ind.iloc[-1,:][ind.iloc[-1,:]==882109].index & tics


# In[481]:


#传入数据
ind_882106_test = factor_analysis(COP.loc[:,ind_882106])
ind_882106_test.Re = Re.loc[:,ind_882106]
ind_882106_test.MarketCap = MC

ind_882114_test = factor_analysis(COP.loc[:,ind_882114])
ind_882114_test.Re = Re.loc[:,ind_882114]
ind_882114_test.MarketCap = MC

ind_882109_test = factor_analysis(COP.loc[:,ind_882109])
ind_882109_test.Re = Re.loc[:,ind_882109]
ind_882109_test.MarketCap = MC


# In[482]:


#882106 等值加权
ind_882106_test.E_weight(ob_t = COP.index,tt='882106_COP')


# In[483]:


#882106 市值加权
ind_882106_test.MC_weight(ob_t = COP.index,tt='882106_COP')


# In[484]:


#882114 等值加权
ind_882114_test.E_weight(ob_t = COP.index,tt='882114_COP')


# In[485]:


#882114 市值加权
ind_882114_test.MC_weight(ob_t = COP.index,tt='882114_COP')


# In[486]:


#882109 等值加权
ind_882109_test.E_weight(ob_t = COP.index,tt='882109_COP')


# In[487]:


#882109 市值加权
ind_882109_test.MC_weight(ob_t = COP.index,tt='882109_COP')


# In[488]:


t_value = []
for i in ob_t:
    y = Re.loc[i,:].dropna()
    x = COP.loc[i,y.index]
    X = np.mat(np.stack([np.ones(shape=x.shape),x]).T)
    y = np.mat(y).T
    beta = (X.T*X).I*X.T*y
    u = y-X*beta

    #estimated variation of regression
    varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
    se = np.sqrt(np.diag(varbeta))

    #record estimation
    t_value.append((beta.T/se)[0,1])
t_value = pd.Series(t_value,index = ob_t)
plt.plot(t_value)
plt.title('t_value_COP')
plt.hlines(2,ob_t[0],ob_t[-1],colors='black')
plt.hlines(-2,ob_t[0],ob_t[-1],colors='black')
print(pd.Series({'mean':np. mean(t_value),'mean(abs)':np.mean(abs(t_value)),'std':np.std(t_value),'|t|>2':t_value[abs(t_value)>2].count()/t_value.count()}))        

