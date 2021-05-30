#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np, pandas as pd, matplotlib.pyplot as plt
import datetime
from scipy import stats
from math import sqrt
import math,os


# In[3]:


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


# In[4]:


#获取因子CVD的原始数据cashvol
#这里因为没有直接的cashvol，我采用了vwap*volume
#首先读取index 和 colume
tic = data['ticker_names']
dates = pd.to_datetime(data['dates'])
fd_dates = pd.to_datetime(data['fd_dates'])
fd_tick = data['fd_tic']

#读取vwap volume close以及和fama模型相关的市值，BP，ROE，资产回报率
vwap = pd.DataFrame(data['VWAP'],index = tic,columns=dates).T
volume = pd.DataFrame(data['Volume'],index = tic,columns=dates).T
ClosePrice = pd.DataFrame(data['ClosePrice'],index = tic,columns=dates).T
MarketCap = pd.DataFrame(data['CAPQ0_FLOAT_A'],index = tic,columns=dates).T
BPS = pd.DataFrame(data['FIQ0_S_FA_BPS'].T,index = fd_dates,columns=fd_tick)[tic]
TotalAsset = pd.DataFrame(data['BSQ0_TOT_ASSETS'].T,index = fd_dates,columns=fd_tick)[tic]
ROE = pd.DataFrame(data['FIQ0_S_FA_ROE'].T,index = fd_dates,columns=fd_tick)[tic]

#相乘得到cashvol
cashvol = vwap*volume
cashvol.tail()


# In[5]:


#对fama5因子相关数据进行数据处理，获得SMB HML RMW CMA的原始对应数据
TotalAsset = TotalAsset.resample('M').last()
TotalAsset = np.log(TotalAsset)-np.log(TotalAsset.shift(1))
TotalAsset[abs(TotalAsset)==np.inf]=np.nan
ROE = ROE.resample('M').last()
BPS = BPS.resample('M').last()
MC = MarketCap.resample('M').last()


# In[6]:


#对数据进行按月聚合
ClosePrice = ClosePrice.resample('M').last()

#取log return
Re = np.log(ClosePrice.shift(-1)) - np.log(ClosePrice)
Re 


# In[7]:


#获取上证指数月度return
market_index = pd.DataFrame(data['index_ClosePrice'],index=data['index_ticker_names'],columns=pd.to_datetime(data['EPS_dates'])).T['000001']
market_index_return = np.log(market_index.shift(-1))-np.log(market_index)
market_index_return_m = market_index_return.resample('M').sum()


# In[8]:


#获取SMB HML RMW CMA各组收益
smb = []
hml = []
rmw = []
cma = []

ob_d = Re.index[-101:-1]
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


# In[9]:


#原文为向前滚动6个月的daily数据，此处取120天
roll = cashvol.rolling(120)
cvd_raw = roll.std()/roll.mean()

#聚合为月频
cvd = cvd_raw.resample('M').last()
cvd.tail()


# In[10]:


#取最近50个有效观察值
ob_t = Re.index[-51:-1]

def get_observations(Re,MarketCap,factors,ob_t):    
    #dropna and infs
    factor = factors.loc[ob_t,:].copy()
    #处理极端数据
    factor[factor>10]=np.NaN
    factor[factor==0]=np.NaN
    factor.dropna(axis=1,inplace=True)
    Re = Re.loc[ob_t,factor.columns]
    MarketCap = MarketCap.resample('M').last().loc[ob_t,factor.columns]
    return(Re,MarketCap,factor)

Re_latest,MarketCap_latest,cvd_latest = get_observations(Re,MarketCap,cvd,ob_t)
print('start_date ',ob_t[0],'\nend_date ',ob_t[-1])


# In[11]:


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
    
statistics(cvd_latest.values.reshape(1,-1))


# In[12]:


#发现cvd指标类似lognormal分布，取log后再检验统计量
statistics(np.log(cvd_latest.values.reshape(1,-1)))


# In[13]:


#group by cvd monthly and label groups
groups_latest = cvd_latest.apply(lambda x: pd.qcut(x,10,labels=list(range(10))),axis=1)
groups_latest.head()


# In[23]:


#创建分析类
class factor_analysis(object):
    def __init__(self, groups):
        self.groups = groups
    
    #等值加权
    def E_weight(self,ob_t,tt='cvd'):
        ew_re = pd.DataFrame(None,index=ob_t,columns=list(range(10)))
        for i in ob_t:
            ew_re.loc[i,:] = self.Re.loc[i,:].groupby(by=self.groups.loc[i,:]).mean()

        #多空组合
        ew_re['l-h'] = ew_re[0]-ew_re[9]
        
        #cumulative return
        ew_re_cum = ew_re.cumsum()

        #plot
        plt.figure(figsize=(20,24))
        plt.subplot(211)
        for i in ew_re.columns:
            plt.plot(ew_re[i],label='group'+str(i))
        plt.title('equal_weighted_log_return_'+tt)
        plt.legend()

        plt.subplot(212)
        for i in ew_re_cum.columns:
            plt.plot(ew_re_cum[i],label='group'+str(i))
        plt.title('equal_weighted_cumulative_log_return_'+tt)
        plt.legend()
        
        m = np.exp(ew_re_cum.iloc[-1,:].mean())
        def evaluate(df1,rf=0.0384,n=50/12):
            df1 = df1.astype('float')
            
            #CAPM
            X = np.mat(np.stack([np.ones(market_index_return_m[ob_t].shape),market_index_return_m[ob_t]]).T).astype('float')
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
            X = np.mat(np.stack([np.ones(market_index_return_m[ob_t].shape),market_index_return_m[ob_t],hml[ob_t],smb[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            fama3_alpha = beta[0][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_fama3 = beta.T/se
            
            #fama 5-factor
            
            X = np.mat(np.stack([np.ones(market_index_return_m[ob_t].shape),market_index_return_m[ob_t],hml[ob_t],smb[ob_t],rmw[ob_t],cma[ob_t]]).T).astype('float')
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

        for i in ob_t:
            mw_re.loc[i,:] = mw.loc[i,:].groupby(by=self.groups.loc[i,:]).sum()/self.MarketCap.loc[i,:].groupby(by=self.groups.loc[i,:]).sum()

        #多空组合
        mw_re['l-h'] = mw_re[0]-mw_re[9]
        
        #cumulative return
        mw_re_cum = mw_re.cumsum()

        #plot
        plt.figure(figsize=(20,24))
        plt.subplot(211)
        for i in mw_re.columns:
            plt.plot(mw_re[i],label='group'+str(i))
        plt.title('MarketCap_weighted_log_return_'+tt)
        plt.legend()

        plt.subplot(212)
        for i in mw_re_cum.columns:
            plt.plot(mw_re_cum[i],label='group'+str(i))
        plt.title('MarketCap_weighted_cumulative_log_return_'+tt)
        plt.legend()
        
        m = np.exp(mw_re_cum.iloc[-1,:].mean())
        def evaluate(df1,rf=0.0384,n=50/12):
            df1 = df1.astype('float')
            
            #CAPM
            X = np.mat(np.stack([np.ones(market_index_return_m[ob_t].shape),market_index_return_m[ob_t]]).T).astype('float')
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
            X = np.mat(np.stack([np.ones(market_index_return_m[ob_t].shape),market_index_return_m[ob_t],hml[ob_t],smb[ob_t]]).T).astype('float')
            y = np.mat(df1).T
            beta = (X.T*X).I*X.T*y
            fama3_alpha = beta[0][0,0]
            u = y-X*beta
            
            #estimated variation of regression
            varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
            se = np.sqrt(np.diag(varbeta))
            t_value_fama3 = beta.T/se
            
            #fama 5-factor
            
            X = np.mat(np.stack([np.ones(market_index_return_m[ob_t].shape),market_index_return_m[ob_t],hml[ob_t],smb[ob_t],rmw[ob_t],cma[ob_t]]).T).astype('float')
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


# In[24]:


#传入参数
cvd_ana = factor_analysis(groups_latest)
cvd_ana.Re = Re_latest
cvd_ana.MarketCap = MarketCap_latest

#等值加权
cvd_ana.E_weight(ob_t = ob_t,tt='cvd')


# In[25]:


#市值加权
cvd_ana.MC_weight(ob_t = ob_t,tt='cvd')


# In[58]:


#测试不同时期是否具有相同指标，选取再往前50个观察日
new_obt = ClosePrice.index[-101:-51]
Re_test,MarketCap_test,cvd_new = get_observations(Re,MarketCap,cvd,new_obt)
print('start_date ',new_obt[0],'\nend_date ',new_obt[-1])
cma.fillna(method='ffill',inplace = True)

#分组
groups_test = cvd_new.apply(lambda x: pd.qcut(x,10,labels=list(range(10))),axis=1)
groups_test.head()


# In[59]:


#传入参数
cvd_test = factor_analysis(groups_test)
cvd_test.Re = Re_test
cvd_test.MarketCap = MarketCap_test

#等值加权
cvd_test.E_weight(ob_t = new_obt,tt='cvd')


# In[60]:


#市值加权
cvd_test.MC_weight(ob_t = new_obt,tt='cvd')


# In[61]:


#选取不同行业
industry = pd.DataFrame(data['WIND_IDX_INDCLS2_MEMBER'],index = tic,columns=dates).T


# In[62]:


industry.loc[ob_t[0],:].value_counts()


# In[63]:


#我们选取wind行业标码为 882101(材料) 882102(资本货物) 882120（技术硬件与设备)的行业
ind_882101 = industry.loc[ob_t[0]][industry.loc[ob_t[0]]==882101].index
ind_882102 = industry.loc[ob_t[0]][industry.loc[ob_t[0]]==882102].index
ind_882120 = industry.loc[ob_t[0]][industry.loc[ob_t[0]]==882120].index


# In[64]:


def get_observations_ind(Re,MarketCap,factors,ob_t,tics):    
    #dropna and infs
    factor = factors.loc[ob_t,tics].copy()
    #处理极端数据
    factor[factor>10]=np.NaN
    factor[factor==0]=np.NaN
    factor.dropna(axis=1,inplace=True)
    Re = Re.loc[ob_t,factor.columns]
    MarketCap = MarketCap.resample('M').last().loc[ob_t,factor.columns]
    return(Re,MarketCap,factor)

#数据处理
Re_ind_882101,MarketCap_ind_882101,cvd_ind_882101 = get_observations_ind(Re,MarketCap,cvd,ob_t,ind_882101)
Re_ind_882102,MarketCap_ind_882102,cvd_ind_882102 = get_observations_ind(Re,MarketCap,cvd,ob_t,ind_882102)
Re_ind_882120,MarketCap_ind_882120,cvd_ind_882120 = get_observations_ind(Re,MarketCap,cvd,ob_t,ind_882120)
groups_ind_882101 = cvd_ind_882101.apply(lambda x: pd.qcut(x,10,labels=list(range(10))),axis=1)
groups_ind_882102 = cvd_ind_882102.apply(lambda x: pd.qcut(x,10,labels=list(range(10))),axis=1)
groups_ind_882120 = cvd_ind_882120.apply(lambda x: pd.qcut(x,10,labels=list(range(10))),axis=1)
print('start_date ',ob_t[0],'\nend_date ',ob_t[-1])


# In[65]:


#传入数据
ind_882101_test = factor_analysis(groups_ind_882101)
ind_882101_test.Re = Re_ind_882101
ind_882101_test.MarketCap = MarketCap_ind_882101

ind_882102_test = factor_analysis(groups_ind_882102)
ind_882102_test.Re = Re_ind_882102
ind_882102_test.MarketCap = MarketCap_ind_882102

ind_882120_test = factor_analysis(groups_ind_882120)
ind_882120_test.Re = Re_ind_882120
ind_882120_test.MarketCap = MarketCap_ind_882120


# In[66]:


#882101等值加权
ind_882101_test.E_weight(ob_t = ob_t,tt='cvd_ind_882101')


# In[67]:


#882101市值加权
ind_882101_test.MC_weight(ob_t = ob_t,tt='cvd_ind_882101')


# In[68]:


#882102等值加权
ind_882102_test.E_weight(ob_t = ob_t,tt='cvd_ind_882102')


# In[69]:


#882102市值加权
ind_882102_test.MC_weight(ob_t = ob_t,tt='cvd_ind_882102')


# In[70]:


#882101等值加权
ind_882120_test.E_weight(ob_t = ob_t,tt='cvd_ind_882120')


# In[71]:


#882101市值加权
ind_882120_test.MC_weight(ob_t = ob_t,tt='cvd_ind_882120')


# In[72]:


#取最近50个有效观察值
ob_t = Re.index[-51:-1]

def get_observations(Re,MarketCap,factors,ob_t):    
    #dropna and infs
    factor = factors.loc[ob_t,:].copy()
    #处理极端数据
    factor[factor>10]=np.NaN
    factor[factor==0]=np.NaN
    factor.dropna(axis=1,inplace=True)
    Re = Re.loc[ob_t,factor.columns].dropna(axis=1)
    factor = factor.loc[:,Re.columns]
    MarketCap = MarketCap.resample('M').last().loc[ob_t,factor.columns]
    return(Re,MarketCap,factor)

Re_latest,MarketCap_latest,cvd_latest = get_observations(Re,MarketCap,cvd,ob_t)
print('start_date ',ob_t[0],'\nend_date ',ob_t[-1])


# In[73]:


#回归分析的数据处理
#取log，同时对极端值进行处理，进行normalization
#cvd_latest=np.log(cvd_latest)
def normalization(data,ax=1):
    def centrilization(x):
        y=abs(x-x.median()).median()*5
        x[x>(x.median()+y)]=x.median()+y
        x[x<(x.median()-y)]=x.median()-y
        return(x)
    data=data.apply(centrilization,axis=ax)
    data=data.apply(lambda x: (x-x.mean())/x.std(),axis=ax)
    return(data)
cvd_norm = normalization(cvd_latest,ax=0)
Re_norm = normalization(Re_latest,ax=0)
statistics(cvd_norm.iloc[0,:])
statistics(Re_norm.iloc[0,:])


# In[78]:


t_value = []
ob_t = Re.index[-51:-1]
for i in ob_t:
    X = np.mat(np.stack([np.ones(shape=cvd_norm.loc[i,:].shape),cvd_norm.loc[i,:]]).T)
    y = np.mat(Re_norm.loc[i,:].to_numpy()).T
    beta = (X.T*X).I*X.T*y
    u = y-X*beta

    #estimated variation of regression
    varbeta = (u.T*u)[0,0]/(len(y)-len(beta))*(X.T*X).I
    se = np.sqrt(np.diag(varbeta))

    #record estimation
    t_value.append((beta.T/se)[0,1])
t_value = pd.Series(t_value,index = ob_t)
plt.plot(t_value,color='red')
plt.hlines(2,ob_t[0],ob_t[-1],colors='green')
plt.hlines(-2,ob_t[0],ob_t[-1],colors='green')
plt.title('t_value_cvd')
print(pd.Series({'mean':np. mean(t_value),'mean(abs)':np.mean(abs(t_value)),'std':np.std(t_value),'|t|>2':t_value[abs(t_value)>2].count()/t_value.count()}))        

