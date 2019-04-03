
# coding: utf-8

# # Taux de change du Yen Japonnais

# Nous allons effectuer une analyse descriptive du taux de change du Yen depuis 1999. D'abord il faut importer les données. Il faudrat aussi convertir les dates en type 'datetime'. Nous avons 2 colonnes dont Date et FME (Foreign money exchanges)

# In[9]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import statsmodels.tsa.stattools
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("datas.csv", sep=";")
df['Date'] = pd.to_datetime(df['Date'])

print(df.head(7))


# #### Représentation graphique:

# In[2]:


df.plot(x='Date',y='FME',kind='line',figsize=(12,4))


# In[3]:


fme =df['FME']


# #### Description statistique

# In[4]:


print('Taille de l\'échantillon:',len(fme))
print('Moyenne :',round(np.mean(fme),2))
print('Variance :',round(np.var(fme),2))
print('Ecart-type :',round(np.std(fme),2))
print('Médianne:',np.median(fme))
print('Quantile 1/4:',np.percentile(fme,25))
print('Quantile 3/4:',np.percentile(fme,75))
print('Maximum du FME',max(fme))
print('Minimi du FME',min(fme))
print('Etendue:',max(fme)-min(fme))


# On peut utiliser des classes d'étendue 8

# In[5]:


def classes_statistiques(ech,nbc):
    start=min(ech)
    end = (max(ech)-min(ech))/nbc
    i=1
    while(i<=end):
        sum=0
        j=0
        while(j<len(ech)):
            if ech[j]>start and ech[j]<start+end:
                sum +=1
                j+=1
            else:
                j+=1
        print(start,'<= x <=',start+end," = ",sum)
        start+=end
        i+=1        

classes_statistiques(fme,10)


# ### Prévision pour Mars 2019

# Prenons les données antérieurs à mars 2019 et calculons les moyennes et écart-types mobiles sur 30 jours:

# In[6]:


antMars = df[0:5161]

# Indexation des FME par les dates
aMarsI = antMars.set_index(['Date'])

# moyenne mobile
rolmean = aMarsI.rolling(window=30).mean()
print('rolmean: ',rolmean)

# écart type mobile
rolstd = aMarsI.rolling(window=30).std()
print('rolstd: ',rolstd)


# In[7]:


#Représentation graphique des moyenne et écart-type mobiles

orig_point = plt.plot(aMarsI, color='blue', label='Original')
rm_plot = plt.plot(rolmean, color='red', label='moyenne mobile')
rstd_plot = plt.plot(rolstd, color='green', label='écart-type mobile')
plt.legend(loc='best')


# Procédons maintenant au test de Dickey Fuller qui détermine si une série est stationnaire ou non

# In[8]:


print('Test de Dickey Fuller')
aMarsTestDF = adfuller(aMarsI['FME'],autolag='AIC')

#On arrange l'affichage
aMarsOutput = pd.Series(aMarsTestDF[0:4],index=['Test statistic','p-value','décalages utilisés','Nombre d\'observations utilisées'])
for key,value in aMarsTestDF[4].items():
    aMarsOutput['Valeur critique (%s)'%key] = value
print(aMarsOutput)


# La p-value est inférieur à la valeur critique de 5%, on ne rejette donc pas l'hypothèse que c'est une série non-stationnaire.
# Nous pouvons essayer de transformer la série en série stationnaire, pour ce faire, nous allons commencer par transformer les données en logarithme.

# In[9]:


aMars_Log = np.log(aMarsI)
plt.plot(aMars_Log)

movingAverage =  aMars_Log.rolling(window=12).mean()
movingSTD =  aMars_Log.rolling(window=12).std()

plt.plot(aMars_Log, color = 'blue')
plt.plot(movingAverage, color= 'red')


# In[10]:


aMarsTestDF_Log = adfuller(aMars_Log['FME'],autolag='AIC')

aMars_LogOutput = pd.Series(aMarsTestDF_Log[0:4],index=['Test statistic','p-value','décalages utilisés','Nombre d\'observations utilisées'])
for key,value in aMarsTestDF_Log[4].items():
    aMars_LogOutput['Valeur critique (%s)'%key] = value
print(aMars_LogOutput)


# In[12]:


moyPondExp = aMars_Log.ewm(halflife=12, alpha=None, min_periods=0, adjust=True).mean()
plt.plot(aMars_Log)
plt.plot(moyPondExp,color="red")


# In[ ]:


aMars_LogMoinsMPE = aMars_Log - moyPondExp


# In[ ]:


def test_stationarity(timeseries):
    """
    Test stationarity using moving average statistics and Dickey-Fuller test
    Source: https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    """
    
    # Determing rolling statistics
    rolmean = timeseries.rolling(window = 12, center = False).mean()
    rolstd = timeseries.rolling(window = 12, center = False).std()
    
    # Plot rolling statistics:
    orig = plt.plot(timeseries, 
                    color = 'blue', 
                    label = 'Original')
    mean = plt.plot(rolmean, 
                    color = 'red', 
                    label = 'Rolling Mean')
    std = plt.plot(rolstd, 
                   color = 'black', 
                   label = 'Rolling Std')
    plt.legend(loc = 'best')
    plt.title('Moyenne mobile et écart-type mobile')
    plt.xticks(rotation = 45)
    plt.show(block = False)
    plt.close()
    
    # Perform Dickey-Fuller test:
    TestDF = adfuller(timeseries['FME'],autolag='AIC')

    #On arrange l'affichage
    output = pd.Series(TestDF[0:4],index=['Test statistic','p-value','décalages utilisés','Nombre d\'observations utilisées'])
    for key,value in TestDF[4].items():
        output['Valeur critique (%s)'%key] = value
    print(output)


# In[ ]:


test_stationarity(aMars_LogMoinsMPE)


# A présent nos données sont stationnaires.
# On peut donc décomposer les séries (tendance,saison,résidu).

# In[ ]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompo = seasonal_decompose(aMars_Log,freq = 365)

tendance=decompo.trend
saison=decompo.seasonal
residu=decompo.resid

plt.subplot(411)
plt.plot(aMars_Log,label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(tendance)
plt.plot(tendance,label='Tendance',color ='blue')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(saison)
plt.plot(saison,label='Saison',color ='blue')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residu)
plt.plot(residu,label='résidu',color ='blue')
plt.legend(loc='best')

plt.tight_layout()


# Pour préparer la prédiction, nous allons appliquer le modèle ARIMA mais faire la prédiction avec Prophet. Voici donc les graphes d'auto-corrélation et d'auto-corrélation partiels

# ##### ARIMA

# In[17]:


from statsmodels.tsa.stattools import acf,pacf
from statsmodels.tsa.arima_model import ARIMA

aMars_LogShift = aMars_Log - aMars_Log.shift()
plt.plot(aMars_LogShift)


# ACF et PACF

# In[18]:


from pandas import Series
from matplotlib import pyplot

from statsmodels.graphics.tsaplots import plot_acf
series = aMars_Log
plot_acf(series)
pyplot.show()


# In[15]:


from statsmodels.graphics.tsaplots import plot_pacf
series = aMars_Log
plot_pacf(series, lags=50)
pyplot.show()


# Testons avec différents paramètres

# In[41]:


m = ARIMA(aMars_Log,order=(1,0,0))
resAR = m.fit(disp=1)


# In[28]:


m = ARIMA(aMars_Log,order=(0,0,1))
resAR = m.fit(disp=1)

plt.plot(aMars_Log)
plt.plot(resAR.fittedvalues,color ='red')


# In[29]:


m = ARIMA(aMars_Log,order=(0,1,0))
resAR = m.fit(disp=1)

plt.plot(aMars_Log)
plt.plot(resAR.fittedvalues,color ='red')


# #### Prophet

# Essayons maintenant d'avoir une piste de prédiction avec Prophet

# In[4]:


from fbprophet import Prophet


antMars = df[0:5161]
fev = df[5144:5163]

m = Prophet()

antMars.rename(columns={'Date':'ds'}, inplace=True)
antMars.rename(columns={'FME':'y'}, inplace=True)

print('avant mars',antMars)

m.fit(antMars)

future = m.make_future_dataframe(periods=5160)
future.tail()

prevMars = future[5162:5192]

forecast = m.predict(prevMars)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

print(forecast['yhat'])
prevision = forecast['yhat']

mprev=plt.plot(prevMars,prevision)

fev.plot(x='Date',y='FME')


# In[11]:


print(forecast['yhat'])


# Voici les données de prévision du mois de mars et au dessus les données reéls de février, on peut voir que dans les données de prévision varient beaucoup périodiquement, tandis que le cours du mois de février ne varie de manière irrégulière. Une modèle de résaux de neuronne peut aussi être appliqué, ou bien une variation sur la longueur des données à appliquer sur prophet peut aussi donner des prévisions qui peuvent être plus ressemblante a la réalité.
