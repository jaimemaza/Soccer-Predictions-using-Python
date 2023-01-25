#!/usr/bin/env python
# coding: utf-8

# In[117]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from math import sqrt
import numpy as np
import pandas as pd
import scipy as sp
import scipy


# In[118]:


FileName = "Football teams.csv"
def readFile(FileName):
    data = pd.read_csv(FileName, header=0)
    return data


# In[119]:


def exploreData(data):
    from sklearn import preprocessing
 
    label_encoder = preprocessing.LabelEncoder()
    data['Team']= label_encoder.fit_transform(data['Team'])
    data['Tournament']= label_encoder.fit_transform(data['Tournament'])
    
    import matplotlib.pyplot as plt
    
    chartFileName = "JaimeMazaFinal.png"
    
    x=data['Goals']
    y=data['Shots pg']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x

    fig = plt.figure(figsize=(20, 30))
    ax1 = fig.add_subplot(7,3,1)
    ax1.plot(x, y, '.', color='limegreen')
    ax1.plot(x, yHat, '-', color='dodgerblue')
    ax1.set_xlabel("Goals", fontsize=14, labelpad=15)
    ax1.set_ylabel("Shots pg", fontsize=14, labelpad=15)
    
    x=data['Goals']
    y=data['Possession%']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x

    ax2 = fig.add_subplot(7,3,2)
    ax2.plot(x, y, '.', color='limegreen')
    ax2.plot(x, yHat, '-')
    ax2.set_xlabel("Goals", fontsize=14, labelpad=15)
    ax2.set_ylabel("Possession%", fontsize=14, labelpad=15)
    
    
    x=data['Possession%']
    y=data['Shots pg']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x
    
    ax4 = fig.add_subplot(7,3,4)
    ax4.plot(x, y, '.', color='limegreen')
    ax4.plot(x, yHat, '-')
    ax4.set_xlabel("Possession%", fontsize=14, labelpad=15)
    ax4.set_ylabel("Shots pg", fontsize=14, labelpad=15)
    
    x=data['Goals']
    y=data['Rating']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x
    
    ax3 = fig.add_subplot(7,3,5)
    ax3.plot(x, y, '.', color='limegreen')
    ax3.plot(x, yHat, '-')
    ax3.set_xlabel("Goals", fontsize=14, labelpad=15)
    ax3.set_ylabel("Rating", fontsize=14, labelpad=15)
    
    x=data['Tournament']
    y=data['Rating']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x
    
    ax5 = fig.add_subplot(7,3,6)
    ax5.plot(x, y, '.', color='limegreen')
    ax5.plot(x, yHat, '-')
    ax5.set_xlabel("Tournament", fontsize=14, labelpad=15)
    ax5.set_ylabel("Rating", fontsize=14, labelpad=15)
    
    x=data['Tournament']
    y=data['Goals']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x
    
    ax6 = fig.add_subplot(7,3,7)
    ax6.plot(x, y, '.', color='limegreen')
    ax6.plot(x, yHat, '-')
    ax6.set_xlabel("Tournament", fontsize=14, labelpad=15)
    ax6.set_ylabel("Goals", fontsize=14, labelpad=15)
    
    x=data['Tournament']
    y=data['red_cards']
    
    slope, intercept, rvalue, pvalue, se = scipy.stats.linregress(x, y)
    yHat = intercept + slope * x
    
    ax7 = fig.add_subplot(7,3,3)
    ax7.plot(x, y, '.', color='limegreen')
    ax7.plot(x, yHat, '-')
    ax7.set_xlabel("Tournament", fontsize=14, labelpad=15)
    ax7.set_ylabel("red_cards", fontsize=14, labelpad=15)
    
    fig.tight_layout()

    fig.savefig(chartFileName)
    


# In[120]:


def predictionTechniques(data):
    from sklearn import preprocessing
    from sklearn.naive_bayes import GaussianNB
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    
    
    count=0
    AccuracyValuesGoals = []
    while count<31:
        label_encoder = preprocessing.LabelEncoder()
        data['Team']= label_encoder.fit_transform(data['Team'])
        data['Tournament']= label_encoder.fit_transform(data['Tournament'])
    
        xcolumns = ['Team', 'Tournament','Shots pg','yellow_cards','red_cards','Possession%','Pass%','AerialsWon','Rating']
        x = data[xcolumns].values
        y = data['Goals'].values

    
        xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size=0.25, shuffle=True)
                
        
        algorithm = GaussianNB()
        algorithm.fit(xtrain, ytrain)
        y_pred = algorithm.predict(xtest)
            
        matrix = confusion_matrix(ytest, y_pred)
    
        precision = precision_score(ytest, y_pred, average='weighted', labels=np.unique(y_pred))
        
        
    
        AccuracyValuesGoals.append(precision)
    
        count +=1
        
    count2=0
    AccuracyValuesTournament = []
    while count2<31:
        label_encoder = preprocessing.LabelEncoder()
        data['Team']= label_encoder.fit_transform(data['Team'])
        data['Tournament']= label_encoder.fit_transform(data['Tournament'])
    
        xcolumns = ['Team', 'Goals','Shots pg','yellow_cards','red_cards','Possession%','Pass%','AerialsWon','Rating']
        x = data[xcolumns].values
        y = data['Tournament'].values

    
        xtrain, xtest, ytrain, ytest = train_test_split(x, y , test_size=0.25, shuffle=True)
                
        
        algorithm = GaussianNB()
        algorithm.fit(xtrain, ytrain)
        y_pred = algorithm.predict(xtest)
            
        matrix = confusion_matrix(ytest, y_pred)
    
        precision = precision_score(ytest, y_pred, average='weighted', labels=np.unique(y_pred))
        
        
    
        AccuracyValuesTournament.append(precision)
    
        count2 +=1
        
    return AccuracyValuesGoals, AccuracyValuesTournament
    


# In[121]:


def descriptiveStatistics(data, AccuracyValuesGoals, AccuracyValuesTournament):
    AccuracyValuesDF = pd.DataFrame()
    AccuracyValuesDF['Accuracy Goal']  = AccuracyValuesGoals
    AccuracyValuesDF['Accuracy Tournament']  = AccuracyValuesTournament
    
    for column in AccuracyValuesDF:
        mean = AccuracyValuesDF[column].mean()
        median = AccuracyValuesDF[column].median()
        mode = AccuracyValuesDF[column].mode()[0]
        stdev = AccuracyValuesDF[column].std()
        minimum = AccuracyValuesDF[column].min()
        maximum = AccuracyValuesDF[column].max()
        therange = maximum - minimum
        print(f"\n{column}:")
        print(f"\t Mean = {mean:.2f}")
        print(f"\t Median = {median:.2f}")
        print(f"\t Mode = {mode:.2f}")
        print(f"\t Standart Deviation = {stdev:.2f}")
        print(f"\t Range = {therange:.2f}")
    


# In[122]:


df = pd.DataFrame(['https://www.kaggle.com/varpit94/football-teams-rankings-stats'])
def make_clickable(val):
    print("\n \nThe data used for this project has been collected from the next link:")
    return '<a href="{}">{}</a>'.format(val,val)


# In[126]:


print('Welcome to my Introduction to Data Science - Final Project:')
print('\nLINK: https://www.kaggle.com/varpit94/football-teams-rankings-stats')
print('\n')
print('Descriptive statistics:')
data = readFile(FileName)
AccuracyValuesGoals, AccuracyValuesTournament = predictionTechniques(data)
descriptiveStatistics(data, AccuracyValuesGoals, AccuracyValuesTournament)
exploreData(data)
df.style.format(make_clickable)


# In[ ]:




