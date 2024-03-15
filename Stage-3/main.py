#!/usr/bin/env python


import pandas as pd
import numpy as np
#import pickle_compat
#pickle_compat.patch()
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
import math
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from scipy import stats
from scipy.fftpack import rfft
from scipy.integrate import simps
from scipy.stats import iqr
from scipy.signal import periodogram

warnings.filterwarnings("ignore")

def getMTD(InsulinData):
    time =[]
    InsulinVal =[]
    InsulinLev =[]
    Time1=[]
    Time2 =[]
    MealTime = []
    Difference =[]
    ColValue= InsulinData['BWZ Carb Input (grams)']
    MaxValue= ColValue.max()
    MinValue = ColValue.min()
    CalcValues = math.ceil(MaxValue-MinValue/60)
     
    for i in InsulinData['datetime']:
        time.append(i)
    for i in InsulinData['BWZ Carb Input (grams)']:
        InsulinVal.append(i)
    for i,j in enumerate(time):
        if(i<len(time)-1):
            Difference.append((time[i+1]-time[i]).total_seconds()/3600)
    Time1 = time[0:-1]
    Time2 = time[1:]
    CalcValues=[]
    for i in InsulinVal[0:-1]:
        CalcValues.append(0 if (i>=MinValue and i<=MinValue+20)
                          else 1 if (i>=MinValue+21 and i<=MinValue+40)
                          else 2 if(i>=MinValue+41 and i<=MinValue+60) 
                          else 3 if(i>=MinValue+61 and i<=MinValue+80)
                          else 4 if(i>=MinValue+81 and i<=MinValue+100) 
                          else 5 )
    ListValues = list(zip(Time1, Time2, Difference,CalcValues))
    for j in ListValues:
        if j[2]>2.5:
            MealTime.append(j[0])
            InsulinLev.append(j[3])
        else:
            continue
    return MealTime,InsulinLev

def getMD(mealTimes,startTime,endTime,insulinLevels,new_Glucose_data):
    newMealDataRows = []
    for j,newTime in enumerate(mealTimes):
        meal_index_start= new_Glucose_data[new_Glucose_data['datetime'].between(newTime+ pd.DateOffset(hours=startTime),newTime + pd.DateOffset(hours=endTime))]
        
        if meal_index_start.shape[0]<8:
            del insulinLevels[j]
            continue
        GlucoseValues = meal_index_start['Sensor Glucose (mg/dL)'].to_numpy()
        mean = meal_index_start['Sensor Glucose (mg/dL)'].mean()
        missing_values_count = 30 - len(GlucoseValues)
        if missing_values_count > 0:
            for i in range(missing_values_count):
                GlucoseValues = np.append(GlucoseValues, mean)
        newMealDataRows.append(GlucoseValues[0:30])
    return pd.DataFrame(data=newMealDataRows),insulinLevels


def CD(insulin_data,Glucose_data):
    mealData = pd.DataFrame()
    Glucose_data['Sensor Glucose (mg/dL)'] = Glucose_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    insulin_data= insulin_data[::-1]
    Glucose_data= Glucose_data[::-1]
    insulin_data['datetime']= insulin_data['Date']+" "+insulin_data['Time']
    insulin_data['datetime']=pd.to_datetime(insulin_data['datetime'])
    Glucose_data['datetime']= Glucose_data['Date']+" "+Glucose_data['Time']
    Glucose_data['datetime']=pd.to_datetime(insulin_data['datetime'])
    
    InsulinDataNew = insulin_data[['datetime','BWZ Carb Input (grams)']]
    GlucoseDataNew = Glucose_data[['datetime','Sensor Glucose (mg/dL)']]

    InsulinNew1 = InsulinDataNew[(InsulinDataNew['BWZ Carb Input (grams)']>0) ]
    MealTime,InsulinLev = getMTD(InsulinNew1)
    MealData,InsulinLev_New = getMD(MealTime,-0.5,2,InsulinLev,GlucoseDataNew)

    return MealData,InsulinLev_New


def createMFMatrix(inputMealdata):
    index = inputMealdata.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    cleanMealData = inputMealdata.drop(inputMealdata.index[index]).reset_index().drop(columns="index")
    cleanMealData = cleanMealData.interpolate(method="linear", axis=1)
    indexToDropAgain = cleanMealData.isna().sum(axis=1).replace(0, np.nan).dropna().index
    cleanMealData = cleanMealData.drop(inputMealdata.index[indexToDropAgain]).reset_index().drop(columns="index")
    cleanMealData = cleanMealData.dropna().reset_index().drop(columns="index")
    (
        powerFirstMax,
        powerSecondMax,
        powerThirdMax,
        indexFirstMax,
        indexSecondMax,
        rms_val,
        auc_val,
    ) = ([], [], [], [], [], [], [])
    for i in range(len(cleanMealData)):
        arr = abs(rfft(cleanMealData.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sndOrdArr = abs(rfft(cleanMealData.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sndOrdArr.sort()
        powerFirstMax.append(sndOrdArr[-2])
        powerSecondMax.append(sndOrdArr[-3])
        powerThirdMax.append(sndOrdArr[-4])
        indexFirstMax.append(arr.index(sndOrdArr[-2]))
        indexSecondMax.append(arr.index(sndOrdArr[-3]))
        rms_row = np.sqrt(np.mean(cleanMealData.iloc[i, 0:30] ** 2))
        rms_val.append(rms_row)
        auc_row = abs(simps(cleanMealData.iloc[i, 0:30], dx=1))
        auc_val.append(auc_row)
    featuredMealMat = pd.DataFrame()

    velocity = np.diff(cleanMealData, axis=1)
    velocity_min = np.min(velocity, axis=1)
    velocity_max = np.max(velocity, axis=1)
    velocity_mean = np.mean(velocity, axis=1)

    acceleration = np.diff(velocity, axis=1)
    acceleration_min = np.min(acceleration, axis=1)
    acceleration_max = np.max(acceleration, axis=1)
    acceleration_mean = np.mean(acceleration, axis=1)

    featuredMealMat['velocity_min'] = velocity_min
    featuredMealMat['velocity_max'] = velocity_max
    featuredMealMat['velocity_mean'] = velocity_mean
    featuredMealMat['acceleration_min'] = acceleration_min
    featuredMealMat['acceleration_max'] = acceleration_max
    featuredMealMat['acceleration_mean'] = acceleration_mean
    row_entropies = cleanMealData.apply(row_entropy, axis=1)
    featuredMealMat['row_entropies'] = row_entropies
    iqr_values = cleanMealData.apply(iqr, axis=1)
    featuredMealMat['iqr_values'] = iqr_values

    power_first_max = []
    power_second_max = []
    power_third_max = []
    power_fourth_max = []
    power_fifth_max = []
    power_sixth_max = []
    for it, rowdt in enumerate(cleanMealData.iloc[:, 0:30].values.tolist()):
        ara = abs(rfft(rowdt)).tolist()
        sort_ara = abs(rfft(rowdt)).tolist()
        sort_ara.sort()
        power_first_max.append(sort_ara[-2])
        power_second_max.append(sort_ara[-3])
        power_third_max.append(sort_ara[-4])
        power_fourth_max.append(sort_ara[-5])
        power_fifth_max.append(sort_ara[-6])
        power_sixth_max.append(sort_ara[-7])

    featuredMealMat['fft col1'] = power_first_max
    featuredMealMat['fft col2'] = power_second_max
    featuredMealMat['fft col3'] = power_third_max
    featuredMealMat['fft col4'] = power_fourth_max
    featuredMealMat['fft col5'] = power_fifth_max
    featuredMealMat['fft col6'] = power_sixth_max
    frequencies, psd_values = periodogram(cleanMealData, axis=1)

    psd1_values = np.mean(psd_values[:, 0:6], axis=1)
    psd2_values = np.mean(psd_values[:, 5:11], axis=1)
    psd3_values = np.mean(psd_values[:, 10:16], axis=1)
    featuredMealMat['psd1_values'] = psd1_values
    featuredMealMat['psd2_values'] = psd2_values
    featuredMealMat['psd3_values'] = psd3_values
    print("Feature Mat\n",featuredMealMat)
    print("lengths\n",len(featuredMealMat))
    return featuredMealMat

def row_entropy(row):
    values, counts = np.unique(row, return_counts=True)
    probs = counts / len(row)
    entropy = np.sum(-probs * np.log2(probs))
    return entropy


def Mean_Abso_value(param):
    MeanVal = 0
    for i in range(0, len(param) - 1):
        MeanVal = MeanVal + np.abs(param[(i + 1)] - param[i])
    return MeanVal / len(param)

def Entropy(param):
    Length = len(param)
    entropy = 0
    if Length <= 1:
        return 0
    else:
        value, count = np.unique(param, return_counts=True)
        ratio = count / Length
        nonZero_ratio = np.count_nonzero(ratio)
        if nonZero_ratio <= 1:
            return 0
        for i in ratio:
            entropy -= i * np.log2(i)
        return entropy   

def RMS(param):
    RMS = 0
    for i in range(0, len(param) - 1):
        
        RMS = RMS + np.square(param[i])
    return np.sqrt(RMS / len(param))

def FF(param):
    FF = fft(param)
    Length = len(param)
    i = 2/300
    amplitude = []
    frequency = np.linspace(0, Length * i, Length)
    for amp in FF:
        amplitude.append(np.abs(amp))
    sortedAmplitude = amplitude
    sortedAmplitude = sorted(sortedAmplitude)
    MaxAmp = sortedAmplitude[(-2)]
    MaxFreq = frequency.tolist()[amplitude.index(MaxAmp)]
    return [MaxAmp, MaxFreq]


def ZeroCrossing(row, xAxis):
    slopes = [
     0]
    ZeroCross = list()
    ZeroCrossingRate = 0
    X = [i for i in range(xAxis)][::-1]
    Y = row[::-1]
    for index in range(0, len(X) - 1):
        slopes.append((Y[(index + 1)] - Y[index]) / (X[(index + 1)] - X[index]))

    for index in range(0, len(slopes) - 1):
        if slopes[index] * slopes[(index + 1)] < 0:
            ZeroCross.append([slopes[(index + 1)] - slopes[index], X[(index + 1)]])

    ZeroCrossingRate = np.sum([np.abs(np.sign(slopes[(i + 1)]) - np.sign(slopes[i])) for i in range(0, len(slopes) - 1)]) / (2 * len(slopes))
    if len(ZeroCross) > 0:
        return [max(ZeroCross)[0], ZeroCrossingRate]
    else:
        return [
         0, 0]

def Glucose(MealNomealdata):
    Glucose=pd.DataFrame()
    for i in range(0, MealNomealdata.shape[0]):
        param = MealNomealdata.iloc[i, :].tolist()
        Glucose = Glucose.append({ 
         'Minimum Value':min(param), 
         'Maximum Value':max(param),
         'Mean of Absolute Values1':Mean_Abso_value(param[:13]), 
         'Mean of Absolute Values2':Mean_Abso_value(param[13:]), 
         'Max_Zero_Crossing':ZeroCrossing(param, MealNomealdata.shape[1])[0], 
         'Zero_Crossing_Rate':ZeroCrossing(param, MealNomealdata.shape[1])[1], 
         'Root Mean Square':RMS(param),
         'Entropy':RMS(param), 
         'Max FFT Amplitude1':FF(param[:13])[0], 
         'Max FFT Frequency1':FF(param[:13])[1], 
         'Max FFT Amplitude2':FF(param[13:])[0], 
         'Max FFT Frequency2':FF(param[13:])[1]},
          ignore_index=True)
    return Glucose

def Features(MealData):
    Features = Glucose(MealData.iloc[:,:-1])
    
    
    stdScaler = StandardScaler()
    StandardMeal = stdScaler.fit_transform(Features)
    
    pca = PCA(n_components=12)
    pca.fit(StandardMeal)
    
    with open('pcs_Glucose_data.pkl', 'wb') as (file):
        pickle.dump(pca, file)
        
    meal_pca = pd.DataFrame(pca.fit_transform(StandardMeal))
    return meal_pca

def Calculate_Entropy(CalcValues):
    EntropyMealValue= []
    for InsulinValue in CalcValues:
        InsulinValue = np.array(InsulinValue)
        InsulinValue = InsulinValue / float(InsulinValue.sum())
        CalcValueEntropy = (InsulinValue * [ np.log2(Glucose) if Glucose!=0 else 0 for Glucose in InsulinValue]).sum()
        EntropyMealValue += [CalcValueEntropy]
   
    return EntropyMealValue

def Purity(CalcValues):
    MealPurity = []
    for InsulinValue in CalcValues:
        InsulinValue = np.array(InsulinValue)
        InsulinValue = InsulinValue / float(InsulinValue.sum())
        CalcPurity = InsulinValue.max()
        MealPurity += [CalcPurity]
    return MealPurity

def Calculate_DBSCAN(dbscan,test,meal_pca2):
     for i in test.index:
         dbscan=0
         for index,row in meal_pca2[meal_pca2['clusters']==i].iterrows(): 
             test_row=list(test.iloc[0,:])
             meal_row=list(row[:-1])
             for j in range(0,12):
                 dbscan+=((test_row[j]-meal_row[j])**2)
     return dbscan

def Calculate_DBSCAN(dbscan,test,meal_pca2):
    for i in test.index:
        dbscan=0
        for index,row in meal_pca2[meal_pca2['clusters']==i].iterrows(): 
            test_row=list(test.iloc[0,:])
            meal_row=list(row[:-1])
            for j in range(0,12):
                dbscan+=((test_row[j]-meal_row[j])**2)
    return dbscan

def Calculate_CM(groundTruth,Clustered,k):
    Matrix= np.zeros((k, k))
    for i,j in enumerate(groundTruth):
         val1 = j
         val2 = Clustered[i]
         Matrix[val1,val2]+=1
    return Matrix

if __name__=='__main__':
       
    insulin_data=pd.read_csv("InsulinData.csv",low_memory=False)
    Glucose_data=pd.read_csv("CGMData.csv",low_memory=False)
    patient_data,InsulinLev = CD(insulin_data,Glucose_data)

    tt = createMFMatrix(patient_data)

    meal_pca = Features(patient_data)

kmeans = KMeans(n_clusters=6,max_iter=7000)
kmeans.fit_predict(meal_pca)
pLabels=list(kmeans.labels_)
df = pd.DataFrame()
df['bins']=InsulinLev

df['kmeans_clusters']=pLabels 
print("Kmeans bins",df['kmeans_clusters'])

Matrix = Calculate_CM(df['bins'],df['kmeans_clusters'],6)
print("Kmeans",Matrix)
MatrixEntropy = Calculate_Entropy(Matrix)
MatrixPurity = Purity(Matrix)
Count = np.array([InsulinValue.sum() for InsulinValue in Matrix])
CountVal = Count / float(Count.sum())

KMeanSSE = kmeans.inertia_
KMeansPurity =  (MatrixPurity*CountVal).sum()
KMeansEntropy = -(MatrixEntropy*CountVal).sum()

DBSCANData=pd.DataFrame()
db = DBSCAN(eps=0.127,min_samples=7)
clusters = db.fit_predict(meal_pca)
DBSCANData=pd.DataFrame({'pc1':list(meal_pca.iloc[:,0]),'pc2':list(meal_pca.iloc[:,1]),'clusters':list(clusters)})
OutliersData=DBSCANData[DBSCANData['clusters']==-1].iloc[:,0:2]


initial_value=0
bins = 6
i = max(DBSCANData['clusters'])
while i<bins-1:
        MaxLabel=stats.mode(DBSCANData['clusters']).mode[0] 
        ClusterData=DBSCANData[DBSCANData['clusters']==stats.mode(DBSCANData['clusters']).mode[0]] #mode(dbscan_df['clusters'])]
        bi_kmeans= KMeans(n_clusters=2,max_iter=1000, algorithm = 'auto').fit(ClusterData)
        bi_pLabels=list(bi_kmeans.labels_)
        ClusterData['bi_pcluster']=bi_pLabels
        ClusterData=ClusterData.replace(to_replace =0,  value =MaxLabel) 
        ClusterData=ClusterData.replace(to_replace =1,  value =max(DBSCANData['clusters'])+1) 
       
        for x,y in zip(ClusterData['pc1'],ClusterData['pc2']):
            NewDataLabel=ClusterData.loc[(ClusterData['pc1'] == x) & (ClusterData['pc2'] == y)]
            DBSCANData.loc[(DBSCANData['pc1'] == x) & (DBSCANData['pc2'] == y),'clusters']=NewDataLabel['bi_pcluster']
        df['clusters']=DBSCANData['clusters']
        i+=1  

MatrixDBSCAN = Calculate_CM(df['bins'],DBSCANData['clusters'],6)
print("DBSCAN matrix",MatrixDBSCAN)
ClusterEntropy = Calculate_Entropy(MatrixDBSCAN)
ClusterPurity = Purity(MatrixDBSCAN)
Count = np.array([InsulinValue.sum() for InsulinValue in MatrixDBSCAN])
CountVal = Count / float(Count.sum())

meal_pca2= meal_pca. join(DBSCANData['clusters'])
Centroids = meal_pca2.groupby(DBSCANData['clusters']).mean()
dbscan = Calculate_DBSCAN(initial_value,Centroids.iloc[:, : 12],meal_pca2)
DBSCANPurity =  (ClusterPurity*CountVal).sum()        
DBSCANEntropy = -(ClusterEntropy*CountVal).sum()

FinalData = pd.DataFrame([[KMeanSSE,dbscan,KMeansEntropy,DBSCANEntropy,KMeansPurity,DBSCANPurity]],columns=['K-Means SSE','DBSCAN SSE','K-Means entropy','DBSCAN entropy','K-Means purity','DBSCAN purity'])
FinalData=FinalData.fillna(0)
FinalData.to_csv('Results.csv',index=False,header=None)