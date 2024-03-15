import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.decomposition import PCA
import pickle
from sklearn.metrics import precision_score, recall_score, accuracy_score

def val_cgm_data(cgm_data, timestamps, isMeal):
    data = []
    num_of_cls = 0
    length = len(timestamps) if isMeal else len(timestamps) - 1
    for i in range(length):
        ts = timestamps[i]
        if isMeal == True:
            meal_start = pd.to_datetime(ts - pd.Timedelta(minutes=30))
            meal_end = pd.to_datetime(ts + pd.Timedelta(minutes=120))
            num_of_cls = 30
        else:
            meal_start = pd.to_datetime(ts + pd.Timedelta(minutes=120))
            meal_end = timestamps[i + 1]
            num_of_cls = 24
        ts_excluded = (cgm_data['date_time'] >= meal_start) & (cgm_data['date_time'] <= meal_end)
        cgm_filter = cgm_data.loc[ts_excluded]['Sensor Glucose (mg/dL)'].values.tolist()
        data.append(cgm_filter[:num_of_cls])

    return pd.DataFrame(data)

def data_preprocess(insu_data, cgm_data):
    insu_data = insu_data.sort_values(by='date_time')
    insu_data = insu_data[insu_data['BWZ Carb Input (grams)'] != 0.0].dropna()
    insu_data = insu_data.reset_index().drop(columns='index')
    cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',
                                                                                        limit_direction='both')

    return insu_data, cgm_data

def meal_info(insu_data, gluco_data):
    ins_data, cgm_data = data_preprocess(insu_data, gluco_data)

    valid_timestamps = []
    for i in range(len(ins_data['date_time']) - 1):
        ins1 = ins_data['date_time'][i]
        ins2 = ins_data['date_time'][i + 1]
        if (ins2 - ins1).seconds / 60.0 >= 120:
            valid_timestamps.append(ins1)

    meal_data = val_cgm_data(cgm_data, valid_timestamps, isMeal=True)
    return meal_data.dropna()


def nonmeal_info(insu_data, gluco_data):
    ins_data, cgm_data = data_preprocess(insu_data, gluco_data)

    valid_timestamps = []
    for i in range(len(ins_data['date_time']) - 1):
        ins1 = ins_data['date_time'][i]
        ins2 = ins_data['date_time'][i + 1]
        if (ins2 - ins1).seconds / 60.0 >= 240:
            valid_timestamps.append(ins1)

    data = val_cgm_data(cgm_data, valid_timestamps, isMeal=False)
    return data.dropna()

def cal_zero_crosses(row_values, k_max):
    zero_crosses = []

    sl = np.diff(row_values)
    sign = 1 if sl[0] > 0 else 0

    for i in range(1, len(sl)):
        cur_slope_sign = 1 if sl[i] > 0 else 0
        if sign != cur_slope_sign:
            zero_crosses.append([sl[i] - sl[i - 1], i])
        sign = cur_slope_sign

    return sorted(zero_crosses, reverse=True)[k_max] if k_max < len(zero_crosses) else [0, 0]

def get_fourier_trans(row, k_max):
    fast_fourier = fft(row)
    amp = sorted([np.abs(amp) for amp in fast_fourier])
    return amp[-k_max]

def feature_extract(cgm_data):
    features = pd.DataFrame()
    rows = cgm_data.shape[0]

    for i in range(0, rows):
        row_data = cgm_data.iloc[i, :].tolist()

        first_max = cal_zero_crosses(row_data, 0)
        second_max = cal_zero_crosses(row_data, 1)
        third_max = cal_zero_crosses(row_data, 2)

        fft_second_max = get_fourier_trans(row_data, 2)
        fft_third_max = get_fourier_trans(row_data, 3)
        fft_fourth_max = get_fourier_trans(row_data, 4)

        # Append these values as features of this row into features list
        features_per_sample_dict = {
            'ZeroCross Max 1': first_max[0], 'ZeroCross Max 1 index': first_max[1],
            'ZeroCross Max 2': second_max[0], 'ZeroCross Max 2 index': second_max[1],
            'ZeroCross Max 3': third_max[0], 'ZeroCross Max 3 index': third_max[1],
            'FFTAmpl Max 2': fft_second_max, 'FFTAmpl Max 3': fft_third_max, 'FFTAmpl Max 4': fft_fourth_max
        }
        features = features.append(features_per_sample_dict, ignore_index=True)

    return features

acc_scores = []
precision = []
recall = []
f1 = []
score = []

def train(samples, labels):
    model = 'Meal_NoMeal_Classifier'

    k = 10
    kfold = KFold(n_splits=10, shuffle=True, random_state=1)

    for i, j in kfold.split(samples, labels):
        X_train, X_test = samples.iloc[i], samples.iloc[j]
        Y_train, Y_test = labels.iloc[i], labels.iloc[j]

        model = GaussianProcessClassifier(kernel=1.0 * RBF(1), random_state=0)
        model.fit(X_train, Y_train)
        pred_Y = model.predict(X_test)

        score.append(model.score(X_test, Y_test))

        pr = precision_score(Y_test, pred_Y, average='binary')
        re = recall_score(Y_test, pred_Y, average='binary')

        precision.append(pr)
        recall.append(re)
        f1.append(2 * pr * re / (pr + re))

        acc_scores.append(accuracy_score(Y_test, pred_Y))

    return model

def dump_data(meal_pca, noMeal_pca):
    pca_data = pd.concat([meal_pca, noMeal_pca])
    samples = pca_data.iloc[:, :-1]
    labels = pca_data.iloc[:, -1]

    model = train(samples, labels)
    with open('training_modal.pkl', 'wb') as file:
        pickle.dump(model, file)


def main():
    columns = ['Date', 'Time', 'BWZ Carb Input (grams)']
    insulin_data1 = pd.read_csv('InsulinData.csv', usecols=columns)
    insulin_data2 = pd.read_csv('Insulin_patient2.csv', usecols=columns)
    insu_data = pd.concat([insulin_data1, insulin_data2])
    cgm_columns = ['Date', 'Time', 'Sensor Glucose (mg/dL)']
    cgm_data1 = pd.read_csv('CGMData.csv', usecols=cgm_columns)
    cgm_data2 = pd.read_csv('CGM_patient2.csv', usecols=cgm_columns)
    cgm_data = pd.concat([cgm_data1, cgm_data2])

    insu_data['date_time'] = pd.to_datetime(insu_data['Date'] + " " + insu_data['Time'])
    cgm_data['date_time'] = pd.to_datetime(cgm_data['Date'] + " " + cgm_data['Time'])
    meal_data = meal_info(insu_data, cgm_data)
    noMeal_data = nonmeal_info(insu_data, cgm_data)

    pca = PCA(n_components=7)

    noMeal_feature = feature_extract(noMeal_data)
    noMeal_feature = (noMeal_feature - noMeal_feature.mean()) / (noMeal_feature.max() - noMeal_feature.min())
    noMeal_std = StandardScaler().fit_transform(noMeal_feature)
    noMeal_pca = pd.DataFrame(pca.fit_transform(noMeal_std))
    noMeal_pca['class'] = 0

    meal_feature = feature_extract(meal_data)
    meal_feature = (meal_feature - meal_feature.mean()) / (meal_feature.max() - meal_feature.min())
    meal_std = StandardScaler().fit_transform(meal_feature)
    meal_pca = pd.DataFrame(pca.fit_transform(meal_std))
    meal_pca['class'] = 1

    dump_data(meal_pca, noMeal_pca)

    print('Avg Score:', (np.sum(score) / 10) * 100)
    print('Avg Accuracy Score:', (np.sum(acc_scores) / 10) * 100)
    print('Avg precision Score:', (np.sum(precision) / 10) * 100)
    print('Avg recall Score:', (np.sum(recall) / 10) * 100)
    print('Avg F1 Score:', (np.sum(f1) / 10) * 100)


main()