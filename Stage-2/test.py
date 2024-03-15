import pandas as pd
import pickle

from train import feature_extract
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

with open('training_modal.pkl', 'rb') as file:
    model = pickle.load(file)
    test_df = pd.read_csv('test.csv', header=None)

pca = PCA(n_components = 7)

feature = feature_extract(test_df)
feature_test = (feature - feature.mean()) / (feature.max() - feature.min())
std = StandardScaler().fit_transform(feature_test)
pca_test = pca.fit_transform(std)

predict = pd.DataFrame(model.predict(pca_test))
predict.to_csv("Results.csv", header=None, index=False)