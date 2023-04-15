from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from AntiBERTy import AntiBERTyEncoder
import pandas as pd

# load models
svm = joblib.load('./models/150423_svm.joblib')

data72 = pd.read_csv('./data/combined_datasets_72.csv')

selected_features = data72.columns

# svm pipeline with standard scaler
svm_pipe = Pipeline([
    ('encoder', AntiBERTyEncoder()),
    ('feature_selection', ColumnTransformer(transformers=[('num', 'passthrough', selected_features)])),
    ('scaler', StandardScaler()),
    ('svm', svm)])


if __name__ == '__main__':
    # test pipeline
    X = ['ELQMTQSPASLAVSLGQRATISCKASQSVDYDGDSYMNWYQQKPGQPPKLLIYAASNLESGIPARFSGSGSRTDFTLTINPVETDDVATYYCQQSHEDPYTFGGGTKLEIK',
'LESGAELVKPGASVKLSCKASGYIFTTYWMQWVKQRPGQGLEWIGEIHPSNGLTNYNEKFKSKATLTVDKSSTTAYMQLSSLTSEDSAVYYCSKGRELGRFAYWGQGTLVTVSA']

    print(svm_pipe.predict(X))
#%%
