from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import joblib
from AntiBERTy import AntiBERTyEncoder
import pandas as pd

# load models
svm = joblib.load('./models/200423_svm_model.joblib')

data72 = pd.read_csv('./data/combined_datasets_72.csv')

#scaler = joblib.load('./models/150423_standscaler.joblib')

def select_columns(X):
    # Replace this list with the names of the columns you want to select
    selected_features = data72.columns
    X.columns = ['{}'.format(i) for i in range(len(X.columns))]
    X_new = X[selected_features]
    X_new.columns = ['{}'.format(i) for i in range(len(X_new.columns))]
    return X_new


# svm pipeline with standard scaler
svm_pipe = Pipeline([
    ('encoder', AntiBERTyEncoder()),
    ('selector', FunctionTransformer(select_columns)),
    ('svm', svm)])


if __name__ == '__main__':
    # test pipeline
    X = ['ELQMTQSPASLAVSLGQRATISCKASQSVDYDGDSYMNWYQQKPGQPPKLLIYAASNLESGIPARFSGSGSRTDFTLTINPVETDDVATYYCQQSHEDPYTFGGGTKLEIK',
'LESGAELVKPGASVKLSCKASGYIFTTYWMQWVKQRPGQGLEWIGEIHPSNGLTNYNEKFKSKATLTVDKSSTTAYMQLSSLTSEDSAVYYCSKGRELGRFAYWGQGTLVTVSA']


    print(svm_pipe.predict(X))

