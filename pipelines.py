from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import joblib
from AntiBERTy import AntiBERTyEncoder
import pandas as pd

# load models
svm = joblib.load('../models/05062023_svm_60.joblib')
rf = joblib.load('../models/05062023_rf_60.joblib')
gbt = joblib.load('../models/05062023_gb_60.joblib')

data60 = pd.read_csv('../data/combined_datasets_60.csv')


def select_columns(X):
    """

    :param X: antiBERTy encoded dataset.
    :return: reduced dataset with selected features.
    """

    selected_features = data60.columns
    X.columns = ['{}'.format(i) for i in range(len(X.columns))]
    X_new = X[selected_features]
    X_new.columns = ['{}'.format(i) for i in range(len(X_new.columns))]
    return X_new


svm_pipe = Pipeline([
    ('encoder', AntiBERTyEncoder()),
    ('selector', FunctionTransformer(select_columns)),
    ('svm', svm)])

rf_pipe = Pipeline([
    ('encoder', AntiBERTyEncoder()),
    ('selector', FunctionTransformer(select_columns)),
    ('rf', rf)])

gbt_pipe = Pipeline([
    ('encoder', AntiBERTyEncoder()),
    ('selector', FunctionTransformer(select_columns)),
    ('gbt', gbt)])


if __name__ == '__main__':
    # test pipeline
    X = ['QVQLQQSGGELAKPGASVKVSCKASGYTFSSFWMHWVRQAPGQGLEWIGYINPRSGYTEYNEIFRDKATMTTDTSTSTAYMELSSLRSEDTAVYYCASFLGRGAMDYWGQGTTVTVSS',
'DIQMTQSPSSLSASVGDRVTITCRASQDISNYLAWYQQKPGKAPKLLIYYTSKIHSGVPSRFSGSGSGTDYTFTISSLQPEDIATYYCQQGNTFPYTFGQGTKVEIK']

    #print(svm_pipe.predict(X))
    #print(rf_pipe.predict(X))
    print(gbt_pipe.predict(X))


