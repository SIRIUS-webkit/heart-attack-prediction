import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("heart.csv")
output_file = f'model.bin'

duplicate_rows_data = df[df.duplicated()]
df = df.drop_duplicates()

def normalization(data):
    cont_cols = ['age', 'trtbps', 'chol', 'thalachh', 'oldpeak']
    st_scaler = StandardScaler()
    X_norm=data.copy()
    X_norm[cont_cols]=pd.DataFrame(st_scaler.fit_transform(X_norm[cont_cols]), columns=[cont_cols])
    return X_norm

def train(X,y):
    model = LogisticRegression(random_state=42, solver='lbfgs', max_iter=3000)
    model = model.fit(X, y)
    return model

def predict(model,X):
    y_pred = model.predict(X)
    return y_pred

norm_data = normalization(df)

df_train, df_test = train_test_split(norm_data, test_size=0.2, random_state=56)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.output.values
y_test = df_test.output.values

df_train = df_train.fillna(0)
df_test = df_test.fillna(0)

del df_train['output']
del df_test['output']

print("----- training model ------")
model = train(df_train,y_train)
y_pred = predict(model,df_test)
auc = roc_auc_score(y_test,y_pred)
print(auc)

print("----- saving model ------")
f_out = open(output_file,"wb")
pickle.dump(model,f_out)
f_out.close()