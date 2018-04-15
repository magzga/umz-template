import pandas as pd
import os
import seaborn as sns
from sklearn import linear_model

flats_train = pd.read_csv(os.path.join('train', 'train.tsv'), sep = '\t', names= ['price','isNew','rooms','floor','location','sqrMeters'])
flats_test = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep = '\t', names= ['isNew','rooms','floor','location','sqrMeters'])
flats_test_A = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep = '\t', names= ['isNew','rooms','floor','location','sqrMeters'])

flats_X_train = pd.DataFrame(flats_train, columns=['rooms', 'floor', 'sqrMeters'])
flats_X_test = pd.DataFrame(flats_test, columns=['rooms', 'floor', 'sqrMeters'])
flats_X_test_A = pd.DataFrame(flats_test_A, columns=['rooms', 'floor', 'sqrMeters'])
flats_y_train = flats_train['price']

reg = linear_model.LinearRegression()
reg.fit(flats_X_train, flats_y_train)
flats_y_pred = reg.predict(flats_X_test)
print(reg.coef_)

file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(reg.predict(flats_X_test)):
    file.write(str(line)+'\n')

file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(reg.predict(flats_X_test_A)):
    file.write(str(line)+'\n')