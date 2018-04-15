import pandas as pd
import os
import seaborn as sns
from sklearn import linear_model

cars_train = pd.read_csv(os.path.join('train', 'in.tsv'), sep = '\t', names= ['price','mileage','year','brand','engineType','engineCapacity'])
cars_test = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep = '\t', names= ['mileage','year','brand','engineType','engineCapacity'])
cars_test_A = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep = '\t', names= ['mileage','year','brand','engineType','engineCapacity'])

cars_X_train = pd.DataFrame(cars_train, columns=['mileage', 'year','engineCapacity'])
cars_X_test = pd.DataFrame(cars_test, columns=['mileage', 'year','engineCapacity'])
cars_X_test_A = pd.DataFrame(cars_test_A, columns=['mileage', 'year','engineCapacity'])
cars_y_train = cars_train['price']

reg = linear_model.LinearRegression()
reg.fit(cars_X_train, cars_y_train)
cars_y_pred = reg.predict(cars_X_test)
print(reg.coef_)

file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(reg.predict(cars_X_test)):
    file.write(str(line)+'\n')

file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(reg.predict(cars_X_test_A)):
    file.write(str(line)+'\n')