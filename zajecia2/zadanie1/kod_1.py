import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names=[
                     "Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

print(rtrain.describe())
print('-' * 100)
print('Occupancy % on training set: ', end='')
print(sum(rtrain.Occupancy) / len(rtrain))
print('zero rule model accuracy on training set is',
      1 - sum(rtrain.Occupancy) / len(rtrain))
print('-'*100)

lr_full = LogisticRegression()
X = pd.DataFrame(rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
lr_full.fit(X, rtrain.Occupancy)
print('lr model on all variables except date accuracy on training data: ', end='')
print(sum(lr_full.predict(X) == rtrain.Occupancy) / len(rtrain))
print('True Positives: ', end ='')
TP_train = sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1))
print(TP_train)
print('True Negatives: ', end ='')
TN_train = sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 0))
print(TN_train)
print('False Positives: ', end ='')
FP_train = sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 1))
print(FP_train)
print('False Negatives: ', end ='')
FN_train = sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 0))
print(FN_train)
print('lr model on all variables except date sensitivity on training data: ', end = '')
print(TP_train/(TP_train + FN_train))
print('lr model on all variables except date specifity on training data: ', end = '')
print(TN_train/(TN_train + FP_train))
print('-'*100)

rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])

print('Occupancy % on dev set: ', end='')
print(sum(rdev_expected.y) / len(rdev_expected))
print('zero rule model accuracy on dev set is',
      1 - sum(rdev_expected.y) / len(rdev_expected))
print('-'*100)

print('lr model on all variables except date accuracy on dev data: ', end = '')
print(sum(lr_full.predict(rdev) == rdev_expected['y'] ) / len(rdev_expected))
print('True Positives: ', end ='')
TP_dev = sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 1))
print(TP_dev)
print('True Negatives: ', end ='')
TN_dev = sum((lr_full.predict(rdev) == rdev_expected['y']) & (lr_full.predict(rdev) == 0))
print(TN_dev)
print('False Positives: ', end ='')
FP_dev = sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 1))
print(FP_dev)
print('False Negatives: ', end ='')
FN_dev = sum((lr_full.predict(rdev) != rdev_expected['y']) & (lr_full.predict(rdev) == 0))
print(FN_dev)
print('lr model on all variables except date sensitivity on dev data: ', end = '')
print(TP_dev/(TP_dev + FN_dev))
print('lr model on all variables except date specifity on dev data: ', end = '')
print(TN_dev/(TN_dev + FP_dev))

rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rtest = pd.DataFrame(rtest,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

print('-'*100)
print('writing to the expected file')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(lr_full.predict(rdev)):
   file.write(str(line)+'\n')

file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(lr_full.predict(rtest)):
   file.write(str(line)+'\n')

print('-'*100)
print('plotting...')

sns.regplot(x=rdev.CO2, y=rdev_expected.y,
            logistic=True, y_jitter=.1)
plt.show()