import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


rtrain = pd.read_csv(os.path.join('train', 'train.tsv'), sep='\t', names=[
                     "Occupancy", "date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])

print(rtrain.describe())
print('-' * 100)
print('Occupancy % :', end='')
print(sum(rtrain.Occupancy) / len(rtrain))
print('zero rule model accuracy on training set is',
      1 - sum(rtrain.Occupancy) / len(rtrain))
print('-' * 100)


lr = LogisticRegression()
lr.fit(rtrain.CO2.values.reshape(-1, 1), rtrain.Occupancy)

print('lr model on CO2 only accuracy on training data: ', end='')
print(sum(lr.predict(rtrain.CO2.values.reshape(-1, 1))
          == rtrain.Occupancy) / len(rtrain))


print('-'*100)

lr_full = LogisticRegression()
X = pd.DataFrame(
    rtrain, columns=['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio'])
lr_full.fit(X, rtrain.Occupancy)
print('lr model on all variables except date accuracy on training data: ', end='')
print(sum(lr_full.predict(X) == rtrain.Occupancy) / len(rtrain))
print('True Positives: ', end ='')
print(sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 1)))
print('True Negatives: ', end ='')
print(sum((lr_full.predict(X) == rtrain.Occupancy) & (lr_full.predict(X) == 0)))
print('False Positives: ', end ='')
print(sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 1)))
print('False Negatives: ', end ='')
print(sum((lr_full.predict(X) != rtrain.Occupancy) & (lr_full.predict(X) == 0)))
print('-'*100)
print('-'*100)
print('-'*100)


rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["date", "Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev = pd.DataFrame(rdev,columns = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"])
rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names=['y'])
print('accuracy on dev data (full model):', end = '')
print(sum(lr_full.predict(rdev) == rdev_expected['y'] ) / len(rdev_expected))

print('-'*100)
print('writing to the expected file')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(lr_full.predict(rdev)):
   file.write(str(line)+'\n')
print('-'*100)
print('plotting...')

sns.regplot(x=rdev.CO2, y=rdev_expected.y,
            logistic=True, y_jitter=.1)
plt.show()
