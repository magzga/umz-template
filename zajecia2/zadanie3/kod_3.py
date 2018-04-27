import pandas as pd
import seaborn as sns
import os
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

rtrain = pd.read_csv(os.path.join('train', 'in.tsv'), sep='\t', names = ["results", "p1_1", "p1_2", "p2_1", "p2_2", "p3_1",
                                                                         "p3_2", "p4_1", "p4_2", "p5_1", "p5_2", "p6_1", "p6_2",
                                                                         "p7_1", "p7_2", "p8_1", "p8_2", "p9_1", "p9_2", "p10_1",
                                                                         "p10_2", "p11_1", "p11_2", "p12_1", "p12_2", "p13_1",
                                                                         "p13_2", "p14_1", "p14_2", "p15_1", "p15_2", "p16_1",
                                                                         "p16_2", "p17_1", "p17_2"])

print(rtrain.describe())
print('-' * 100)
print('Good % on training set: ', end='')
print(sum(rtrain.results == "g") / len(rtrain))
print('zero rule model accuracy on training set is',
      sum(rtrain.results == "g") / len(rtrain))
print('-'*100)

lr = LogisticRegression()
X = pd.DataFrame(rtrain, columns=['p1_1', 'p1_2', 'p2_1', 'p2_2', 'p3_1', 'p3_2', 'p4_1', 'p4_2', 'p5_1', 'p5_2',
                                  'p6_1', 'p6_2', 'p7_1', 'p7_2', 'p8_1', 'p8_2', 'p9_1', 'p9_2', 'p10_1', 'p10_2',
                                  'p11_1', 'p11_2', 'p12_1', 'p12_2', 'p13_1', 'p13_2', 'p14_1', 'p14_2', 'p15_1',
                                  'p15_2', 'p16_1', 'p16_2', 'p17_1', 'p17_2'])
lr.fit(X, rtrain.results)
print('lr model on first puls number accuracy on training data: ', end='')
print(sum(lr.predict(X) == rtrain.results) / len(rtrain))
print('True Positives: ', end ='')
TP_train = sum((lr.predict(X) == rtrain.results) & (lr.predict(X) == "g"))
print(TP_train)
print('True Negatives: ', end ='')
TN_train = sum((lr.predict(X) == rtrain.results) & (lr.predict(X) == "b"))
print(TN_train)
print('False Positives: ', end ='')
FP_train = sum((lr.predict(X) != rtrain.results) & (lr.predict(X) == "g"))
print(FP_train)
print('False Negatives: ', end ='')
FN_train = sum((lr.predict(X) != rtrain.results) & (lr.predict(X) == "b"))
print(FN_train)
print('lr model on first pulse number sensitivity on training data: ', end = '')
print(TP_train/(TP_train + FN_train))
print('lr model on first pulse number specifity on training data: ', end = '')
print(TN_train/(TN_train + FP_train))
print('-'*100)

rdev = pd.read_csv(os.path.join('dev-0', 'in.tsv'), sep='\t', names=["p1_1", "p1_2", "p2_1", "p2_2", "p3_1", "p3_2",
                                                                     "p4_1", "p4_2", "p5_1", "p5_2", "p6_1", "p6_2",
                                                                     "p7_1", "p7_2", "p8_1", "p8_2", "p9_1", "p9_2",
                                                                     "p10_1", "p10_2", "p11_1", "p11_2", "p12_1",
                                                                     "p12_2", "p13_1", "p13_2", "p14_1", "p14_2",
                                                                     "p15_1", "p15_2", "p16_1", "p16_2", "p17_1", "p17_2"])

rdev_expected = pd.read_csv(os.path.join('dev-0', 'expected.tsv'), sep='\t', names = ["results"])

rdev = pd.DataFrame(rdev,columns=['p1_1', 'p1_2', 'p2_1', 'p2_2', 'p3_1', 'p3_2', 'p4_1', 'p4_2', 'p5_1', 'p5_2',
                                  'p6_1', 'p6_2', 'p7_1', 'p7_2', 'p8_1', 'p8_2', 'p9_1', 'p9_2', 'p10_1', 'p10_2',
                                  'p11_1', 'p11_2', 'p12_1', 'p12_2', 'p13_1', 'p13_2', 'p14_1', 'p14_2', 'p15_1',
                                  'p15_2', 'p16_1', 'p16_2', 'p17_1', 'p17_2'])
print('Good % on dev set: ', end='')
print(sum(rdev_expected.results == "g") / len(rdev_expected.results))
print('zero rule model accuracy on dev set is',
      sum(rdev_expected.results == "g") / len(rdev_expected.results))
print('-'*100)

print('lr model on first puls number accuracy on dev data: ', end = '')
print(sum(lr.predict(rdev) == rdev_expected.results) / len(rdev_expected))
print('True Positives: ', end ='')
TP_dev = sum((lr.predict(rdev) == rdev_expected.results) & (lr.predict(rdev) == "g"))
print(TP_dev)
print('True Negatives: ', end ='')
TN_dev = sum((lr.predict(rdev) == rdev_expected.results) & (lr.predict(rdev) == "b"))
print(TN_dev)
print('False Positives: ', end ='')
FP_dev = sum((lr.predict(rdev) != rdev_expected.results) & (lr.predict(rdev) == "g"))
print(FP_dev)
print('False Negatives: ', end ='')
FN_dev = sum((lr.predict(rdev) != rdev_expected.results) & (lr.predict(rdev) == "b"))
print(FN_dev)
print('lr model on first puls number sensitivity on dev data: ', end = '')
print(TP_dev/(TP_dev + FN_dev))
print('lr model on first puls number specifity on dev data: ', end = '')
print(TN_dev/(TN_dev + FP_dev))

rtest = pd.read_csv(os.path.join('test-A', 'in.tsv'), sep='\t', names=["p1_1", "p1_2", "p2_1", "p2_2", "p3_1", "p3_2",
                                                                       "p4_1", "p4_2", "p5_1", "p5_2", "p6_1", "p6_2",
                                                                       "p7_1", "p7_2", "p8_1", "p8_2", "p9_1", "p9_2",
                                                                       "p10_1", "p10_2", "p11_1", "p11_2", "p12_1",
                                                                       "p12_2", "p13_1", "p13_2", "p14_1", "p14_2",
                                                                       "p15_1", "p15_2", "p16_1", "p16_2", "p17_1", "p17_2"])
rtest = pd.DataFrame(rtest,columns = ['p1_1', 'p1_2', 'p2_1', 'p2_2', 'p3_1', 'p3_2', 'p4_1', 'p4_2', 'p5_1', 'p5_2',
                                      'p6_1', 'p6_2', 'p7_1', 'p7_2', 'p8_1', 'p8_2', 'p9_1', 'p9_2', 'p10_1', 'p10_2',
                                      'p11_1', 'p11_2', 'p12_1', 'p12_2', 'p13_1', 'p13_2', 'p14_1', 'p14_2', 'p15_1',
                                      'p15_2', 'p16_1', 'p16_2', 'p17_1', 'p17_2'])

print('-'*100)
print('writing to the expected file')
file = open(os.path.join('dev-0', 'out.tsv'), 'w')
for line in list(lr.predict(rdev)):
   file.write(str(line)+'\n')


file = open(os.path.join('test-A', 'out.tsv'), 'w')
for line in list(lr.predict(rtest)):
   file.write(str(line)+'\n')

print('-'*100)
print('plotting...')

rdev_expected.results = rdev_expected.results.map(lambda x: 1 if x == "g" else 0, 'results')
sns.regplot(x=rdev.p3_1, y=rdev_expected.results,
           logistic=True, y_jitter=.1)
plt.legend(["Results: \n  1 = good  \n  0 = bad"], loc="upper left")
plt.show()