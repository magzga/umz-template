import pandas as pd

report = pd.read_csv('train.tsv',sep = '\t', names = ['price', 'isNew', 'rooms', 'floor', 'location', 'sqrMetres'])

print(report.describe())
#print(report.columns)
