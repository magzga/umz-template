from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import os
import pandas as pd
from keras import optimizers
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
r = pd.read_csv(os.path.join("train", "in.tsv"), header=None, names=[
                "price", "mileage", "year", "brand", "engingeType", "engineCapacity"], sep='\t')

X_train = pd.DataFrame(r, columns=["mileage", "year", "engineCapacity"])
Y_train = pd.DataFrame(r, columns=["price"])
scaler.fit(X_train)
X_train_scale = pd.DataFrame(scaler.transform(X_train), columns=["mileage", "year", "engineCapacity"])


def create_baseline():
    # stworzenie modelu sieci neuronowej
    model = Sequential()
    # dodanie jednego neuronu, wejście do tego neuronu to ilość cech, funkcja aktywacji sigmoid, początkowe wartości wektorów to zero.
    model.add(Dense(4, input_dim=X_train_scale.shape[1], activation='relu', kernel_initializer='normal'))
    model.add(Dense(1, kernel_initializer='normal'))
    # stworzenie funkcji kosztu stochastic gradient descent
    # sgd = optimizers.SGD(lr=0.1)
    # kompilacja modelu
    model.compile(loss='mean_squared_error',optimizer='adam')

    # rysowanie architektury sieci, jeżeli ktoś ma zainstalowane odpowiednie biblioteki
    # from keras.utils import plot_model
    # plot_model(model, to_file='model.png')
    return model

estimator = KerasRegressor(build_fn=create_baseline, epochs=100, batch_size = 2, verbose=True)

estimator.fit(X_train, Y_train)
predictions_train = estimator.predict(X_train)

r = pd.read_csv(os.path.join("dev-0", "in.tsv"), header=None, names=[
                "mileage", "year", "brand", "engingeType", "engineCapacity"], sep='\t')
X_dev = pd.DataFrame(r, columns=["mileage", "year", "engineCapacity"])
Y_dev = pd.read_csv(os.path.join("dev-0", "expected.tsv"),header=None, names=["price"], sep='\t')

predictions_dev = estimator.predict(X_dev)

with open(os.path.join("dev-0", "out.tsv"), 'w') as file:
    for line in list(estimator.predict(X_dev)):
        file.write(str(line) + '\n')

r = pd.read_csv(os.path.join("test-A", "in.tsv"), header=None, names=[
                "mileage", "year", "brand", "engingeType", "engineCapacity"], sep='\t')
X_test = pd.DataFrame(r, columns=["mileage", "year", "engineCapacity"])

predictions_test = estimator.predict(X_test)

with open(os.path.join("test-A", "out.tsv"), 'w') as file:
    for line in list(estimator.predict(X_test)):
        file.write(str(list) + '\n')