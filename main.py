import sklearn.linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn. metrics import mean_absolute_error as mae
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
sns.set()

highway1 = pd.read_csv('Highway1.csv')
pd.set_option('display.max_columns', None)

corrs = highway1.corr()
print(corrs)
file = open("task.txt", "w")
file.write(str(corrs))
file.close()

fig1 = ff.create_annotated_heatmap(z=corrs.values, x=list(corrs.columns), y=list(corrs.index), annotation_text=corrs.round(2).values, showscale=True)

X = highway1.values[:, 1:]
X_ForTest = highway1.values[:, 1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)

print("(1) rate, (2) len, (3) adt, (4) trks, (5) sigs1, (6) slim, (7) shld, (8) lane, (9) acpt, (10) itg, (11) lwid")
AxX = int(input("Какой будет первый признак? : "))
AA = X[:, 0]
AA1 = X[:, 1]
AA2 = X[:, 3]
match(AxX):
    case 1:
        AA1 = highway1.values[:, 1]
        XL = 'rate'
        xXx = 0
    case 2:
        AA1 = highway1.values[:, 2]
        XL = 'len'
        xXx = 1
    case 3:
        AA1 = highway1.values[:, 3]
        XL = 'adt'
        xXx = 2
    case 4:
        AA1 = highway1.values[:, 4]
        XL = 'trks'
        xXx = 3
    case 5:
        AA1 = highway1.values[:, 5]
        XL = 'sigs1'
        xXx = 4
    case 6:
        AA1 = highway1.values[:, 6]
        XL = 'slim'
        xXx = 5
    case 7:
        AA1 = highway1.values[:, 7]
        XL = 'shld'
        xXx = 6
    case 8:
        AA1 = highway1.values[:, 8]
        XL = 'lane'
        xXx = 7
    case 9:
        AA1 = highway1.values[:, 9]
        XL = 'acpt'
        xXx = 8
    case 10:
        AA1 = highway1.values[:, 10]
        XL = 'itg'
        xXx = 9
    case 11:
        AA1 = highway1.values[:, 11]
        XL = 'lwid'
        xXx = 10

AxY = int(input("Какой будет второй признак? : "))
match(AxY):
    case 1:
        AA2 = highway1.values[:, 1]
        YL = 'rate'
        yYy = 0
    case 2:
        AA2 = highway1.values[:, 2]
        YL = 'len'
        yYy = 1
    case 3:
        AA2 = highway1.values[:, 3]
        YL = 'adt'
        yYy = 2
    case 4:
        AA2 = highway1.values[:, 4]
        YL = 'trks'
        yYy = 3
    case 5:
        AA2 = highway1.values[:, 5]
        YL = 'sigs1'
        yYy = 4
    case 6:
        AA2 = highway1.values[:, 6]
        YL = 'slim'
        yYy = 5
    case 7:
        AA2 = highway1.values[:, 7]
        YL = 'shld'
        yYy = 6
    case 8:
        AA2 = highway1.values[:, 8]
        YL = 'lane'
        yYy = 7
    case 9:
        AA2 = highway1.values[:, 9]
        YL = 'acpt'
        yYy = 8
    case 10:
        AA2 = highway1.values[:, 10]
        YL = 'itg'
        yYy = 9
    case 11:
        AA2 = highway1.values[:, 11]
        YL = 'lwid'
        yYy = 10

plt.ylabel(YL, fontsize=16)
area = np.pi * (X[:, 1])**2
plt.scatter(AA1, AA2, alpha=0.5)
plt.xlabel(XL, fontsize=18)
plt.title("-")
plt.show()
plt.clf()

in_trainXsplit = train_test_split(AA1, test_size=0.2)
ActualXSplit = train_test_split(AA1, test_size=0.2)
in_trainX0 = in_trainXsplit[0]
in_trainX1 = in_trainXsplit[1]
actualX0 = ActualXSplit[0]
actualX1 = ActualXSplit[1]

for x in actualX1:
    np.append(actualX0, x)
ActualX = np.array(actualX0, dtype=float)

for x in in_trainX1:
    np.append(in_trainX0, x)
in_trainX = np.array(in_trainX0, dtype=float)

in_trainYsplit = train_test_split(AA2, test_size=0.2)
ActualYSplit = train_test_split(AA2, test_size=0.2)
in_trainY0 = in_trainYsplit[0]
in_trainY1 = in_trainYsplit[1]
actualY0 = ActualYSplit[0]
actualY1 = ActualYSplit[1]

for x in actualY1:
    np.append(actualY0, x)
ActualY = np.array(actualY0, dtype=float)

for x in in_trainY1:
    np.append(in_trainY0, x)
in_trainY = np.array(in_trainY0, dtype=float)

# Угловой коэффициент = (f(y2)-f(y1)) / y2 - y1
slope_intercept = np.polyfit(in_trainX, in_trainY, 1)
print()
print("Угловой коэффициент обучающей выборки = " + str(slope_intercept[0]))
print("Пересечение = " + str(slope_intercept[1]))
print()

mse1 = np.square(np.subtract(ActualX, in_trainX)).mean()
mse2 = np.square(np.subtract(ActualY, in_trainY)).mean()
mae1 = mae(ActualX, in_trainX)
mae2 = mae(ActualY, in_trainY)
print("Среднеквадратичная ошибка по первому признаку : " + str(mse1))
print("Среднеквадратичная ошибка по второму признаку : " + str(mse2))
print()
print("Средняя абсолютная ошибка по первому признаку : " + str(mae1))
print("Средняя абсолютная ошибка по второму признаку : " + str(mae2))

area = np.pi * (X[:, 1])**2
ax = plt.subplots()

plot1 = plt.scatter(in_trainX, in_trainY, c = 'r')
plot2 = plt.scatter(ActualX, ActualY, c = [[0.1, 0.63, 0.55]])
plt.legend([plot1, plot2], ["Train", "Test"])
plt.xlabel(XL, fontsize=18)
plt.ylabel(YL, fontsize=16)
plt.title("-")
plt.show()
plt.clf()

regressor = LinearRegression()
regressor.fit(in_trainX.reshape(-1, 1), in_trainY.reshape(-1, 1))
y_pred = regressor.predict(ActualX.reshape(-1, 1))
plt.scatter(in_trainX, in_trainY, color='g')
plt.plot(ActualX, y_pred, color='k')
plt.xlabel('Violent', fontsize=18)
plt.ylabel('Predict', fontsize=16)
plt.show()
plt.clf()

userVar = input("Your Input Number : ")
y_pred = regressor.predict([[float(userVar)]])
print('Predicted: ' + str(y_pred))
