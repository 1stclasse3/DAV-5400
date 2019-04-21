import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import preprocessing
import matplotlib.pyplot as plt

df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data", header=None)

'''
df는 pd.read_csv를 통하여 받은 데이터이고 각 열에 대한 이름을 바꾸기 위해서 rename을 사용하였다
0으로 지정된 열은 poison으로 1로 지정된 열은 cap-shape 등으로 바꿔줄 수 있는 코드이다
자세한 내용은 아래 사이트에서 확인할 수 있다
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html
'''

df = df.rename(columns={0:"poison", 1:"cap-shape", 2:"ap-surface", 3:"cap-color", 4:"bruises", 5:"odor", 6:"gill-attachment",
                        7:"gill-spacing", 8:"gill-size", 9:"gill-color", 10:"stalk-shape", 11:"stalk-root", 12:"stalk-surface-above-ring",
                        13:"stalk-surface-below-ring", 14:"stalk-color-above-ring", 15:"stalk-color-below-ring", 16:"veil-type",
                        17:"veil-color", 18:"ring-number", 19:"ring-type", 20:"spore-print-color", 21:"population", 22:"habitat"})

'''
column_list를 만들었고 이를 columns로 하는 빈 dataframe인 result를 미리 만들어주었다 
'''

column_list = ["cap-shape", "ap-surface", "cap-color", "bruises", "odor", "gill-attachment", "gill-spacing", "gill-size", "gill-color",
               "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring", "stalk-color-above-ring",
               "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color", "population", "habitat"]
result = pd.DataFrame(columns=column_list)

'''
각 data의 값이 문자로 되어있고 이는 data analysis에서 사용하기 위해서는 각 데이터를 숫자형태로 바꿔야한다
이에 One hot encoding과 Label encoding이 있는데 Feature의 영향력을 확인하기 위해서는 Label encoding을 사용한다
각 data의 값은 아래의 코드를 이용하여 숫자로 바꿀 수 있다
Label encoding에 대한 코드는 아래 주소에서 찾을 수 있었다
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
'''

df = df.apply(preprocessing.LabelEncoder().fit_transform)

'''
DataFrame을 두 개로 나눌 때 사용할 수 있는 코드이다
df는 나누고 싶은 데이터이고 [1]은 나누고 싶어하는 지점이다
만약 [1]이 아니라 [2, 6]라고 적는다면 2번째 열, 6번째 열에서 나누는 것이고 총 3개의 DataFrame으로 나누어진다
poison이라는 첫 열은 결과물이므로 나눠서 X, y를 설정할 필요가 있다
따라서 [1]을 써서 첫 번째 열을 기준으로 나누었고 열을 기준으로 나눈다는 표시로 axis=1을 설정하였다
자세한 내용은 아래 사이트에서 볼 수 있다
https://docs.scipy.org/doc/numpy/reference/generated/numpy.split.html
'''

dfs = np.split(df, [1], axis=1)

X, y = dfs[1], dfs[0]

'''
ExtraTreesClassifier이라는 Random Forest를 이용하는 함수이다
n_estimators는 Random Forest에서 사용할 Tree의 개수를 나타낸다
clf에 우리의 데이터를 fitting시키고 여기서 importance를 추출한다면 각 feature에 대한 중요도를 알 수 있다
importances는 각 feature의 중요도 값을 나타내고 indices는 그 값의 순위를 매칭시키는 것이다
그리고 importances의 값은 result.loc을 이용하여 result라는 DataFrame에 삽입한다
아래의 사이트에서 ExtraTreesClassifier을 살펴볼 수 있다
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
'''

clf = ExtraTreesClassifier(n_estimators=100)
clf = clf.fit(X, y)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

result.loc['Classifier_importances',:] = importances

'''
그림으로 보여주기 위하여 추가로 작성한 코드이다
title로 이름을 설정하고 bar그래프를 이용하여 각 feature가 어떤 값을 가지는지 그림으로 표현하였다
xticks와 xlim은 x축의 단위값을 1단위로 표현하고자 하는 형식적인 코드이다
'''

plt.figure()
plt.title("Feature importances_Classifier")
plt.bar(range(X.shape[1]), importances[indices], color='r')
plt.xticks(range(X.shape[1]), indices+1)
plt.xlim([-1, X.shape[1]])
plt.show()

'''
코드의 내용은 위와 동일하고 Classifier만 하는 것보다 Regressor 또한 보는 것이 좋다고 생각하여 추가하였다
아래의 사이트를 통하여 Regressor을 확인할 수 있고 이하 나머지 내용은 모두 동일하다
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html
'''

clf1 = ExtraTreesRegressor(n_estimators=100)
clf1 = clf1.fit(X, y)
importances = clf1.feature_importances_
indices = np.argsort(importances)[::-1]

result.loc['Regressor_importances',:] = importances

plt.figure()
plt.title("Feature importances_Regressor")
plt.bar(range(X.shape[1]), importances[indices], color='r')
plt.xticks(range(X.shape[1]), indices+1)
plt.xlim([-1, X.shape[1]])
plt.show()

'''
result값을 csv파일로 만들어 확인할 수 있도록 하였다
'''

result.to_csv("result_table.csv")