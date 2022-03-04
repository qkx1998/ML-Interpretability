'''
Permuted Importance：置换特征重要性
https://www.kaggle.com/dansbecker/permutation-importance
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import eli5
from eli5.sklearn import PermutationImportance

data = pd.read_csv('FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  

feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=100,random_state=0).fit(train_X, train_y)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

'''
图解：

朝向顶部的值是最重要的特征，而朝向顶部的值最不重要。

每行中的第一个数字显示随机洗牌（在本例中，使用"准确性"作为性能指标）时模型性能下降的程度。

与数据科学中的大多数内容一样，对列进行洗牌的确切性能变化存在一些随机性。
我们通过重复多个洗牌过程来衡量排列重要性计算中的随机性。
± 后面的数字衡量的是多次洗牌后性能的变化程度

您偶尔会看到排列重要性的负值。
在这些情况下，对随机（或嘈杂）数据的预测恰好比实际数据更准确。
当特征无关紧要（重要性应接近 0）时，就会发生这种情况，但随机机会导致对随机数据的预测更准确。
这在小型数据集中更为常见，如本例中的数据集所示，因为运气/机会的空间更大。
'''
