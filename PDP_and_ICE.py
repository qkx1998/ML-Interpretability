'''
参考自：
https://zhuanlan.zhihu.com/p/435528676
https://www.zhihu.com/search?type=content&q=Partial%20Dependence%20Plots

Partial Dependence Plot：部份依赖关系图，简称PDP，能够展现出一个或两个特征变量对模型预测结果影响的函数关系
Individual Conditional Expectation：个体条件期望，简称ICE，刻画的是每个个体的预测值与单一变量之间的关系

PDP优点：易实施
PDP缺点：不能反映特征变量本身的分布情况。需要假设变量之间严格独立。另外的缺点是样本整体的非均匀效应。

ICE优点：易于理解，能够避免数据异质的问题
ICE缺点：只能反映单一特征变量与目标之间的关系，仍然受制于变量独立假设的要求

特征选择：
当某个特征的PDP曲线几乎水平或者无规律抖动的时候, 这个特征可能是无用的特征
当某个特征的PDP曲线非常陡峭的时候, 说明这个特征的贡献度是比较大的
'''

# -----------sklearn版本要0.24+才可以计算ICE----------------
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.inspection import partial_dependence
from sklearn.inspection import plot_partial_dependence
import matplotlib.pyplot as plt
%matplotlib inline

cal_housing = fetch_california_housing()
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names)
y = cal_housing.target
y -= y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.neural_network import MLPRegressor

model = make_pipeline(QuantileTransformer(),
                      MLPRegressor(hidden_layer_sizes=(50, 50),learning_rate_init=0.01,early_stopping=True))
                    
model.fit(X_train, y_train)

#------------部分依赖图（Partial Dependence Plot)----------------
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(model, X_train, features, kind="average", 
                                  subsample=50,n_jobs=3, grid_resolution=20, random_state=0)


##-----------2D interaction plots------------------
features = ['AveOccup', 'HouseAge', ('AveOccup', 'HouseAge')]
fig, ax = plt.subplots(ncols=3, figsize=(9, 4))
display = plot_partial_dependence(model, X_train, features, kind='average',
                                  n_jobs=3, grid_resolution=20, ax=ax)


#-------------个体条件期望图（Individual Conditional Expectation Plot)-----------------
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(model, X_train, features, kind="individual",
                                  subsample=50,n_jobs=3, grid_resolution=20, random_state=0)


#-------------both = PDP + ICE-----------------
features = ['MedInc', 'AveOccup', 'HouseAge', 'AveRooms']
display = plot_partial_dependence(model, X_train, features, kind="both",
                                  subsample=50,n_jobs=3, grid_resolution=20, random_state=0)
                                  
                                  
