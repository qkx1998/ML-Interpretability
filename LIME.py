'''
LIME: Local Interpretable Model-agnostic Explanations
'''
import numpy as np
import lime
import lime.lime_tabular
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('FIFA 2018 Statistics.csv')

feats = [i for i in data.columns if data[i].dtype in [np.int64]]
x = np.array(data[feats].fillna(-99999))
y = (data['Man of the Match'] == "Yes")  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 400)

model_xgb = xgb.XGBClassifier(
                            learning_rate =0.05,
                             n_estimators=50,
                             max_depth=3,
                             min_child_weight=1
                            ).fit(X_train, y_train)

# 生成解释器
explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=feats,
                                                   class_names=[0,1], discretize_continuous=True)
# 对局部点的解释
i = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test[i], model_xgb.predict_proba, num_features=6)

# 显示详细信息图
exp.show_in_notebook(show_table=True, show_all=True)

# 显示权重图
exp.as_pyplot_figure()

