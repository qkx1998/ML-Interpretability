'''
参考自https://zhuanlan.zhihu.com/p/83412330
'''
import shap
import xgboost

shap.initjs()  # notebook环境下，加载用于可视化的JS代码

# 我们先训练好一个XGBoost model
X,y = shap.datasets.boston()
model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)


#--------Explainer---------
#在SHAP中进行模型解释需要先创建一个explainer，SHAP支持很多类型的explainer(例如deep, gradient, kernel, linear, tree, sampling)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值


#---------Local Interper---------
#Local可解释性提供了预测的细节，侧重于解释单个预测是如何生成的。它可以帮助决策者信任模型，并且解释各个特征是如何影响模型单次的决策。

#可视化单个prediction的解释   如果不想用JS,传入matplotlib=True
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])

#基本值(base_value)是我们传入数据集上模型预测值的均值，可以通过自己计算来验证：y_base = explainer.expected_value
y_base = explainer.expected_value
pred = model.predict(xgboost.DMatrix(X))
print(y_base == pred.mean())

#对多个样本进行可视化解释
shap.force_plot(explainer.expected_value, shap_values, X)


#--------Global Interper------------
#Global可解释性：寻求理解模型的overall structure(总体结构)。

#summary plot为每个样本绘制其每个特征的SHAP值。
shap.summary_plot(shap_values, X)


#---------Feature Importance-----------
shap.summary_plot(shap_values, X, plot_type="bar")


#-----------interaction value---------------
shap_interaction_values = explainer.shap_interaction_values(X)
shap.summary_plot(shap_interaction_values, X)


#-------------dependence_plot----------------
shap.dependence_plot("RM", shap_values, X)

