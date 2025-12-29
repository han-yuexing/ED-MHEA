# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
df = pd.read_excel('/Users/macaroni/Downloads/pythonProject1/doi_txt/material_compositions.xlsx')

# 预处理函数，只保留±前的数值
def preprocess_value(value):
    # 检查值是否为空或NaN
    if pd.isnull(value) or value == '':
        return 0  # 返回np.nan表示缺失值
    if isinstance(value, str):
        # 如果值包含'±'，只保留前面的数值部分
        value = value.split('±')[0].strip()
        if value:  # 再次检查分割后的字符串是否为空
            return float(value)
        else:
            return 0
    return value

# 应用预处理函数到DataFrame的每个单元格
df_preprocessed = df.applymap(preprocess_value)


# 分离特征和目标变量
X = df_preprocessed.iloc[:-1, :].T  # 特征
y = df_preprocessed.iloc[-1, :].values  # 目标变量
#
#
# #-----------------------初始化随机森林回归器----------------------------------------------------------------
# # 初始化随机森林回归器
# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
#
# # 5折交叉验证
# kf = KFold(n_splits=4, shuffle=True, random_state=42)
#
# # 计算MAE和R2
# mae_scores = cross_val_score(rf_regressor, X, y, cv=kf, scoring='neg_mean_absolute_error')
# r2_scores = cross_val_score(rf_regressor, X, y, cv=kf, scoring='r2')
#
# # 输出MAE和R2的平均值和标准差
# mae_mean, mae_std = -mae_scores.mean(), mae_scores.std()
# r2_mean, r2_std = r2_scores.mean(), r2_scores.std()
#
# mae_mean, mae_std, r2_mean, r2_std
# print(mae_mean, mae_std, r2_mean, r2_std)

#-----------------------SVM---------------------------------------------------------------

from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
import pandas as pd

# 初始化SVR模型
svr = SVR(kernel='rbf')

# 进行5折交叉验证
cv_results = cross_validate(svr, X, y, cv=4, scoring=('r2', 'neg_mean_absolute_error'))

# 获取MAE和R^2分数
mae_scores = -cv_results['test_neg_mean_absolute_error']
r2_scores = cv_results['test_r2']

# 计算分数的平均值和标准差
mae_mean = mae_scores.mean()
mae_std = mae_scores.std()
r2_mean = r2_scores.mean()
r2_std = r2_scores.std()

print(f"MAE: Mean = {mae_mean}, STD = {mae_std}")
print(f"R^2: Mean = {r2_mean}, STD = {r2_std}")
