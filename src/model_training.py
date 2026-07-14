"""
数据科学家薪资预测 - 模型训练模块
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
# ==================== 1. 加载处理后的数据 ====================
print("正在加载处理后的数据...")
df = pd.read_csv('../data/processed_data.csv')
print(f"数据形状: {df.shape}")

# ==================== 2. 分离特征和目标变量 ====================
target_col = 'avg_salary_k'
X = df.drop(target_col, axis=1)
y = df[target_col]

print(f"特征列: {X.columns.tolist()}")

# ==================== 3. 处理分类变量（标签编码） ====================
print("\n正在处理分类变量...")
label_encoders = {}
for col in X.columns:
    if X[col].dtype == 'object':
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
        print(f"  编码列: {col}")

# ==================== 4. 划分训练集和测试集 ====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\n训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# ==================== 5. 定义模型 ====================
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# ==================== 6. 训练和评估模型 ====================
print("\n==================== 模型评估结果 ====================")
results = {}
best_model = None
best_score = -999

for name, model in models.items():
    # 训练
    model.fit(X_train, y_train)
    # 预测
    y_pred = model.predict(X_test)
    # 评估
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print(f"\n{name}:")
    print(f"  MAE:  {mae:.2f} K美元")
    print(f"  RMSE: {rmse:.2f} K美元")
    print(f"  R2:   {r2:.4f}")

    if r2 > best_score:
        best_score = r2
        best_model = model
        best_model_name = name

# ==================== 7. 最佳模型调优 ====================
print(f"\n==================== 最佳模型: {best_model_name} ====================")

if best_model_name == 'Random Forest':
    print("正在对 Random Forest 进行网格搜索调优...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(
        RandomForestRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")

elif best_model_name == 'Gradient Boosting':
    print("正在对 Gradient Boosting 进行网格搜索调优...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1]
    }
    grid_search = GridSearchCV(
        GradientBoostingRegressor(random_state=42),
        param_grid,
        cv=3,
        scoring='r2',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    print(f"最佳参数: {grid_search.best_params_}")

# ==================== 8. 最终模型评估 ====================
y_pred_final = best_model.predict(X_test)
final_r2 = r2_score(y_test, y_pred_final)
final_mae = mean_absolute_error(y_test, y_pred_final)
final_rmse = np.sqrt(mean_squared_error(y_test, y_pred_final))

print(f"\n最终模型性能（调优后）:")
print(f"  MAE:  {final_mae:.2f} K美元")
print(f"  RMSE: {final_rmse:.2f} K美元")
print(f"  R2:   {final_r2:.4f}")

# ==================== 9. 特征重要性分析 ====================
print("\n正在生成特征重要性...")
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 重要特征:")
    print(feature_importance.head(10).to_string(index=False))

    # 保存特征重要性图
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(top_features['feature'][::-1], top_features['importance'][::-1])
    plt.xlabel('Importance')
    plt.title(f'Top 10 Feature Importances ({best_model_name})')
    plt.tight_layout()
    plt.savefig('../models/feature_importance.png', dpi=150)
    print("特征重要性图已保存到: ../models/feature_importance.png")

# ==================== 10. 保存模型 ====================
model_path = '../models/salary_predictor_model.pkl'
joblib.dump(best_model, model_path)
print(f"\n模型已保存到: {model_path}")

# 同时保存标签编码器
joblib.dump(label_encoders, '../models/label_encoders.pkl')
print("标签编码器已保存到: ../models/label_encoders.pkl")

# 保存训练时用到的特征列名
joblib.dump(X.columns.tolist(), '../models/feature_columns.pkl')
print("特征列名已保存到: ../models/feature_columns.pkl")

print("\n==================== 模型训练完成！ ====================")
print("可以运行 app.py 启动 Web 应用")