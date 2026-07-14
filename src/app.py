"""
数据科学家薪资预测 - Streamlit Web 应用
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ==================== 页面配置 ====================
st.set_page_config(
    page_title="数据科学家薪资预测",
    page_icon="💰",
    layout="wide"
)


# ==================== 加载模型 ====================
@st.cache_resource
def load_model():
    model = joblib.load('../models/salary_predictor_model.pkl')
    label_encoders = joblib.load('../models/label_encoders.pkl')
    feature_columns = joblib.load('../models/feature_columns.pkl')
    return model, label_encoders, feature_columns


try:
    model, label_encoders, feature_columns = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("模型文件未找到！请先运行 model_training.py 训练模型。")

# ==================== 标题 ====================
st.title("💰 数据科学家薪资预测系统")
st.markdown("---")

# ==================== 侧边栏 ====================
with st.sidebar:
    st.header("📊 数据说明")
    st.info("""
    **数据集**: Glassdoor 数据科学家职位数据

    **模型**: 基于随机森林/梯度提升的回归模型

    **预测目标**: 年薪（千美元/年）

    **说明**: 预测值仅供参考，实际薪资受多种因素影响。
    """)

    st.markdown("---")
    st.subheader("📂 项目信息")
    st.markdown("""
    - **项目**: Data Scientist Salary Predictor
    - **作者**: [Goodstudy-max962]
    - **技术栈**: Python + Scikit-learn + Streamlit
    - **GitHub**: [https://github.com/Goodstudy-max962/data-science-salary-predictor.git]
    """)

# ==================== 主界面 ====================
if model_loaded:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 请输入职位信息")

        # 用户输入
        job_state = st.selectbox(
            "工作地点（州）",
            ['CA', 'NY', 'TX', 'WA', 'MA', 'IL', 'VA', 'NC', 'NJ', 'Other']
        )

        seniority = st.selectbox(
            "职位级别",
            ['na', 'jr', 'sr', 'manager', 'director']
        )

        size = st.selectbox(
            "公司规模",
            ['1 to 50 employees', '51 to 200 employees', '201 to 500 employees',
             '501 to 1000 employees', '1001 to 5000 employees', '5001+ employees', 'Unknown']
        )

        type_of_ownership = st.selectbox(
            "公司类型",
            ['Company - Private', 'Company - Public', 'Non-profit Organization',
             'Government', 'Unknown']
        )

        industry = st.selectbox(
            "行业",
            ['IT Services', 'Finance', 'Healthcare', 'Education', 'Retail',
             'Manufacturing', 'Media', 'Other', 'Unknown']
        )

    with col2:
        st.subheader("🛠 技能标签")

        python_yn = st.checkbox("Python", value=True)
        R_yn = st.checkbox("R")
        spark = st.checkbox("Spark")
        aws = st.checkbox("AWS")
        excel = st.checkbox("Excel")
        sql = st.checkbox("SQL", value=True)
        tableau = st.checkbox("Tableau")
        hadoop = st.checkbox("Hadoop")
        azure = st.checkbox("Azure")

        st.subheader("📈 其他信息")
        rating = st.slider("公司评分", 0.0, 5.0, 3.5, 0.1)
        age = st.number_input("公司成立年数", 0, 200, 20)

    # ==================== 预测按钮 ====================
    st.markdown("---")
    predict_btn = st.button("🚀 预测薪资", type="primary", use_container_width=True)

    if predict_btn:
        # 构建输入数据
        input_data = {
            'job_state': job_state,
            'seniority': seniority,
            'Size': size,
            'Type of ownership': type_of_ownership,
            'Industry': industry,
            'python_yn': int(python_yn),
            'R_yn': int(R_yn),
            'spark': int(spark),
            'aws': int(aws),
            'excel': int(excel),
            'skill_sql': int(sql),
            'skill_tableau': int(tableau),
            'skill_hadoop': int(hadoop),
            'skill_azure': int(azure),
            'Rating': rating,
            'age': age,
        }

        # 填充默认值
        for col in feature_columns:
            if col not in input_data:
                input_data[col] = 0

        # 按顺序构建 DataFrame
        input_df = pd.DataFrame([input_data])[feature_columns]

        # 处理分类变量
        for col in input_df.columns:
            if col in label_encoders:
                try:
                    input_df[col] = label_encoders[col].transform(input_df[col].astype(str))
                except ValueError:
                    input_df[col] = 0

        # 预测
        prediction = model.predict(input_df)[0]

        # 显示结果
        st.markdown("---")
        st.subheader("📊 预测结果")

        col_result1, col_result2, col_result3 = st.columns(3)

        with col_result1:
            st.metric(
                label="预测年薪",
                value=f"${prediction:.0f}K",
                delta=None
            )

        with col_result2:
            st.metric(
                label="预测月薪",
                value=f"${prediction / 12:.1f}K",
                delta=None
            )

        with col_result3:
            st.metric(
                label="预测年薪（人民币）",
                value=f"¥{prediction * 7.2 / 10:.1f}万",
                delta=None
            )

        # 薪资范围
        st.info(f"""
        💡 **预测解读**:
        - 预测薪资范围: **${prediction - 15:.0f}K - ${prediction + 15:.0f}K** /年
        - 该预测基于 Glassdoor 数据集的机器学习模型
        - 实际薪资可能因公司、谈判能力、市场行情等因素有所浮动
        """)

    # ==================== 特征重要性展示 ====================
    st.markdown("---")
    st.subheader("📈 模型特征重要性")

    importance_path = '../models/feature_importance.png'
    if os.path.exists(importance_path):
        st.image(importance_path, use_column_width=True)
    else:
        st.info("特征重要性图将在模型训练后显示。运行 model_training.py 即可生成。")

else:
    st.warning("⚠️ 请先运行 model_training.py 训练模型，然后刷新此页面。")
    st.markdown("""
    ### 操作步骤：
    1. 确保 `data/processed_data.csv` 存在（运行 `data_preprocessing.py`）
    2. 运行 `src/model_training.py` 训练模型
    3. 刷新此页面
    """)