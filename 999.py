import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
# 设置页面标题
st.set_page_config(page_title="🙌学生成绩分析与预测系统", layout="wide")

# 左侧导航栏
page = st.sidebar.selectbox("导航", ["项目介绍", "专业数据分析", "成绩预测"])

# 主内容区
if page == "项目介绍":
    st.title("🐥学生成绩分析与预测系统")
    st.markdown('***')

    c1, c2 = st.columns(2)

    with c1:
        st.header("🗽项目概述")
        st.write("本项目是一个基于Streamlit的学生成绩分析平台，通过数据可视化技术和机器学习，帮助教育工作者和小牛马学生深入了解学业表现，并且得到满分的期末考试成绩。")

        st.header("🩲主要特点：")
        st.markdown("""
        - 🍗数据可视化: 多维度展示学生学业数据
        - 🧈专业分析: 按专业分类的详细统计分析
        - 🍹智能预测: 基于机器学习模型的成绩预测
        - 🏛学习建议: 根据预测结果提供个性化反馈
        """)
    with c2:
        images = ['123.png',
          '1234.png',
          '12345.png']

        captions = ['专业数据分析', '成绩预测', '成绩预测']
        if 'a' not in st.session_state:
            st.session_state['a'] = 0
    
        def nextimg():
            st.session_state['a'] =(st.session_state['a']+1) % len(images)
    
  
        def next2mg():
            st.session_state['a'] =(st.session_state['a']-1) % len(images)


        # st.image()总共两个参数，url：图片地址 caption:图片的备注
        st.image(images[st.session_state['a']], captions[st.session_state['a']])



        t1, t2 = st.columns(2)
        with t1:
            st.button('上一张', on_click=next2mg, use_container_width=True)

        with t2:
            st.button('下一张', on_click=nextimg, use_container_width=True)



    st.markdown('***')
    st.header("👶项目目标")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.header("🤦‍♂️目标一")
        st.markdown("""
        分析影响因素
        - 识别关键学习指标
        - 探索成绩相关因素
        - 提供数据支持决策
        """)

    with b2:
        st.header("😺目标二")
        st.markdown("""
        可视化展示
        - 专业对比分析
        - 性别差异研究
        - 学习模式识别
        """)

    with b3:
        st.header("🦸‍♂️目标三")
        st.markdown("""
        成绩预测
        - 机器学习模型
        - 个性化预测
        - 及时干预预警
        """)

    st.markdown('***')
    st.subheader('👯‍♀️技术架构')
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.text('数据处理')
        python_code = 'Streamlit'
        st.code(python_code, language=None)

    with a2:
        st.text('可视化')
        python_code = 'pandas \nNumPy'
        st.code(python_code, language=None)        

    with a3:
        st.text('机器学习')
        python_code = 'Plotly \nMatolotlib'
        st.code(python_code, language=None)

    with a4:
        st.text('前端框架')
        python_code = 'Scikit-learn'
        st.code(python_code, language=None)

elif page == "专业数据分析":
    
    st.subheader('👯‍♀️专业数据分析')
    st.markdown('***')

    st.set_page_config(page_title='成绩分析预测', page_icon='🐒', layout='wide')

# 加载数据集
    df = pd.read_csv('student_data_adjusted_rounded.csv')

    st.title('各专业平均期末考试分数')
    c1, c2 = st.columns(2)
    with c1:
        # 按专业分组计算平均期末考试分数，并保留两位小数
        grouped_data = df.groupby('专业')['期末考试分数'].mean().round(2).reset_index()
        # 将数据转换为适合 st.bar_chart 的格式
        bar_chart_data = grouped_data.set_index('专业')['期末考试分数']
        # 使用 st.bar_chart 展示柱状图
        st.bar_chart(bar_chart_data)
    with c2:
        # 将表格横向展示并在 Streamlit 中显示
        st.write('各专业平均期末考试分数：')
        st.dataframe(grouped_data.set_index('专业').T)


    st.title('各专业男女比例')
    a1, a2 = st.columns(2)
    with a1:
        # 按专业和性别分组统计人数
        grouped_data = df.groupby(['专业', '性别'])['学号'].count().reset_index(name='人数')
        # 计算各专业男女比例
        total_by_major = grouped_data.groupby('专业')['人数'].transform('sum')
        grouped_data['比例'] = (grouped_data['人数'] / total_by_major * 100).round(2)
        # 将数据转换为适合 st.bar_chart 的格式
        pivot_data = grouped_data.pivot(index='专业', columns='性别', values='比例')
        # 使用 st.bar_chart 展示柱状图
        st.bar_chart(pivot_data, use_container_width=True)
    with a2:
        # 展示各专业男女比例表格
        st.write('各专业男女比例：')
        st.dataframe(pivot_data)


    st.title('各专业平均每周学习时长（按期末分数排序）')


    d1, d2 = st.columns(2)
    with d1:
# 添加图表大小调节滑块
             col1, col2 = st.columns(2)
             with col1:
                 width = st.slider('图表宽度', min_value=3, max_value=12, value=10, step=1)
             with col2:
                 height = st.slider('图表高度', min_value=3, max_value=4, value=6, step=1)

             plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'Heiti TC', 'SimHei']
             plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
             plt.rcParams['figure.dpi'] = 300

# 计算各专业平均期末考试分数（用于排序）
             core_data = df.groupby('专业')['期末考试分数'].mean().reset_index()
# 计算各专业平均每周学习时长
             study_data = df.groupby('专业')['每周学习时长（小时）'].mean().round(1).reset_index()

# 合并数据并按期末分数排序
             merged_data = pd.merge(study_data, core_data, on='专业')
# 按期末分数升序/降序排序（这里用降序，可根据需要改为ascending=True）
             merged_data = merged_data.sort_values(by='期末考试分数', ascending=False)

# 创建图表（使用滑块值作为尺寸）
             plt.figure(figsize=(width, height))

# 按排序后的顺序绘制柱状图
             bars = plt.bar(merged_data['专业'], merged_data['每周学习时长（小时）'], color='#4A90E2', alpha=0.7)

# 设置标题和标签
             plt.title('各专业平均每周学习时长（按期末分数从高到低排序）')
             plt.xlabel('专业')
             plt.ylabel('每周学习时长（小时）')
             plt.xticks(rotation=45, ha='right')

# 在柱状图上显示数值
             for bar in bars:
                 height_val = bar.get_height()
                 plt.text(bar.get_x() + bar.get_width() / 2, height_val, f'{height_val}', 
                          ha='center', va='bottom', fontsize=10)

             plt.tight_layout()
             st.pyplot(plt)  # 在Streamlit中显示图表
    with d2:
# 在Streamlit中显示表格（按期末分数排序）
             st.write('各专业平均每周学习时长（按期末分数排序）：')
             st.dataframe(merged_data[['专业', '每周学习时长（小时）', '期末考试分数']].set_index('专业'))


# 计算各专业平均每周学习时长
    study_data = df.groupby('专业')['每周学习时长（小时）'].mean().reset_index()

# 页面标题
    st.title('各专业平均每周学习时长分析')

# 创建两列布局
    col1, col2 = st.columns(2)

    with col1:
        st.write('各专业平均每周学习时长柱状图')
    # 绘制柱状图
        plt.figure(figsize=(10, 4))
        plt.bar(study_data['专业'], study_data['每周学习时长（小时）'])
        plt.xlabel('专业')
        plt.ylabel('平均每周学习时长（小时）')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    with col2:
        st.write('各专业平均每周学习时长数据')
    # 显示数据表格
        st.dataframe(study_data)


elif page == "成绩预测":
    

    st.set_page_config(
        page_title="期末成绩预测",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # 深色主题配置
    st.markdown("""
    <style>
        .main {
            background-color: #121212;
            color: white;
        }
        .stTextInput, .stSelectbox, .stNumberInput {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton>button {
            background-color: #e63946;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
        }
        .stProgress>div>div {
            background-color: #4cc9f0;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # 标题
    st.title("📊 期末成绩预测")
    st.markdown("---")
    st.write("请输入学生的学习信息，系统将预测其期末成绩并提供学习建议")

# 加载并预处理数据
    @st.cache_data
    def load_data():
        # 加载数据（使用之前的学生数据）
        df = pd.read_csv('student_data_adjusted_rounded.csv')
        
        # 选择特征和目标变量
        features = ['每周学习时长（小时）', '上课出勤率', '期中考试分数', '作业完成率']
        target = '期末考试分数'
    
        # 划分训练集和测试集
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
    
    # 评估模型
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
    
        return model, features, mae

# 加载模型
    model, features, mae = load_data()

# 创建输入表单
    col1, col2 = st.columns(2)

    with col1:
        student_id = st.text_input("学号", "23311321")
        gender = st.selectbox("性别", ["男", "女"])
        major = st.selectbox("专业", ["人工智能", "信息系统", "大数据管理与应用", "网络安全", "计算机科学", "软件工程"])

    with col2:
        study_hours = st.number_input("每周学习时长（小时）", min_value=0.0, max_value=50.0, value=25.0, step=0.1)
        attendance = st.number_input("上课出勤率", min_value=0.0, max_value=1.0, value=0.9, step=0.01)
        midterm_score = st.number_input("期中考试分数", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
        homework_rate = st.number_input("作业完成率", min_value=0.0, max_value=1.0, value=0.85, step=0.01)

# 预测按钮
    if st.button("预测期末成绩", key="predict_btn"):
    # 准备输入数据
        input_data = np.array([[study_hours, attendance, midterm_score, homework_rate]])
    
    # 预测成绩
        final_score = model.predict(input_data)[0]
        final_score = round(final_score, 1)
    
    # 显示预测结果
        st.markdown("---")
        st.subheader("预测结果")
    
    # 进度条展示
        st.write(f"预测期末成绩: {final_score}分")
        progress = min(100, max(0, int(final_score)))
        st.progress(progress)
    
    # 结果图片和反馈
        col1, col2 = st.columns([3, 1])
        with col1:
            if final_score >= 85:
                st.image("321.png", caption="恭喜！你的成绩非常优秀！")
                st.success("学习建议：保持当前的学习状态，继续挑战更高难度的内容")
            elif final_score >= 60:
                st.image("4321.png", caption="不错的成绩，继续努力！")
                st.info("学习建议：可以增加每周学习时间，提高作业完成质量")
            else:
                st.image("54321.png", caption="需要更加努力哦！")
                st.warning("学习建议：建议增加学习时间，提高出勤率，及时完成作业")
  




    
