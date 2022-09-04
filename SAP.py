import streamlit as st
import pickle
import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.model_selection import cross_val_score
import random
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler

#应用主题
st.set_page_config(
    page_title="ML Medicine",
    page_icon="🐇",
)
#应用标题
st.title('Machine Learning Application for Predicting SAP')


col1, col2, col3 = st.columns(3)
SOS = col1.selectbox("Site of stroke lesion",('Cortex','Cortex-subcortex','Subcortex','Brainstem','Cerebellum'))
NSE = col2.number_input("NSE (ng/mL)",step=0.01,format="%.2f",value=1.45)
PPI = col3.selectbox("PPI",('No','Yes'))
Dysphagia = col1.selectbox("Dysphagia",('No','Yes'))
SS = col2.selectbox("Stroke severity",('Mild stroke','Moderate to severe stroke'))


# str_to_
map = {'Left':0,'Right':1,'Bilateral':2,
       'Single stroke lesion':0,'Multiple stroke lesions':1,
       'Mild stroke':0,'Moderate to severe stroke':1,
       'Cortex':0,'Cortex-subcortex':1,'Subcortex':2,'Brainstem':3,'Cerebellum':4,
       'No':0,'Yes':1}

SOS = map[SOS]
SS =map[SS]
PPI = map[PPI]
Dysphagia =map[Dysphagia]


# 数据读取，特征标注
thyroid_train = pd.read_csv('train.csv', low_memory=False)
features=['SOS', 'NSE','PPI',  'Dysphagia', 'SS']
target='SAP'

###
ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
X_ros, y_ros = ros.fit_resample(thyroid_train[features], thyroid_train[target])

#train and predict
RF = sklearn.ensemble.RandomForestClassifier(n_estimators=32,criterion='entropy',max_features='log2',max_depth=5,random_state=12)
RF.fit(X_ros, y_ros)

sp = 0.5
#figure
is_t = (RF.predict_proba(np.array([[ SOS, NSE,PPI,  Dysphagia, SS]]))[0][1])> sp
prob = (RF.predict_proba(np.array([[SOS, NSE,PPI,  Dysphagia, SS]]))[0][1])*1000//1/10


if is_t:
    result = 'High Risk'
else:
    result = 'Low Risk'
if st.button('Predict'):
    st.markdown('## Risk grouping:  '+str(result))
    if result == 'Low Risk':
        st.balloons()
    st.markdown('## Probability:  '+str(prob)+'%')
