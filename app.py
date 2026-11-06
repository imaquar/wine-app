import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn import metrics


appDir = Path(__file__).parent


@st.cache_resource
def loadModel():
    return joblib.load(appDir / 'model.joblib')


@st.cache_data
def loadTestData():
    return pd.read_csv(appDir / 'test.csv')


@st.cache_data
def getTestProb(_model, testX):
    return _model.predict_proba(testX)[:, 1]


model = loadModel()
testData = loadTestData()
features = testData.drop(columns='target')
config = json.loads((appDir / 'config.json').read_text())

threshold = st.sidebar.slider('порог', 0.0, 1.0, float(config['threshold']), 0.01)

tab1, tab2, tab3, tab4 = st.tabs(['Оценка одного вина', 'Модельные метрики', 'Пакетное предсказание', 'Коэффициенты'])
with tab1:
    values = {}
    for i in features.columns:
        if i == 'is_red':
            color = st.radio('Цвет вина', ['белое', 'красное'])
            if color == 'красное':
                values[i] = 1
            else:
                values[i] = 0
        else:
            values[i] = st.number_input(i, min_value=float(features[i].min()), max_value=float(features[i].max()), value=float(features[i].median()))

    wine = pd.DataFrame([values], columns=features.columns)
    prob = model.predict_proba(wine)[0, 1]
    st.write('Вероятность хорошего вина', prob)
    if prob >= threshold:
        st.write('Рекомендовать')
    else:
        st.write('Не рекомендовать')

with tab2:
    yTrue = testData['target']
    testProb = getTestProb(model, features)
    testPred = (testProb >= threshold).astype(int)

    matrix = metrics.confusion_matrix(yTrue, testPred)
    precision = metrics.precision_score(yTrue, testPred, zero_division=0)
    recall = metrics.recall_score(yTrue, testPred)
    f1 = metrics.f1_score(yTrue, testPred)

    st.write('Матрица ошибок')
    st.write(pd.DataFrame(matrix, index=['истинный 0', 'истинный 1'], columns=['предсказанный 0', 'предсказанный 1']))
    st.write('precision:', precision)
    st.write('recall:', recall)
    st.write('f1:', f1)

    fpr, tpr, x = metrics.roc_curve(yTrue, testProb)
    fprPoint = matrix[0, 1] / (matrix[0, 0] + matrix[0, 1])
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    ax.scatter(fprPoint, recall)
    ax.set_xlabel('fpr')
    ax.set_ylabel('tpr')
    st.write('roc кривая')
    st.pyplot(fig)
    plt.close(fig)

    precisionCurve, recallCurve, x = metrics.precision_recall_curve(yTrue, testProb)
    fig, ax = plt.subplots()
    ax.plot(recallCurve, precisionCurve)
    ax.scatter(recall, precision)
    ax.set_xlabel('recall')
    ax.set_ylabel('precision')
    st.write('pr кривая')
    st.pyplot(fig)
    plt.close(fig)

    st.write('если увеличить порог, рекомендаций будет меньше и часть хороших вин можно пропустить, если уменьшить, рекомендаций будет больше, но среди них может оказаться больше посредственных вин')

with tab3:
    uploadedFile = st.file_uploader('csv файл', type='csv')
    if uploadedFile is not None:
        batch = pd.read_csv(uploadedFile)
        missing = []
        for i in features.columns:
            if i not in batch.columns:
                missing.append(i)
        if missing:
            st.error('Не хватает столбцов')
        else:
            batchProb = model.predict_proba(batch[features.columns])[:, 1]
            result = batch.copy()
            result['вероятность'] = batchProb
            decisions = []
            for i in batchProb:
                if i >= threshold:
                    decisions.append('рекомендовать')
                else:
                    decisions.append('не рекомендовать')
            result['решение'] = decisions
            st.dataframe(result)
            st.download_button('Скачать результат', result.to_csv(index=False).encode('utf-8-sig'), file_name='predictions.csv', mime='text/csv')

with tab4:
    coefs = pd.Series(model.named_steps['logisticregression'].coef_[0], index=features.columns).sort_values()
    fig, ax = plt.subplots()
    coefs.plot.barh(ax=ax)
    ax.set_xlabel('коэффициент')
    st.pyplot(fig)
    plt.close(fig)
    st.write('сильнее всего за хорошее вино:', coefs.idxmax())
    st.write('сильнее всего против:', coefs.idxmin())
