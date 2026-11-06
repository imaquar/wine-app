Python 3.13.0

в терминале в папке wine_app выполнить команды:
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py

для модели используется st.cache_resource, потому что один объект пайплайна можно использовать повторно, для таблицы и вероятностей используется st.cache_data чтобы не читать и не вычислять их заново при изменении порога. без кэшиирования joblib.load вызывался бы при каждом действии
