from django.shortcuts import render
from rest_framework.decorators import api_view
from app import models
from django.http import JsonResponse
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.core.cache import cache


import io
from transformers import T5ForConditionalGeneration, T5Tokenizer

import requests
from requests.auth import HTTPBasicAuth
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from io import BytesIO
import pdfplumber
import pandas as pd
from collections import Counter
from nltk.util import ngrams
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

from django.views.decorators.csrf import csrf_exempt


'''
@csrf_exempt
@api_view(['GET', 'POST'])
def home(request):
    try:
        df_html = ""
        if request.method == 'POST':
            # Verificar si la solicitud es AJAX
            is_ajax = request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest'
            
            # Formato de datos de entrada
            TEXTO = str(request.POST.get('Texto'))
            pred = predecir(TEXTO)
            df_html = pred[['title', 'link']].to_html()

            if is_ajax:
                return JsonResponse({'df_html': df_html})
    # Se genera un error, solo para las solicitudes AJAX
    except Exception as e:
            if is_ajax:
                return JsonResponse({'error': str(e)}, status=400)  # retorna un error con status 400
    # La solicitud no es AJAX, renderizar la plantilla normalmente
    return render(request, "app/home.html", {'df_html': df_html})
'''
@csrf_exempt
@api_view(['GET', 'POST'])
def home(request):
    try:
        if request.method == 'POST':
            TEXTO = str(request.data.get('Texto'))
            pred = predecir(TEXTO)
            data = pred[['title', 'link']].to_dict(orient='records')

            return Response(data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
@csrf_exempt
@api_view(['GET', 'POST'])
def busquedaPdfs(request):
    try:
        if request.method == 'POST':
            query = str(request.data.get('Texto'))
            print("Buscando PDFs en mendeley de "+ query)
            n = 0
            c = 0
            client_id = "15721"
            client_secret = "zFMZriZ98BFUVWCX"
            auth = HTTPBasicAuth(client_id, client_secret)
            payload = {'grant_type': 'client_credentials', 'scope': 'all'}
            response = requests.post('https://api.mendeley.com/oauth/token', auth=auth, data=payload)
            response_data = response.json()
            access_token = response_data['access_token']
            headers = {'Authorization': 'Bearer ' + access_token, 'Accept': 'application/vnd.mendeley-document.1+json'}
            params = {'query': query, 'limit': 10}
            response = requests.get('https://api.mendeley.com/search/catalog', headers=headers, params=params)
            pdfs = []
            email = "edison32314@gmail.com"
            for doc in response.json():
                doi = doc.get('identifiers', {}).get('doi') if doc.get('identifiers') and 'doi' in doc['identifiers'] else None
                if doi:
                    response = requests.get(f'https://api.unpaywall.org/v2/{doi}?email={email}')
                    if response.status_code == 200:
                        data = response.json()
                        if data.get('best_oa_location'):
                            pdf_url = data.get('best_oa_location').get('url_for_pdf')
                            if pdf_url:
                                c = c + 1
                                source = doc.get('source')
                                source = source.get('title') if isinstance(source, dict) else source
                                document_info = {
                                    'title': doc.get('title'),
                                    'type': doc.get('type'),
                                    'year': doc.get('year'),
                                    'abstract': doc.get('abstract'),
                                    'source': source,
                                    'keywords': doc.get('keywords'),
                                    'pdf_link': pdf_url,
                                }
                                pdfs.append(document_info)
                            else:
                                n = n+1
                        else:
                            n = n+1
                    else:
                        n = n+1
            cache.set('pdfs_key', pdfs)
            data = "pdfs gratuitos encontrados:"+str(c)+", pdfs gratuitos no encontrados:"+str(n)
            return Response(data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    
@csrf_exempt
@api_view(['GET', 'POST'])
def busuedalinks(request):
    try:
        if request.method == 'POST':
            print("descargando PDFs, buscando Topicos y generando df")
            tam = len(str(request.data.get('Texto')).split())
            er = 0
            dataset = []
            pdfs = cache.get('pdfs_key')
            cache.delete('pdf_links')
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
            }
            for link in pdfs:
                try:
                    response = requests.get(link['pdf_link'], headers=headers)
                    response.raise_for_status()
                    with pdfplumber.open(BytesIO(response.content)) as pdf:
                        text = ' '.join(page.extract_text() for page in pdf.pages if page.extract_text() is not None)
                    keywords = extract_keywords(text, 10, tam)
                    for keyword in keywords:
                        data = {
                            'title': link['title'],
                            'type': link['type'],
                            'year': link['year'],
                            'keyword': keyword['keyword'],
                            'frecuencia': keyword['frec'],
                            'link': link['pdf_link']}
                        dataset.append(data)
                except Exception as e:
                    er = er + 1
            df = pd.DataFrame(dataset)
            print("links anti boots: ",er)
            df['type'] = df['type'].apply(lambda x: 1 if 'journal' in str(x) else 0)
            column = df['year']
            df['year'] = normalize_column(column)
            imp = np.array([])
            for index, row in df.iterrows():
                i = (row['type'] + row['year'] + row['frecuencia']) / 3
                imp = np.append(imp, i)
            df['importancia'] = imp
            le = LabelEncoder()
            df['title_encoded'] = le.fit_transform(df['title'])
            df['keyword_encoded'] = le.fit_transform(df['keyword'])
            scaler = MinMaxScaler()
            df[['title_encoded', 'keyword_encoded']] = scaler.fit_transform(df[['title_encoded', 'keyword_encoded']])
            print(df.head())
            cache.set('df_key', df)
            data = "links anti bots:"+str(er)
            return Response(data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['GET', 'POST'])
def entrenamoento(request):
    try:
        if request.method == 'GET':
            print("entrenando")
            df = cache.get('df_key')
            df_train = []
            df_test = []
            for i in range(0, len(df), 10):
                batch = df[i:i+10]
                df_train.append(batch[:8])
                df_test.append(batch[2:])
            df_train = pd.concat(df_train)
            df_test = pd.concat(df_test)
            features = ['type', 'year', 'frecuencia', 'title_encoded', 'keyword_encoded']
            target = ['importancia']
            X_train = create_sequences(df_train[features], sequence_length=2)
            X_test = create_sequences(df_test[features], sequence_length=2)
            y_train = create_sequences(df_train[target], sequence_length=2)
            y_test = create_sequences(df_test[target], sequence_length=2)
            model = Sequential()
            model.add(SimpleRNN(10, activation='relu', input_shape=(2, len(features))))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=10, verbose=0)
            cache.set('model_key', model)
            loss = model.evaluate(X_test, y_test, verbose=0)
            y_pred = model.predict(X_test)
            y_test_flat = np.reshape(y_test, (-1,))
            y_pred_flat = np.repeat(np.reshape(y_pred, (-1,)), 2)
            mse = np.mean(np.square(y_test_flat - y_pred_flat))
            mae = np.mean(np.abs(y_test_flat - y_pred_flat))
            rmse = np.sqrt(mse)
            print('Mean Squared Error (MSE):', mse)
            print('Mean Absolute Error (MAE):', mae)
            print('Root Mean Squared Error (RMSE):', rmse)
            data = 'Mean Squared Error (MSE):'+str(mse)+', Mean Absolute Error (MAE):'+str(mae)+', Root Mean Squared Error (RMSE):'+str(rmse)
            return Response(data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
    

@csrf_exempt
@api_view(['GET', 'POST'])
def predict(request):
    try:
        if request.method == 'POST':
            sequence_length=2
            top_n=4
            keyword = str(request.data.get('Texto'))
            df = cache.get('df_key')
            model = cache.get('model_key')
            keyword_data = df[df['keyword'] == keyword].copy()
            features = ['type', 'year', 'frecuencia', 'title_encoded', 'keyword_encoded']
            X = create_sequences(keyword_data[features], sequence_length)
            predictions = model.predict(X)
            flat_predictions = predictions.repeat(sequence_length, axis=0)[:len(keyword_data)]
            keyword_data['importance'] = flat_predictions
            top_rows = keyword_data.sort_values(by='importance', ascending=False).head(top_n)
            cache.set('res_key', top_rows)
            data = top_rows[['title', 'link']].to_dict(orient='records')
            return Response(data, status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)

@csrf_exempt
@api_view(['GET', 'POST'])
def resumen2(request):

    if request.method == 'GET':
        df_res = cache.get('res_key')
        df_summary = pd.DataFrame(columns=['title', 'summary', 'link'])
        for i, row in df_res.iterrows():
            pdf_url = row['link']
            pdf_title = row['title']
            pdf_text = read_pdf_from_url(pdf_url)
            pdf_summary = summarize(pdf_text)
            df_summary.loc[i] = [pdf_title, pdf_summary, pdf_url]
            data = df_summary.to_dict(orient='records')
        return Response(data, status=status.HTTP_200_OK)

    
def extract_keywords(text, num_keywords, ngram_length):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in nltk.corpus.stopwords.words('english') and word.isalnum() and word not in nltk.corpus.stopwords.words('spanish')]
    ngram_freq = Counter(ngrams(words, ngram_length))
    keywords_freq = ngram_freq.most_common(num_keywords)
    max_freq = keywords_freq[0][1] if keywords_freq else 1
    keywords = [{'keyword': ' '.join(words), 'frec': freq / max_freq} for words, freq in keywords_freq]
    return keywords

def normalize_column(column):
    min_val = np.min(column)
    max_val = np.max(column)
    normalized_column = (column - min_val) / (max_val - min_val)
    return normalized_column

def create_sequences(df, sequence_length):
    data = []
    for i in range(0, len(df), sequence_length):
        data.append(df[i:i+sequence_length].values)
    return pad_sequences(data, maxlen=sequence_length, dtype='float32', padding='post')


def read_pdf_from_url(pdf_link):
    print(pdf_link)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }
    response = requests.get(pdf_link, headers=headers)
    response.raise_for_status()
    with pdfplumber.open(BytesIO(response.content)) as pdf:
        text = ' '.join(page.extract_text() for page in pdf.pages if page.extract_text() is not None)
    return text

def summarize(text):
    tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)
    modelT5 = T5ForConditionalGeneration.from_pretrained('t5-base')
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    outputs = modelT5.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=8, early_stopping=True)
    return tokenizer.decode(outputs[0])

def generate_summary_dataframe(df_summary):
    summary_list = []

    for i, row in df_summary.iterrows():
        title = row['title']
        link = row['link']
        summary = row['summary']
        summary_list.append({'Title': title, 'Link': link, 'Summary': summary})

    summary_df = pd.DataFrame(summary_list)
    return summary_df