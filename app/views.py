from django.shortcuts import render
from rest_framework.decorators import api_view
from app import models
from django.http import JsonResponse


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

def get_pdf_links(query, limit):
    n = 0
    c = 0
    # Tus credenciales
    client_id = "15721"
    client_secret = "zFMZriZ98BFUVWCX"

    # Autenticación
    auth = HTTPBasicAuth(client_id, client_secret)
    payload = {'grant_type': 'client_credentials', 'scope': 'all'}

    # Solicitar token de acceso
    response = requests.post('https://api.mendeley.com/oauth/token', auth=auth, data=payload)
    response_data = response.json()
    access_token = response_data['access_token']

    # Usar el token para hacer una petición de búsqueda
    headers = {'Authorization': 'Bearer ' + access_token, 'Accept': 'application/vnd.mendeley-document.1+json'}
    params = {'query': query, 'limit': limit}

    response = requests.get('https://api.mendeley.com/search/catalog', headers=headers, params=params)

    # Lista para almacenar los enlaces a los PDFs
    pdf_links = []

    # Tu correo para la API de Unpaywall
    email = "edison32314@gmail.com"  # Reemplaza esto con tu correo

    # Extraer los DOIs de los resultados y buscar los archivos PDF correspondientes a los DOIs
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
                        pdf_links.append(document_info)
                    else:
                        n = n+1
                else:
                     n = n+1
            else:
                 n = n+1
    
    return pdf_links, n ,c

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
    """
    This function creates sequences from the data.
    """
    data = []
    for i in range(0, len(df), sequence_length):
        data.append(df[i:i+sequence_length].values)
    return pad_sequences(data, maxlen=sequence_length, dtype='float32', padding='post')

def predict_importance(model, keyword, df, sequence_length=2, top_n=4):
    keyword_data = df[df['keyword'] == keyword].copy()
    features = ['type', 'year', 'frecuencia', 'title_encoded', 'keyword_encoded']
    X = create_sequences(keyword_data[features], sequence_length)
    predictions = model.predict(X)
    flat_predictions = predictions.repeat(sequence_length, axis=0)[:len(keyword_data)]
    keyword_data['importance'] = flat_predictions
    top_rows = keyword_data.sort_values(by='importance', ascending=False).head(top_n)
    return top_rows


def predecir(b):
    print(b)
    busqueda = b
    tam = len(busqueda.split())
    links,n,c = get_pdf_links(busqueda , 10)
    print('links de pdfs encontrados: ',c)
    print('links de pdfs gratuitos no encontrados: ',n)
    er = 0
    dataset = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36"
    }

    for link in links:
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

    # Initialize the LabelEncoder
    le = LabelEncoder()
    # Perform label encoding on the 'title' and 'keyword' columns
    df['title_encoded'] = le.fit_transform(df['title'])
    df['keyword_encoded'] = le.fit_transform(df['keyword'])
    scaler = MinMaxScaler()
    df[['title_encoded', 'keyword_encoded']] = scaler.fit_transform(df[['title_encoded', 'keyword_encoded']])

    df_train = []
    df_test = []
    # Loop over the data in steps of 10
    for i in range(0, len(df), 10):
        # Get the current batch of 10 rows
        batch = df[i:i+10]
        
        # Split the batch into training and test data
        df_train.append(batch[:8])
        df_test.append(batch[2:])

    # Concatenate all batches to create the final training and test data
    df_train = pd.concat(df_train)
    df_test = pd.concat(df_test)

    # Select the features and the target
    features = ['type', 'year', 'frecuencia', 'title_encoded', 'keyword_encoded']
    target = ['importancia']

    # Create sequences for the features
    X_train = create_sequences(df_train[features], sequence_length=2)
    X_test = create_sequences(df_test[features], sequence_length=2)

    # Create sequences for the target
    y_train = create_sequences(df_train[target], sequence_length=2)
    y_test = create_sequences(df_test[target], sequence_length=2)

    # Create the RNN model
    model = Sequential()
    model.add(SimpleRNN(10, activation='relu', input_shape=(2, len(features))))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(X_train, y_train, epochs=10, verbose=0)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', loss)

    # Flatten y_test y y_pred para tener una dimensión (n_samples * sequence_length)
    y_pred = model.predict(X_test)

    y_test_flat = np.reshape(y_test, (-1,))
    y_pred_flat = np.repeat(np.reshape(y_pred, (-1,)), 2)

    # Calculate evaluation metrics
    mse = np.mean(np.square(y_test_flat - y_pred_flat))
    mae = np.mean(np.abs(y_test_flat - y_pred_flat))
    rmse = np.sqrt(mse)

    print('Mean Squared Error (MSE):', mse)
    print('Mean Absolute Error (MAE):', mae)
    print('Root Mean Squared Error (RMSE):', rmse)

    pred = predict_importance(model, busqueda, df)
    
    return pred