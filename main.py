from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file, make_response,send_from_directory
from flask_mysqldb import MySQL, MySQLdb
import bcrypt
import werkzeug
from werkzeug.utils import secure_filename
import os
import re
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import ParagraphStyle
from datetime import datetime
from reportlab.lib.styles import getSampleStyleSheet
import nltk
nltk.download('stopwords')
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
import mysql.connector

app = Flask(__name__)

# Railway environment configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Database configuration for Railway
def get_db_connection():
    config = {
        'host': os.environ.get('MYSQLHOST', 'localhost'),
        'user': os.environ.get('MYSQLUSER', 'root'),
        'password': os.environ.get('MYSQLPASSWORD', ''),
        'database': os.environ.get('MYSQLDATABASE', 'reviewclassification'),
        'port': int(os.environ.get('MYSQLPORT', 3306))
    }
    return mysql.connector.connect(**config)

app.secret_key = 'bismillahjuli2024'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'review_app'
app.config['MYSQL_CURSORCLASS'] = 'DictCursor'
mysql = MySQL(app)

@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM users WHERE email=%s',(email,))
        akun = kursor.fetchone()
        kursor.close()

        if akun is not None and len(akun) > 0 :
            if bcrypt.hashpw(password, akun['password'].encode('utf-8')) == akun['password'].encode('utf-8'):
                session['username'] = akun['username']
                session['id'] = akun['id']
                session['level'] = akun['level']
                session['last_login'] = datetime.utcnow()

                kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                kursor.execute('UPDATE users SET last_login = NOW() WHERE email = %s',(email,))
                mysql.connection.commit()
                kursor.close()

                if akun['level'] == 'Admin':
                    return redirect(url_for('adminDashboard'))
                else:
                    return redirect(url_for('dashboard'))
            else:
                flash('Invalid username or password','danger')
                return redirect(url_for('login'))
        else:
            flash('User not found, please try again','danger')
            return redirect(url_for('login'))
    else:
        return render_template('login.html')

def resultPredictFile(id_user):
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    cursor.execute('SELECT predict_file.id FROM predict_file JOIN histories on predict_file.history_id=histories.id WHERE histories.predict_by=%s',(id_user,))
    predict_csv = cursor.fetchall()
    result_csv= len(predict_csv)
    cursor.close()
    return result_csv

def resultPredictText(id_user):
    kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    kursor.execute('SELECT predict_text.id FROM predict_text JOIN histories on predict_text.history_id=histories.id WHERE histories.predict_by=%s',(id_user,))
    predict_text = kursor.fetchall()
    result_text= len(predict_text)
    kursor.close()
    return result_text

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        id_user = session.get('id')

        kursorr = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursorr.execute('SELECT h.*, pt.review_text, pt.class, pf.filename_csv, pf.filename_pdf, pf.count_data, pf.count_result, pf.count_data_0, pf.count_data_1, pf.count_data_2, c.category FROM histories h LEFT JOIN predict_text pt ON h.id = pt.history_id LEFT JOIN predict_file pf ON h.id = pf.history_id JOIN categories c ON h.category_id = c.id WHERE h.predict_by=%s ORDER BY h.created_at DESC LIMIT 3', (id_user,))
        new_histories = kursorr.fetchall()
        kursorr.close()

        resultFile = resultPredictFile(id_user)
        result_text = resultPredictText(id_user)
        return render_template('user_dasbor.html', result_text=result_text, result_csv=resultFile, new_histories=new_histories)
    else:
        flash('Please login first','danger')
        return redirect(url_for('login'))

@app.route('/adminDashboard')
def adminDashboard():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM data_review')
        jumlah_review = kursor.fetchall()
        result_review = len(jumlah_review)
        kursor.close()
        if result_review >= 0:
            kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
            kursor.execute('SELECT * FROM data_review WHERE class=0')
            jumlah_fix_bug = kursor.fetchall()
            result_fix_bug = len(jumlah_fix_bug)
            kursor.close()
            if result_fix_bug >= 0:
                kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                kursor.execute('SELECT * FROM data_review WHERE class=1')
                jumlah_feature_request = kursor.fetchall()
                result_feature_request = len(jumlah_feature_request )
                kursor.close()
                if result_feature_request >= 0:
                    kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    kursor.execute('SELECT * FROM data_review WHERE class=2')
                    jumlah_noninformatif = kursor.fetchall()
                    result_noninformatif = len(jumlah_noninformatif)
                    kursor.close()
        return render_template('admin_dasboar.html', result_review=result_review, result_fix_bug=result_fix_bug, result_feature_request=result_feature_request, result_noninformatif=result_noninformatif)
    else:
        flash('Please login first','danger')
        return redirect(url_for('login'))
    
@app.route('/fixBugChart', methods=['GET'])
def fixBugChart():
    if request.method == 'GET':
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute("SELECT c.category AS label, COUNT(*) AS count FROM data_review AS r JOIN categories AS c ON r.category_id = c.id WHERE r.class = 0 GROUP BY r.category_id")
        bar_chart = kursor.fetchall()

        categories = []
        counts = []

        for row in bar_chart:
            category = row['label']
            count = row['count']
            categories.append(category)
            counts.append(count)

        kursor.close()
        response = {
            'categories': categories,
            'counts': counts
        }
        return jsonify(response)

@app.route('/featureRequestChart', methods=['GET'])
def featureRequestChart():
    if request.method == 'GET':
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute("SELECT c.category AS label, COUNT(*) AS count FROM data_review AS r JOIN categories AS c ON r.category_id = c.id WHERE r.class = 1 GROUP BY r.category_id")
        req_chart = kursor.fetchall()

        categories = []
        counts = []

        for row in req_chart:
            category = row['label']
            count = row['count']
            categories.append(category)
            counts.append(count)

        kursor.close()
        response = {
            'categories': categories,
            'counts': counts
        }
        return jsonify(response)
            
@app.route('/nonInforChart', methods=['GET'])
def nonInforChart():
    if request.method == 'GET':
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute("SELECT c.category AS label, COUNT(*) AS count FROM data_review AS r JOIN categories AS c ON r.category_id = c.id WHERE r.class = 2 GROUP BY r.category_id")
        noninf_chart = kursor.fetchall()

        categories = []
        counts = []

        for row in noninf_chart:
            category = row['label']
            count = row['count']
            categories.append(category)
            counts.append(count)

        kursor.close()
        response = {
            'categories': categories,
            'counts': counts
        }
        return jsonify(response)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/classification')
def classification():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT data_review.id, data_review.review_text, data_review.created_at, data_review.class, categories.category, users.username FROM data_review JOIN categories ON data_review.category_id = categories.id JOIN users ON data_review.added_by = users.id ORDER BY data_review.created_at DESC')
        data_review = kursor.fetchall()
        kursor.close()
        return render_template("classification.html", data_review=data_review)
    else:
        return redirect(url_for('login'))
    
@app.route('/deleteClassification', methods=['POST'])
def deleteClassification():
    if request.method == 'POST':
        delete_ids = request.form.getlist('id')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        for delete_id in delete_ids:
            kursor.execute('DELETE FROM data_review WHERE id = %s', (delete_id,))
            mysql.connection.commit()
        flash('Data review deleted','success')
        return redirect(url_for('classification'))
    else:
        return render_template('classification.html')

@app.route('/managementAccount')
def managementAccount():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM users ORDER BY created_at DESC')
        users = kursor.fetchall()
        kursor.close()
        return render_template("account_management.html", users=users)
    else:
        return redirect(url_for('login'))
    
@app.route('/addUser', methods=['GET','POST'])
def addUser():
    if request.method == 'GET':
        return render_template('account_management.html')
    else:
        username = request.form['username']
        email = request.form['email']
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt())
        level = "Developer"
        createBy = session.get('id') 
        
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM users WHERE email=%s',(email,))
        akun = kursor.fetchone()
        kursor.close()

        if akun is not None and len(akun) > 0 :
            flash('Email alredy exist','danger')
            return redirect(url_for('managementAccount'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('INSERT INTO users(username,email,password,level,created_by) VALUES(%s, %s, %s, %s, %s)', (username, email, hash_password, level,createBy))
            mysql.connection.commit()
            flash('Account added successfully','success')
            return redirect(url_for('managementAccount'))

@app.route('/deleteUser', methods=['GET','POST'])
def deleteUser():
    if request.method == 'POST':
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        for getid in request.form.getlist('delete_checkbox'):
            print(getid)
            kursor.execute('DELETE FROM users WHERE id = {0}'.format(getid))
            mysql.connection.commit()
        flash('Users account deleted','success')
        return redirect(url_for('managementAccount'))
    else:
        return render_template('managementAccount.html')

def show_accuracy_indo():
    data_uji = pd.read_csv('datauji.csv')
    X = data_uji['review_text']
    y = data_uji['class']

    sequences = tokenizer_indo.texts_to_sequences(X)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post')

    y_predict = model_indo.predict(padded_sequences)
    y_pred_labels = np.argmax(y_predict, axis=1)

    accuracy = np.sum(y_pred_labels == y) / len(y)
    accuracy_percent = accuracy * 100
    return accuracy_percent

def show_accuracy_eng():
    data_uji = pd.read_csv('datauji.csv')
    data_uji = data_uji.dropna(subset=['class', 'review_text'])  # pastikan dua kolom tidak NaN
    texts = data_uji['review_text'].tolist()
    true_labels = data_uji['class'].tolist()

    # Cek jika texts kosong
    if not texts:
        return 0

    encodings = mbert_tokenizer_eng(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

    with torch.no_grad():
        logits = model_eng(encodings['input_ids'], encodings['attention_mask'])
    y_pred = torch.argmax(logits, dim=1).numpy()
    label_map = {label: idx for idx, label in enumerate(eng_label_encoder.classes_)}
    # Pastikan semua label ada di label_map
    y_true = np.array([label_map[label] for label in true_labels if label in label_map])

    # Cek jika y_true kosong
    if len(y_true) == 0:
        return 0

    accuracy = np.sum(y_pred == y_true) / len(y_true)
    accuracy_percent = accuracy * 100
    return accuracy_percent


def show_accuracy_multilanguage():
    data_uji = pd.read_csv('datauji.csv')  
    texts = data_uji['review_text'].tolist()
    true_labels = data_uji['class'].tolist()

    # Tokenisasi menggunakan tokenizer mBERT
    encodings = mbert_tokenizer_multilanguage(texts, truncation=True, padding=True, max_length=128, return_tensors='pt')

    # Prediksi dengan model PyTorch (tanpa training)
    with torch.no_grad():
        logits = model_multilanguage(encodings['input_ids'], encodings['attention_mask'])

    y_pred = torch.argmax(logits, dim=1).numpy()

    # Jika label masih berupa teks (string), encode ke angka dulu
    label_map = {label: idx for idx, label in enumerate(multilanguage_label_encoder.classes_)}
    y_true = np.array([label_map[label] for label in true_labels if pd.notna(label) and label in label_map])

    # Cek jika y_true kosong
    if len(y_true) == 0:
        return 0

    # Hitung akurasi
    accuracy = np.sum(y_pred[:len(y_true)] == y_true) / len(y_true)
    accuracy_percent = accuracy * 100
    return accuracy_percent


@app.route('/prediction')
def prediction():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category, id FROM categories')
        category = kursor.fetchall()
        kursor.close()

        accuracy_percent = show_accuracy_eng()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template("prediction.html", category=category, accuracy=formatted_accuracy)
    else:
        return redirect(url_for('login'))

# Load English model & label encoder
eng_model_path = "eng_bilstm_mbert_model.pt"
eng_label_encoder = "eng_label_encoder.pkl"

# Load mBERT tokenizer
mbert_tokenizer_eng = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Load label encoder
with open(eng_label_encoder, "rb") as f:
    eng_label_encoder = pickle.load(f)

# Define BertBiLSTMClassifier for both English and multilanguage models
class BertBiLSTMClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        lstm_out, _ = self.lstm(outputs.last_hidden_state)
        pooled = lstm_out[:, -1, :]
        output = self.dropout(pooled)
        return self.fc(output)

# Load model
num_classes = len(eng_label_encoder.classes_)
model_eng = BertBiLSTMClassifier(hidden_dim=128, num_classes=num_classes)
model_eng.load_state_dict(torch.load('eng_bilstm_mbert_model.pt', map_location=torch.device('cpu')))
model_eng.eval()

def predict_text_english(text):
    inputs = mbert_tokenizer_eng(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model_eng(inputs['input_ids'], inputs['attention_mask'])
    pred_class = torch.argmax(logits, dim=1).item()  # integer 0/1/2
    label_map = {0: 'Fix Bug', 1: 'Feature Request', 2: 'Non-Informatif'}
    prediction_text = label_map.get(pred_class, str(pred_class))
    return pred_class, prediction_text

    
@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        predictBy = session.get('id') 
        category= request.form['category_id']
        name_apps = request.form.get('name_app')
        review_text = request.form.get('review_text') 

        input_text = cleansing_text(review_text)
        text = input_text.lower()
        
        pred_class, prediction_text = predict_text_english(text)  # <-- gunakan dua output


        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category FROM categories WHERE id = %s', (category, ))
        category_name = kursor.fetchone()
        kursor.close()
        category_name = category_name['category'].strip("[]'")

        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO histories(name_app,category_id,predict_by) VALUES(%s,%s,%s)', (name_apps,category,predictBy,))
        historis_id = cursor.lastrowid
        mysql.connection.commit()

        cursorr = mysql.connection.cursor()
        cursorr.execute('INSERT INTO predict_text(review_text,class,history_id) VALUES(%s,%s,%s)', (text,pred_class,historis_id,))
        mysql.connection.commit()

        accuracy_percent = show_accuracy_eng()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template('prediction.html', prediction_text=prediction_text, review_text=text, category_id=category, category_name=category_name, name_apps=name_apps, accuracy=formatted_accuracy)
    else:
        return redirect(url_for('prediction'))

@app.route('/savePredict', methods=['POST'])
def savePredict():
    if request.method == 'POST':
        createBy = session.get('id') 
        review_text = request.form.get('review_text')
        category_id = request.form.get('category_id')
        label = request.form.get('class')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM data_review WHERE review_text=%s',(review_text,))
        review = kursor.fetchone()
        kursor.close()

        if review is not None and len(review) > 0 :
            flash('Data review alredy exist','danger')
            return redirect(url_for('prediction'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('INSERT INTO data_review(review_text,class,category_id,added_by) VALUES(%s,%s,%s,%s)', (review_text,label,category_id,createBy,))
            mysql.connection.commit()
            flash('Data review successfully added','success')
            return redirect(url_for('prediction'))
    else:
        return redirect(url_for('prediction'))

app.config['upload'] = 'upload'

def cleansing_text(text):
    doc = re.sub(r'[^\w\s]', '', text)
    return doc

def perform_prediction(file_path):
    column = request.form.get('column')
    data = pd.read_csv(file_path, delimiter=';')
    columns_to_drop = [col for col in data.columns if col != column]
    data.drop(columns=columns_to_drop, inplace=True)

    jumlah_data = data.shape[0]

    split_data = []
    not_split_data = []

    for text in data[column]:
        sentences = re.split(r'(?<=[.!?])\s+', str(text))
        if len(sentences) > 1:
            split_data.extend(sentences)
        else:
            not_split_data.append(text)

    new_data_split = pd.DataFrame({column: split_data})
    new_data_not_split = pd.DataFrame({column: not_split_data})

    merged_data = pd.concat([new_data_split, new_data_not_split], axis=0)

    merged_data[column] = merged_data[column].apply(cleansing_text)
    merged_data[column] = merged_data[column].str.lower()
    text_column = merged_data[column].tolist()

    # --- Ganti proses prediksi di bawah ini ---
    # Tokenisasi dengan BERT tokenizer
    encodings = mbert_tokenizer_eng(text_column, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        logits = model_eng(encodings['input_ids'], encodings['attention_mask'])
    y_pred = torch.argmax(logits, dim=1).numpy()
    label_map = {0: 'Fix Bug', 1: 'Feature Request', 2: 'Non-Informatif'}
    label_names = [label_map.get(int(i), str(i)) for i in y_pred]
    merged_data['predicted_label'] = label_names

    # Kategori harus sama dengan hasil mapping
    label_categories = ['Fix Bug', 'Feature Request', 'Non-Informatif']
    merged_data['predicted_label'] = pd.Categorical(merged_data['predicted_label'], categories=label_categories, ordered=True)
    merged_data = merged_data.sort_values(by='predicted_label')

    jumlah_result = merged_data.shape[0]
    jumlah_label_0 = (merged_data['predicted_label'] == label_categories[0]).sum()
    jumlah_label_1 = (merged_data['predicted_label'] == label_categories[1]).sum()
    jumlah_label_2 = (merged_data['predicted_label'] == label_categories[2]).sum()

    return merged_data, jumlah_data, jumlah_result, y_pred, jumlah_label_0, jumlah_label_1, jumlah_label_2

def generate_pdf(merged_data, jumlah_data, jumlah_result, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2):
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, 
        pagesize=letter)
    elements = []

    title_style = ParagraphStyle(name='Title', fontSize=20, leading=50, leftIndent=130, rightIndent=110)
    title = Paragraph("Report From Prediction", title_style)
    elements.append(title)
    
    count_style = ParagraphStyle(name='Count', fontSize=13, leading=18)
    text_style = ParagraphStyle(name='Label', fontSize=13, leading=18)
    label_style = ParagraphStyle(name='Label', fontSize=13, leading=18, leftIndent=30)

    count_text = f"The count of data in the CSV file: {jumlah_result} data"
    text = "The predicted result is:"
    label_result = f"The total of data successfully prediction: {jumlah_data} data"
    label_0_text = f"Fix Bug: {jumlah_label_0} data"
    label_1_text = f"Feature Request: {jumlah_label_1} data"
    label_2_text = f"Non-Informatif: {jumlah_label_2} data"
    count_paragraph = Paragraph(count_text, count_style)
    text_paragraph = Paragraph(text, text_style)
    label_result = Paragraph(label_result, label_style)
    label_0_paragraph = Paragraph(label_0_text, label_style)
    label_1_paragraph = Paragraph(label_1_text, label_style)
    label_2_paragraph = Paragraph(label_2_text, label_style)

    # Menambahkan elemen-elemen ke dalam list
    elements.append(count_paragraph)
    elements.append(text_paragraph)
    elements.append(label_result)
    elements.append(label_0_paragraph)
    elements.append(label_1_paragraph)
    elements.append(label_2_paragraph)
    elements.append(Spacer(5, 20))  

    # Menambahkan tabel
    table_data = [["No"] + list(merged_data.columns)] + [[i+1] + row for i, row in enumerate(merged_data.values.tolist())]
    col_widths = [20, 450, 85] 
    
    styles = getSampleStyleSheet()

    for i in range(1, len(table_data)):
        cell_content = table_data[i][1]
        paragraph = Paragraph(cell_content, styles["Normal"])
        table_data[i][1] = paragraph

    table = Table(table_data, colWidths=col_widths)

    # Mengatur gaya tabel
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 20),
        ('BACKGROUND', (0, 1), (-1, -1), colors.white),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))

    elements.append(table)
    elements.append(Spacer(0, 40))  

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_style = ParagraphStyle(name='Time', fontSize=9, leading=20, leftIndent=320, rightIndent=0)
    time = f"Create at: {current_time}"
    time_paragraph = Paragraph(time, time_style)
    elements.append(time_paragraph)

    doc.build(elements)
    pdf_buffer.seek(0)

    return pdf_buffer

@app.route('/scanFile', methods=['POST'])
def scanFile():
    file = request.files['file']
    allowed_extensions = ['csv']

    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension not in allowed_extensions:
        flash('file is not of type csv', 'danger')
        return redirect(url_for('prediction'))
    else:
        count = 1
        file_name_without_ext = os.path.splitext(filename)[0]
        new_filename_csv = filename

        while os.path.exists(os.path.join('D:\\Kuliah\\Program\\upload', new_filename_csv)):
            new_filename_csv = f"{file_name_without_ext}({count}).{file_extension}"
            count += 1

        file_path = os.path.join('D:\\Kuliah\\Program\\upload', new_filename_csv)
        file.save(file_path)

        data = pd.read_csv(file_path)
        field = list(data.columns)
        columns = []
        for column in field:
            columns.extend(column.split(';'))

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category, id FROM categories')
        category = kursor.fetchall()
        kursor.close()

        accuracy_percent = show_accuracy_eng()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template('prediction.html', columns=columns, filename=new_filename_csv, category=category, accuracy=formatted_accuracy)

def is_column_non_string(check, field):
    column_data = check[field]
    is_not_string = column_data.apply(lambda x: not isinstance(x, str))
    return any(is_not_string)

@app.route('/predictFile', methods=['POST'])
def predictFile():
        name_app = request.form.get('name_app')
        field = request.form.get('column')

        file_csv = request.form.get('file_csv')
        file_path = os.path.join('D:\\Kuliah\\Program\\upload', file_csv)
        check = pd.read_csv(file_path, delimiter=';')

        if field in check.columns:
            if is_column_non_string(check, field):
                flash('File contains non-string columns', 'danger')
                return redirect(url_for('prediction'))
            
            merged_data, jumlah_data, jumlah_result, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2 = perform_prediction(file_path)
            pdf_buffer = generate_pdf(merged_data, jumlah_result, jumlah_data, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2)

            upload_folder = 'D:\\Kuliah\\Program\\download'
            allowed_extensions = {'pdf'}

            filename = secure_filename(f"{name_app}_Predict_Report.pdf")
            file_path = os.path.join(upload_folder, filename)

            counter = 0
            while os.path.exists(file_path):
                name,ext = os.path.splitext(filename)
                counter += 1
                filename = f"{name_app}_Prediction_Report_({counter}){ext}"
                file_path = os.path.join(upload_folder, filename)

            final_file_path = os.path.join(upload_folder, filename)

            with open(final_file_path, 'wb') as file:
                file.write(pdf_buffer.getbuffer())

            #save to database
            predictBy = session.get('id') 
            category_id = request.form.get('category_id')

            cursor = mysql.connection.cursor()
            cursor.execute('INSERT INTO histories(name_app,category_id,predict_by) VALUES(%s,%s,%s)', (name_app,category_id,predictBy,))
            historis_id = cursor.lastrowid
            mysql.connection.commit()

            cursorr = mysql.connection.cursor()
            cursorr.execute('INSERT INTO predict_file(filename_csv,count_data,filename_pdf,count_result,count_data_0,count_data_1,count_data_2,history_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)', (file_csv,jumlah_data,filename,jumlah_result,jumlah_label_0,jumlah_label_1,jumlah_label_2,historis_id,))
            mysql.connection.commit()

            return send_file(
                final_file_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=os.path.basename(final_file_path)        
            )
        else:
            return f"field is {field}, {file_csv}"

@app.route('/prediction_indo')
def prediction_indo():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category, id FROM categories')
        category = kursor.fetchall()
        kursor.close()

        accuracy_percent = show_accuracy_indo()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template("prediction_indo.html", category=category, accuracy=formatted_accuracy)
    else:
        return redirect(url_for('login'))
    
# Load model .h5
model_indo = load_model("indo_bilstm_fastext.h5", compile=False)
# Load tokenizer
with open("indo_tokenizer.pkl", "rb") as f:
    tokenizer_indo = pickle.load(f)
# Load label encoder
with open("indo_label_encoder.pkl", "rb") as f:
     label_encoder_indo = pickle.load(f)
    
# Inisialisasi stemmer dan stopwords
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian') + stopwords.words('english'))

# Fungsi preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|@\S+|[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# Fungsi klasifikasi
def classify_review_indo(text):
    text = preprocess_text(text)  
    sequence = tokenizer_indo.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)
    prediction = model_indo.predict(padded)
    label_index = np.argmax(prediction, axis=1)[0]
    label = label_encoder_indo.inverse_transform([label_index])[0]
    return label


#model = None
#tokenizer = None
@app.route('/predict_indo', methods=['POST', 'GET'])
def predict_indo():
    if request.method == 'POST':
        predictBy = session.get('id') 
        category= request.form['category_id']
        name_apps = request.form.get('name_app')
        review_text = request.form.get('review_text') 

        input_text = cleansing_text(review_text)
        text = input_text.lower()
        text_features = [text]
        new_data_tokens = tokenizer_indo.texts_to_sequences(text_features)
        new_data_padded = pad_sequences(new_data_tokens, maxlen=100, padding='post')

        new_data_pred = model_indo.predict(new_data_padded)
        new_data_pred_classes = np.argmax(new_data_pred, axis=1)
        label_map = {0: "Fix Bug", 1: "Feature Request", 2: "Non-Informatif"}
        prediction_text = label_map.get(int(new_data_pred_classes[0]), str(new_data_pred_classes[0]))
        

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category FROM categories WHERE id = %s', (category, ))
        category_name = kursor.fetchone()
        kursor.close()
        category_name = category_name['category'].strip("[]'")

        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO histories(name_app,category_id,predict_by) VALUES(%s,%s,%s)', (name_apps,category,predictBy,))
        historis_id = cursor.lastrowid
        mysql.connection.commit()

        cursorr = mysql.connection.cursor()
        cursorr.execute('INSERT INTO predict_text(review_text,class,history_id) VALUES(%s,%s,%s)', (text, int(new_data_pred_classes[0]),historis_id,))
        mysql.connection.commit()

        accuracy_percent = show_accuracy_indo()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template('prediction_indo.html', prediction_text=prediction_text, review_text=text, category_id=category, category_name=category_name, name_apps=name_apps, accuracy=formatted_accuracy)
    else:
        return redirect(url_for('prediction_indo'))

@app.route('/savePredict_indo', methods=['POST'])
def savePredict_indo():
    if request.method == 'POST':
        createBy = session.get('id') 
        review_text = request.form.get('review_text')
        category_id = request.form.get('category_id')
        label = request.form.get('class')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM data_review WHERE review_text=%s',(review_text,))
        review = kursor.fetchone()
        kursor.close()

        if review is not None and len(review) > 0 :
            flash('Data review alredy exist','danger')
            return redirect(url_for('prediction'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('INSERT INTO data_review(review_text,class,category_id,added_by) VALUES(%s,%s,%s,%s)', (review_text,label,category_id,createBy,))
            mysql.connection.commit()
            flash('Data review successfully added','success')
            return redirect(url_for('prediction_indo'))
    else:
        return redirect(url_for('prediction_indo'))

@app.route('/scanFile_indo', methods=['POST'])
def scanFile_indo():
    file = request.files['file']
    allowed_extensions = ['csv']

    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension not in allowed_extensions:
        flash('file is not of type csv', 'danger')
        return redirect(url_for('prediction_indo'))
    else:
        count = 1
        file_name_without_ext = os.path.splitext(filename)[0]
        new_filename_csv = filename

        while os.path.exists(os.path.join('D:\\Kuliah\\Program\\upload', new_filename_csv)):
            new_filename_csv = f"{file_name_without_ext}({count}).{file_extension}"
            count += 1

        file_path = os.path.join('D:\\Kuliah\\Program\\upload', new_filename_csv)
        file.save(file_path)

        data = pd.read_csv(file_path)
        field = list(data.columns)
        columns = []
        for column in field:
            columns.extend(column.split(';'))

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category, id FROM categories')
        category = kursor.fetchall()
        kursor.close()

        accuracy_percent = show_accuracy_indo()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template('prediction_indo.html', columns=columns, filename=new_filename_csv, category=category, accuracy=formatted_accuracy)

def perform_prediction_indo(file_path):
    column = request.form.get('column')
    data = pd.read_csv(file_path, delimiter=';')
    columns_to_drop = [col for col in data.columns if col != column]
    data.drop(columns=columns_to_drop, inplace=True)

    jumlah_data = data.shape[0]

    split_data = []
    not_split_data = []

    for text in data[column]:
        sentences = re.split(r'(?<=[.!?])\s+', str(text))
        if len(sentences) > 1:
            split_data.extend(sentences)
        else:
            not_split_data.append(text)

    new_data_split = pd.DataFrame({column: split_data})
    new_data_not_split = pd.DataFrame({column: not_split_data})
  
    merged_data = pd.concat([new_data_split, new_data_not_split], axis=0)

    merged_data[column] = merged_data[column].apply(cleansing_text)
    merged_data[column] = merged_data[column].str.lower()
    text_column = merged_data[column].tolist()

    # --- Ganti proses prediksi di bawah ini ---
    # Tokenisasi dengan BERT tokenizer
    encodings = mbert_tokenizer_multilanguage(text_column, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        logits = model_multilanguage(encodings['input_ids'], encodings['attention_mask'])
    y_pred = torch.argmax(logits, dim=1).numpy()
    label_map = {0: 'Fix Bug', 1: 'Feature Request', 2: 'Non-Informatif'}
    label_names = [label_map.get(int(i), str(i)) for i in y_pred]
    merged_data['predicted_label'] = label_names

    # Kategori harus sama dengan hasil mapping
    label_categories = ['Fix Bug', 'Feature Request', 'Non-Informatif']
    merged_data['predicted_label'] = pd.Categorical(merged_data['predicted_label'], categories=label_categories, ordered=True)
    merged_data = merged_data.sort_values(by='predicted_label')

    jumlah_result = merged_data.shape[0]
    jumlah_label_0 = (merged_data['predicted_label'] == label_categories[0]).sum()
    jumlah_label_1 = (merged_data['predicted_label'] == label_categories[1]).sum()
    jumlah_label_2 = (merged_data['predicted_label'] == label_categories[2]).sum()

    return merged_data, jumlah_data, jumlah_result, y_pred, jumlah_label_0, jumlah_label_1, jumlah_label_2

@app.route('/predictFile_indo', methods=['POST'])
def predictFile_indo():
        name_app = request.form.get('name_app')
        field = request.form.get('column')

        file_csv = request.form.get('file_csv')
        file_path = os.path.join('D:\\Kuliah\\Program\\upload', file_csv)
        check = pd.read_csv(file_path, delimiter=';')
 
        if field in check.columns:
            if is_column_non_string(check, field):
                flash('File contains non-string columns', 'danger')
                return redirect(url_for('prediction'))
            
            merged_data, jumlah_data, jumlah_result, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2 = perform_prediction_indo(file_path)
            pdf_buffer = generate_pdf(merged_data, jumlah_result, jumlah_data, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2)

            upload_folder = 'D:\\Kuliah\\Program\\download'
            allowed_extensions = {'pdf'}

            filename = secure_filename(f"{name_app}_Predict_Report.pdf")
            file_path = os.path.join(upload_folder, filename)

            counter = 0
            while os.path.exists(file_path):
                name,ext = os.path.splitext(filename)
                counter += 1
                filename = f"{name_app}_Prediction_Report_({counter}){ext}"
                file_path = os.path.join(upload_folder, filename)

            final_file_path = os.path.join(upload_folder, filename)

            with open(final_file_path, 'wb') as file:
                file.write(pdf_buffer.getbuffer())

            #save to database
            predictBy = session.get('id') 
            category_id = request.form.get('category_id')

            cursor = mysql.connection.cursor()
            cursor.execute('INSERT INTO histories(name_app,category_id,predict_by) VALUES(%s,%s,%s)', (name_app,category_id,predictBy,))
            historis_id = cursor.lastrowid
            mysql.connection.commit()

            cursorr = mysql.connection.cursor()
            cursorr.execute('INSERT INTO predict_file(filename_csv,count_data,filename_pdf,count_result,count_data_0,count_data_1,count_data_2,history_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)', (file_csv,jumlah_data,filename,jumlah_result,jumlah_label_0,jumlah_label_1,jumlah_label_2,historis_id,))
            mysql.connection.commit()

            return send_file(
                final_file_path,
                mimetype='application/pdf',
                as_attachment=True,
                download_name=os.path.basename(final_file_path)        
            )
        else:
            return f"field is {field}, {file_csv}"

@app.route('/prediction_multilanguage')
def prediction_multilanguage():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category, id FROM categories')
        category = kursor.fetchall()
        kursor.close()

        accuracy_percent = show_accuracy_multilanguage()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template("prediction_multilanguage.html", category=category, accuracy=formatted_accuracy)
    else:
        return redirect(url_for('login'))
    
   # Load Code-Mixed model & label encoder
multilanguage_model_path = "code_mixed_bilstm_mbert_model.pt"
multilanguage_label_encoder_path = "code_mixed_label_encoder.pkl"

# Load mBERT tokenizer for multilanguage
mbert_tokenizer_multilanguage = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Load label encoder
with open(multilanguage_label_encoder_path, "rb") as f:
     multilanguage_label_encoder = pickle.load(f)
 
 # Load PyTorch model   
    
num_classes_cm = len(multilanguage_label_encoder.classes_)
model_multilanguage = BertBiLSTMClassifier(hidden_dim=128, num_classes=num_classes_cm)
model_multilanguage.load_state_dict(torch.load(multilanguage_model_path, map_location=torch.device('cpu')))
model_multilanguage.eval()

def predict_text_multilanguage(text):
    inputs = mbert_tokenizer_multilanguage(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model_multilanguage(inputs['input_ids'], inputs['attention_mask'])
    pred_class = torch.argmax(logits, dim=1).item()
    label_map = {0: 'Fix Bug', 1: 'Feature Request', 2: 'Non-Informatif'}
    prediction_text = label_map.get(pred_class, str(pred_class))
    return pred_class, prediction_text


@app.route('/predict_multilanguage', methods=['POST', 'GET'])
def predict_multilanguage():
    if request.method == 'POST':
        predictBy = session.get('id') 
        category= request.form['category_id']
        name_apps = request.form.get('name_app')
        review_text = request.form.get('review_text') 

        input_text = cleansing_text(review_text)
        text = input_text.lower()

        pred_class, prediction_text = predict_text_multilanguage(text)

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category FROM categories WHERE id = %s', (category, ))
        category_name = kursor.fetchone()
        kursor.close()
        category_name = category_name['category'].strip("[]'")

        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO histories(name_app,category_id,predict_by) VALUES(%s,%s,%s)', (name_apps,category,predictBy,))
        historis_id = cursor.lastrowid
        mysql.connection.commit()

        cursorr = mysql.connection.cursor()
        cursorr.execute('INSERT INTO predict_text(review_text,class,history_id) VALUES(%s,%s,%s)', (text,pred_class,historis_id,))
        mysql.connection.commit()

        accuracy_percent = show_accuracy_multilanguage()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template('prediction_multilanguage.html', prediction_text=prediction_text, review_text=text, category_id=category, category_name=category_name, name_apps=name_apps, accuracy=formatted_accuracy)
    else:
        return redirect(url_for('prediction_multilanguage'))

@app.route('/savePredict_multilanguage', methods=['POST'])
def savePredict_multilanguage():
    if request.method == 'POST':
        createBy = session.get('id') 
        review_text = request.form.get('review_text')
        category_id = request.form.get('category_id')
        label = request.form.get('class')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM data_review WHERE review_text=%s',(review_text,))
        review = kursor.fetchone()
        kursor.close()

        if review is not None and len(review) > 0 :
            flash('Data review alredy exist','danger')
            return redirect(url_for('prediction_multilanguage'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('INSERT INTO data_review(review_text,class,category_id,added_by) VALUES(%s,%s,%s,%s)', (review_text,label,category_id,createBy,))
            mysql.connection.commit()
            flash('Data review successfully added','success')
            return redirect(url_for('prediction_multilanguage'))
    else:
        return redirect(url_for('prediction_multilanguage'))    
    
@app.route('/scanFile_multilanguage', methods=['POST'])
def scanFile_multilanguage():
    file = request.files['file']
    allowed_extensions = ['csv']

    filename = secure_filename(file.filename)
    file_extension = filename.rsplit('.', 1)[1].lower()

    if file_extension not in allowed_extensions:
        flash('file is not of type csv', 'danger')
        return redirect(url_for('prediction_multilanguage'))
    else:
        count = 1
        file_name_without_ext = os.path.splitext(filename)[0]
        new_filename_csv = filename

        while os.path.exists(os.path.join('D:\\Kuliah\\Program\\upload', new_filename_csv)):
            new_filename_csv = f"{file_name_without_ext}({count}).{file_extension}"
            count += 1

        file_path = os.path.join('D:\\Kuliah\\Program\\upload', new_filename_csv)
        file.save(file_path)

        data = pd.read_csv(file_path)
        field = list(data.columns)
        columns = []
        for column in field:
            columns.extend(column.split(';'))

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT category, id FROM categories')
        category = kursor.fetchall()
        kursor.close()

        accuracy_percent = show_accuracy_multilanguage()
        formatted_accuracy = "{:.0f}%".format(accuracy_percent)
        return render_template('prediction_multilanguage.html', columns=columns, filename=new_filename_csv, category=category, accuracy=formatted_accuracy) 

def perform_prediction_multilanguage(file_path):
    column = request.form.get('column')
    data = pd.read_csv(file_path, delimiter=';')
    columns_to_drop = [col for col in data.columns if col != column]
    data.drop(columns=columns_to_drop, inplace=True)

    jumlah_data = data.shape[0]

    split_data = []
    not_split_data = []

    for text in data[column]:
        sentences = re.split(r'(?<=[.!?])\s+', str(text))
        if len(sentences) > 1:
            split_data.extend(sentences)
        else:
            not_split_data.append(text)

    new_data_split = pd.DataFrame({column: split_data})
    new_data_not_split = pd.DataFrame({column: not_split_data})

    merged_data = pd.concat([new_data_split, new_data_not_split], axis=0)

    merged_data[column] = merged_data[column].apply(cleansing_text)
    merged_data[column] = merged_data[column].str.lower()
    text_column = merged_data[column].tolist()

    # --- Ganti proses prediksi di bawah ini ---
    # Tokenisasi dengan BERT tokenizer
    encodings = mbert_tokenizer_multilanguage(text_column, truncation=True, padding=True, max_length=128, return_tensors='pt')
    with torch.no_grad():
        logits = model_multilanguage(encodings['input_ids'], encodings['attention_mask'])
    y_pred = torch.argmax(logits, dim=1).numpy()
    label_map = {0: 'Fix Bug', 1: 'Feature Request', 2: 'Non-Informatif'}
    label_names = [label_map.get(int(i), str(i)) for i in y_pred]
    merged_data['predicted_label'] = label_names

    # Kategori harus sama dengan hasil mapping
    label_categories = ['Fix Bug', 'Feature Request', 'Non-Informatif']
    merged_data['predicted_label'] = pd.Categorical(merged_data['predicted_label'], categories=label_categories, ordered=True)
    merged_data = merged_data.sort_values(by='predicted_label')

    jumlah_result = merged_data.shape[0]
    jumlah_label_0 = (merged_data['predicted_label'] == label_categories[0]).sum()
    jumlah_label_1 = (merged_data['predicted_label'] == label_categories[1]).sum()
    jumlah_label_2 = (merged_data['predicted_label'] == label_categories[2]).sum()
    
    return merged_data, jumlah_data, jumlah_result, y_pred, jumlah_label_0, jumlah_label_1, jumlah_label_2

@app.route('/predictFile_multilanguage', methods=['POST'])
def predictFile_multilanguage():   
    name_app = request.form.get('name_app')
    field = request.form.get('column')

    file_csv = request.form.get('file_csv')
    file_path = os.path.join('D:\\Kuliah\\Program\\upload', file_csv)
    check = pd.read_csv(file_path, delimiter=';')

    if field in check.columns:
        if is_column_non_string(check, field):
            flash('File contains non-string columns', 'danger')
            return redirect(url_for('prediction_multilanguage'))
        
        merged_data, jumlah_data, jumlah_result, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2 = perform_prediction_multilanguage(file_path)
        pdf_buffer = generate_pdf(merged_data, jumlah_result, jumlah_data, predicted_labels, jumlah_label_0, jumlah_label_1, jumlah_label_2)

        upload_folder = 'D:\\Kuliah\\Program\\download'
        allowed_extensions = {'pdf'}

        filename = secure_filename(f"{name_app}_Predict_Report.pdf")
        file_path = os.path.join(upload_folder, filename)

        counter = 0
        while os.path.exists(file_path):
            name,ext = os.path.splitext(filename)
            counter += 1
            filename = f"{name_app}_Prediction_Report_({counter}){ext}"
            file_path = os.path.join(upload_folder, filename)

        final_file_path = os.path.join(upload_folder, filename)

        with open(final_file_path, 'wb') as file:
            file.write(pdf_buffer.getbuffer())

        #save to database
        predictBy = session.get('id') 
        category_id = request.form.get('category_id')
        cursor = mysql.connection.cursor()
        cursor.execute('INSERT INTO histories(name_app,category_id,predict_by) VALUES(%s,%s,%s)', (name_app,category_id,predictBy,))
    
        historis_id = cursor.lastrowid
        mysql.connection.commit()
        cursorr = mysql.connection.cursor()
        cursorr.execute('INSERT INTO predict_file(filename_csv,count_data,filename_pdf,count_result,count_data_0,count_data_1,count_data_2,history_id) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)', (file_csv,jumlah_data,filename,jumlah_result,jumlah_label_0,jumlah_label_1,jumlah_label_2,historis_id,))
        mysql.connection.commit()   
        return send_file(
            final_file_path,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=os.path.basename(final_file_path)
        )
    else:
        return f"field is {field}, {file_csv}"
    
    
    
@app.route('/category')
def category():
    if 'username' in session:
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM categories ORDER BY created_at DESC')
        data = kursor.fetchall()
        kursor.close()
        return render_template("category.html", categories=data)
    else:
        return redirect(url_for('login'))
        
@app.route('/createCategory', methods=['GET','POST'])
def createCategory():
    createBy = session.get('id') 
    if request.method == 'GET':
        return render_template('category.html', by=createBy)
    else:
        category = request.form['category']
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM categories WHERE category=%s',(category,))
        data = kursor.fetchone()
        kursor.close()

        if data is not None and len(data) > 0 :
            flash('Category alredy exist','danger')
            return redirect(url_for('category'))
        else:
            if len(category) < 4 or len(category) > 20:
                flash('Input field length must be between 4 and 20 characters','warning')
                return redirect(url_for('category')) 
            else:
                kursor = mysql.connection.cursor()
                kursor.execute('INSERT INTO categories(category,created_by) VALUES(%s,%s)', (category,createBy,))
                mysql.connection.commit()
                flash('Category added successfully','success')
                return redirect(url_for('category'))

@app.route('/updateCategory', methods=['GET','POST'])
def updateCategory():
    if request.method == 'GET':
        return render_template('category.html')
    else:
        category_id = request.form.get('category_id')
        category_name = request.form.get('category_name')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM categories WHERE category=%s',(category_name,))
        data = kursor.fetchone()
        kursor.close()

        if data is not None and len(data) > 0 :
            flash('Category alredy exist','danger')
            return redirect(url_for('category'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('UPDATE categories SET category=%s WHERE id=%s', (category_name, category_id,))
            mysql.connection.commit()
            flash('Update category successfully','success')
            return redirect(url_for('category'))
            
@app.route('/deleteCategory', methods=['GET','POST'])
def deleteCategory():
    if request.method == 'POST':
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        for getid in request.form.getlist('delete_checkbox'):
            print(getid)
            kursor.execute('DELETE FROM categories WHERE id = {0}'.format(getid))
            mysql.connection.commit()
        flash('Categories deleted','success')
        return redirect(url_for('category'))
    else:
        return render_template('category.html')

@app.route('/myAccount')
def myAccount():
    if 'username' in session:
        session_id = session.get('id')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM users WHERE id=%s', (session_id,))
        user= kursor.fetchone()
        kursor.close()
        if user:
           return render_template("myaccount.html", user=user)
        else:
            flash('User not found','danger')
            return redirect(url_for('dashboard'))
    else:
        return redirect(url_for('login'))
    
@app.route('/changeProfil', methods=['GET','POST'])
def changeProfil():
    if request.method == 'GET':
        return render_template('myaccount.html')
    else:
        id_user = session.get('id')
        email = request.form.get('email')
        username = request.form.get('username')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM users WHERE (email=%s OR username=%s) AND id!=%s', (email, username, id_user))
        data = kursor.fetchall()
        kursor.close()

        if data :
            for row in data:
                if row['email'] == email and row['username'] == username:
                    flash('Email and username alredy exist','danger')
                    return redirect(url_for('myAccount'))
                elif row['email'] == email:
                    kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    kursor.execute('SELECT * FROM users WHERE username=%s AND id=%s', (username, id_user))
                    cek_username = kursor.fetchall()
                    kursor.close()

                    if cek_username:
                        flash('Email and username alredy exist','danger')
                        return redirect(url_for('myAccount'))
                    else:
                        kursor = mysql.connection.cursor()
                        kursor.execute('UPDATE users SET username=%s WHERE id=%s', (username,id_user, ))
                        mysql.connection.commit()
                        flash('Update your username successfully','success')
                        return redirect(url_for('myAccount'))
                elif row['username'] == username:
                    kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    kursor.execute('SELECT * FROM users WHERE email=%s AND id=%s', (email, id_user))
                    cek_email = kursor.fetchall()
                    kursor.close()
                
                    if cek_email:
                        flash('Username and email alredy exist','danger')
                        return redirect(url_for('myAccount'))
                    else:
                        kursor = mysql.connection.cursor()
                        kursor.execute('UPDATE users SET email=%s WHERE id=%s', (email,id_user, ))
                        mysql.connection.commit()
                        flash('Update your email successfully','success')
                        return redirect(url_for('myAccount'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('UPDATE users SET username=%s, email=%s WHERE id=%s', (username,email,id_user, ))
            mysql.connection.commit()
            flash('Change your account successfully','success')
            return redirect(url_for('myAccount'))

@app.route('/updatePassword', methods=['GET','POST'])
def updatePassword():
    if request.method == 'GET':
        return render_template('myaccount.html')
    else:
        id_user = session.get('id')
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt())

        kursor = mysql.connection.cursor()
        kursor.execute('SELECT password FROM users WHERE id=%s', (id_user,))
        stored_password = kursor.fetchone()
        kursor.close()

        if stored_password:
            stored_passwords = stored_password['password']
            if bcrypt.checkpw(password, stored_passwords.encode('utf-8')):
                flash('Cannot be changed, the password is the same as the old password','warning')
                return redirect(url_for('myAccount'))
            else:
                kursor = mysql.connection.cursor()
                kursor.execute('UPDATE users SET password=%s WHERE id=%s', (hash_password,id_user, ))
                mysql.connection.commit()
                flash('Update Your Password Successfully','success')
                return redirect(url_for('myAccount'))
        else:
            flash('Cannot be changed your password','warning')
            return redirect(url_for('myAccount'))

@app.route('/changeUser', methods=['GET','POST'])
def changeUser():
    if request.method == 'GET':
        return render_template('account_management.html')
    else:
        id_user= request.form.get('id')
        email = request.form.get('email')
        username = request.form.get('username')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT * FROM users WHERE (email=%s OR username=%s) AND id!=%s', (email, username, id_user))
        data = kursor.fetchall()
        kursor.close()

        if data :
            for row in data:
                if row['email'] == email and row['username'] == username:
                    flash('Email and Username alredy exist','danger')
                    return redirect(url_for('managementAccount'))
                elif row['email'] == email:
                    kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    kursor.execute('SELECT * FROM users WHERE username=%s AND id=%s', (username, id_user))
                    cek_username = kursor.fetchall()
                    kursor.close()

                    if cek_username:
                        flash('Email and username alredy exist','danger')
                        return redirect(url_for('managementAccount'))
                    else:
                        kursor = mysql.connection.cursor()
                        kursor.execute('UPDATE users SET username=%s WHERE id=%s', (username,id_user, ))
                        mysql.connection.commit()
                        flash('Update Username successfully','success')
                        return redirect(url_for('managementAccount'))
                elif row['username'] == username:
                    kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
                    kursor.execute('SELECT * FROM users WHERE email=%s AND id=%s', (email, id_user))
                    cek_email = kursor.fetchall()
                    kursor.close()
                
                    if cek_email:
                        flash('Username and email alredy exist','danger')
                        return redirect(url_for('managementAccount'))
                    else:
                        kursor = mysql.connection.cursor()
                        kursor.execute('UPDATE users SET email=%s WHERE id=%s', (email,id_user, ))
                        mysql.connection.commit()
                        flash('Update user email successfully','success')
                        return redirect(url_for('managementAccount'))
        else:
            kursor = mysql.connection.cursor()
            kursor.execute('UPDATE users SET username=%s, email=%s WHERE id=%s', (username,email,id_user, ))
            mysql.connection.commit()
            flash('change user account successfully','success')
            return redirect(url_for('managementAccount'))

@app.route('/changePassword', methods=['GET','POST'])
def changePassword():
    if request.method == 'GET':
        return render_template('account_management.html')
    else:
        id_user= request.form.get('id')
        password = request.form['password'].encode('utf-8')
        hash_password = bcrypt.hashpw(password, bcrypt.gensalt())

        kursor = mysql.connection.cursor()
        kursor.execute('SELECT password FROM users WHERE id=%s', (id_user,))
        stored_password = kursor.fetchone()
        kursor.close()

        if stored_password:
            stored_passwords = stored_password['password']
            if bcrypt.checkpw(password, stored_passwords.encode('utf-8')):
                flash('Cannot be changed, the password is the same as the old password','warning')
                return redirect(url_for('managementAccount'))
            else:
                kursor = mysql.connection.cursor()
                kursor.execute('UPDATE users SET password=%s WHERE id=%s', (hash_password,id_user, ))
                mysql.connection.commit()
                flash('Change password user successfully','success')
                return redirect(url_for('managementAccount'))
        else:
            flash('Cannot be changed user password','warning')
            return redirect(url_for('managementAccount'))

@app.route('/history')
def history():
    if 'username' in session:
        id_user = session.get('id')
        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        kursor.execute('SELECT h.*, pt.*, pf.*, c.category FROM histories h LEFT JOIN predict_text pt ON h.id = pt.history_id LEFT JOIN predict_file pf ON h.id = pf.history_id JOIN categories c ON h.category_id = c.id WHERE h.predict_by = %s ORDER BY h.created_at DESC', (id_user,))
        histories = kursor.fetchall()
        kursor.close()
        return render_template("history.html", histories=histories)
    else:
        return redirect(url_for('login'))
    
@app.route('/deleteHistory', methods=['POST'])
def deleteHistory():
    if request.method == 'POST':
        delete_ids = request.form.getlist('id')

        kursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        try:
            for delete_id in delete_ids:
                kursor.execute("DELETE FROM predict_text WHERE history_id = %s", (delete_id,))

                kursor.execute("DELETE FROM predict_file WHERE history_id = %s", (delete_id,))

                kursor.execute("DELETE FROM histories WHERE id = %s", (delete_id,))
            mysql.connection.commit()
            flash('Data history deleted','success')
            return redirect(url_for('history'))
        except Exception as e:
            kursor.execute("ROLLBACK")
            flash(' ',str(e),'danger')
    else:
        return render_template('history.html')

@app.route('/downloadCsv/<filename>', methods=['GET'])
def download_csv(filename):
    file_directory = 'D:\\Kuliah\\Program\\upload'
    file_path = os.path.join(file_directory, filename)
    return send_file(file_path, as_attachment=True)

@app.route('/downloadPdf/<filename>', methods=['GET'])
def download_pdf(filename):
    file_directory = 'D:\\Kuliah\\Program\\download'
    file_path = os.path.join(file_directory, filename)
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)


