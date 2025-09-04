import streamlit as st
import pandas as pd
import joblib
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Download stopwords
nltk.download('stopwords')
stopword_indonesia = set(stopwords.words('indonesian'))

# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Load model dan vectorizer
vectorizer = joblib.load("tfidf_vectorizer.pkl")
model = joblib.load("model.pkl")

# Preprocessing Lengkap
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'@\w+|#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopword_indonesia]
    stemmed = [stemmer.stem(word) for word in tokens]
    return ' '.join(stemmed)

# ============================
#       STREAMLIT UI
# ============================
st.title("📊 Aplikasi Prediksi Sentimen")

# ====================
# Prediksi Manual
# ====================
st.subheader("📝 Prediksi Manual")
user_input = st.text_input("Masukkan teks opini:")

if st.button("Prediksi Teks"):
    if user_input:
        cleaned_text = preprocess_text(user_input)
        X = vectorizer.transform([cleaned_text])
        prediction = model.predict(X)[0]
        label = "Positif" if prediction in [1, "positif"] else "Negatif"

        st.write("🔍 Hasil Preprocessing:", cleaned_text)
        st.success(f"✅ Hasil Prediksi: {label}")
    else:
        st.warning("⚠️ Masukkan teks terlebih dahulu.")

# ===============================
# Prediksi dari File CSV Upload
# ===============================
st.subheader("📁 Prediksi dari File CSV")
uploaded_file = st.file_uploader("Upload file CSV (harus memiliki kolom teks)", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except:
        st.error("❌ Gagal membaca file. Pastikan formatnya benar.")
        st.stop()

    if df.empty or len(df.columns) == 0:
        st.error("❌ File kosong atau tidak valid.")
        st.stop()

    selected_column = st.selectbox("📌 Pilih kolom teks untuk dianalisis:", df.columns)

    # Proses: Preprocessing + Prediksi + Visualisasi
    df = df.dropna(subset=[selected_column])

    st.info("⏱️ Mengukur waktu komputasi...")

    t0 = time.time()

    df['clean_text'] = df[selected_column].astype(str).apply(preprocess_text)
    t1 = time.time()
    tfidf_input = vectorizer.transform(df['clean_text'])
    t2 = time.time()
    predictions = model.predict(tfidf_input)
    t3 = time.time()

    df['prediksi'] = ['Positif' if p in [1, "positif"] else 'Negatif' for p in predictions]

    # Hasil
    st.write("📄 Hasil Prediksi Sentimen:")
    st.dataframe(df[[selected_column, 'clean_text', 'prediksi']])

    # Waktu Komputasi
    st.markdown("### ⏱️ Waktu Komputasi")
    st.write(f"🧹 Preprocessing: {t1 - t0:.2f} detik")
    st.write(f"🧠 TF-IDF Transformasi: {t2 - t1:.2f} detik")
    st.write(f"🤖 Prediksi Model: {t3 - t2:.2f} detik")
    st.write(f"🕒 Total: {t3 - t0:.2f} detik")

    # Unduh hasil prediksi
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Unduh Hasil Prediksi", data=csv, file_name='hasil_prediksi.csv', mime='text/csv')

    # ==================
    # Visualisasi
    # ==================
    st.subheader("📊 Visualisasi Hasil Sentimen")

    label_counts = df['prediksi'].value_counts()

    # Bar chart
    fig, ax = plt.subplots()
    sns.barplot(x=label_counts.index, y=label_counts.values, palette='pastel', ax=ax)
    ax.set_title("Distribusi Sentimen")
    ax.set_ylabel("Jumlah")
    ax.set_xlabel("Label Sentimen")
    st.pyplot(fig)

    # Pie chart
    fig2, ax2 = plt.subplots()
    ax2.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
    ax2.axis('equal')
    st.pyplot(fig2)
