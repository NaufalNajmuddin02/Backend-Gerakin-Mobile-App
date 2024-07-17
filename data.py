import streamlit as st
from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud
from matplotlib.animation import FuncAnimation
import pandas as pd
import datetime

# Membuat koneksi dengan MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client['gesture_database']
collection = db['gesture_logs']

# Mengambil data kata-kata dari MongoDB hanya dengan label kata
words_data = collection.find({}, {"_id": 0, "class": 1})
words_list = [word.get("class", " ") for word in words_data]  # Gunakan get untuk menghindari KeyError

# Menghitung total kata-kata
total_words = len(words_list)

# Menampilkan data di Streamlit
st.title("Data Gerakin Deteksi Bahasa Isyarat")
st.write(f"Total Kata: {total_words}")

# Menggunakan WordCloud untuk menampilkan data
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(' '.join(words_list))

# Menampilkan wordcloud menggunakan matplotlib
st.title("Data kata yang ada")
st.image(wordcloud.to_array(), use_column_width=True)

# Menghitung frekuensi kata-kata unik
unique_words, counts = np.unique(words_list, return_counts=True)

# Menampilkan visualisasi frekuensi kata-kata
st.title("Total Kata-kata")

# Membuat bar chart untuk menampilkan proporsi kata-kata
fig, ax = plt.subplots()
bars = ax.bar(unique_words, counts, color='skyblue')
ax.set_xlabel('Kata-kata')
ax.set_ylabel('Frekuensi')
ax.set_title('Total Kata-kata')
plt.xticks(rotation=45)

# Menambahkan interaktivitas untuk menampilkan angka saat di klik
def autolabel(rects):
    """Attach a text label beside each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width(), height / 2),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center', fontsize=8)

autolabel(bars)  # Menambahkan label pada bar chart

# Fungsi animasi untuk mengubah data dalam bar chart
def animate(frame):
    for bar, count in zip(bars, counts):
        bar.set_height(frame * count / 100)  # Animasi dengan mengubah tinggi batang
    return bars

# Menjalankan animasi dengan matplotlib FuncAnimation
ani = FuncAnimation(fig, animate, frames=range(1, 101), interval=50, repeat=True)

# Menampilkan grafik dengan Streamlit
st.pyplot(fig)

# Menampilkan data harian berdasarkan timestamp dan class
st.title("Data Harian Berdasarkan Tanggal Dan Hasil prediksi")

# Mengambil data kata-kata dari MongoDB dengan timestamp dan class
daily_data = collection.find({}, {"_id": 0, "timestamp": 1, "class": 1})
df = pd.DataFrame(list(daily_data))

# Konversi kolom timestamp ke datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Menambahkan kolom 'day' untuk hari
df['day'] = df['timestamp'].dt.date

# Menghitung frekuensi kelas berdasarkan hari
daily_counts = df.groupby(['day', 'class']).size().unstack(fill_value=0)

# Membuat grafik untuk menampilkan data harian
fig, ax = plt.subplots()
daily_counts.plot(kind='bar', stacked=True, ax=ax)
ax.set_xlabel('Day')
ax.set_ylabel('Frequency')
ax.set_title('Daily Class Detection Frequency')
plt.xticks(rotation=45)

# Menambahkan angka di samping batang
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(bar.get_x() + bar.get_width(), bar.get_height() / 2),
                        xytext=(3, 0),  # 3 points horizontal offset
                        textcoords="offset points",
                        ha='left', va='center', fontsize=5)

# Menampilkan grafik dengan Streamlit
st.pyplot(fig)

# Menampilkan data pengguna berdasarkan gender dan age_group
st.title("Data Pengguna Berdasarkan Jenis kelamin dan Umur")

# Mengambil data pengguna dari MongoDB dengan gender dan age_group
user_data = collection.find({}, {"_id": 0, "gender": 1, "age_group": 1})
df_user = pd.DataFrame(list(user_data))

# Menghitung frekuensi pengguna berdasarkan gender
gender_counts = df_user['gender'].value_counts()

# Menghitung frekuensi pengguna berdasarkan age_group
age_group_counts = df_user['age_group'].value_counts()

# Membuat grafik untuk menampilkan data gender
fig, ax = plt.subplots()
gender_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_xlabel('Gender')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Pengguna by Jenis Kelamin')
plt.xticks(rotation=0)

# Menambahkan angka di samping batang
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width(), height / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha='left', va='center', fontsize=8)

# Menampilkan grafik dengan Streamlit
st.pyplot(fig)

# Membuat grafik untuk menampilkan data age_group
fig, ax = plt.subplots()
age_group_counts.plot(kind='bar', color='skyblue', ax=ax)
ax.set_xlabel('Age Group')
ax.set_ylabel('Frequency')
ax.set_title('Frequency of Users by Age Group')
plt.xticks(rotation=0)

# Menambahkan angka di samping batang
for bar in ax.patches:
    height = bar.get_height()
    ax.annotate(f'{int(height)}',
                xy=(bar.get_x() + bar.get_width(), height / 2),
                xytext=(3, 0),  # 3 points horizontal offset
                textcoords="offset points",
                ha='left', va='center', fontsize=8)

# Menampilkan grafik dengan Streamlit
st.pyplot(fig)

# Menampilkan data yang baru diinput berdasarkan tanggal terbaru
st.title("Data Terbaru yang Di Input")

# Mengambil data terbaru dari MongoDB berdasarkan timestamp
latest_data = collection.find().sort("timestamp", -1).limit(10)
df_latest = pd.DataFrame(list(latest_data))

# Menampilkan data terbaru di Streamlit
st.write("Data terbaru berdasarkan timestamp:")
st.dataframe(df_latest)

# Menutup koneksi MongoDB
client.close()
