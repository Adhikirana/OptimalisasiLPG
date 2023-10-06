#-------------------------------------------------------------------------------------------------
# LIBRARY YANG DIGUNAKAN
#-------------------------------------------------------------------------------------------------
import pandas as pd
from pmdarima import auto_arima
import math
import numpy as np
#--------------------------------------------------------------------------------------------------
# MEMBACA DATA DARI FILE EXCEL
#--------------------------------------------------------------------------------------------------
file_path = "DataDasarElpiji.xlsx"
data = pd.read_excel(file_path, sheet_name="Sheet1")
# Definisikan kolom-kolom sebagai variabel terpisah
KebutuhanTabungGas = data['Kebutuhan Tabung Gas']
KenaikanSales = data['Kenaikan Sales']
BufferStock = data['Buffer Stock']
KebutuhanRolling = data['Kebutuhan Rolling']
PertumbuhanChanel = data['Pertumbuhan Chanel']
#---------------------------------------------------------------------------------------------------
# PREDIKSI KEBUTUHAN TABUNG GAS DALAM PERIODE WAKTU TERTENTU
#---------------------------------------------------------------------------------------------------
# Penambahan indeks waktu berdasarkan jumlah data
data['Time_Index'] = range(1, len(data) + 1)
data.set_index('Time_Index', inplace=True)
# Metode auto_arima akan mencari model ARIMA terbaik berdasarkan data yang diberikan
ARIMAkebutuhantabunggas = auto_arima(KebutuhanTabungGas, seasonal=False, m=1, trace=True)
# Membuat prediksi untuk periode berikutnya
HasilPrediksi1 = ARIMAkebutuhantabunggas.predict(n_periods=1)
# Menampilkan hasil prediksi
print('Prediksi Kebutuhan Tabung Gas periode ke  ' + str(HasilPrediksi1))
#---------------------------------------------------------------------------------------------------
# PREDIKSI KENAIKAN SALES DALAM PERIODE WAKTU TERTENTU
#---------------------------------------------------------------------------------------------------
data['Time_Index'] = range(1, len(data) + 1)
data.set_index('Time_Index', inplace=True)
# Metode auto_arima akan mencari model ARIMA terbaik berdasarkan data yang diberikan
ARIMAkenaikansales = auto_arima(KenaikanSales, seasonal=False, m=1, trace=True)
# Membuat prediksi untuk periode berikutnya
HasilPrediksi2 = ARIMAkenaikansales.predict(n_periods=1)
# Menampilkan hasil prediksi
print('Prediksi Kenaikan Sales periode ke  ' + str(HasilPrediksi2))
#---------------------------------------------------------------------------------------------------
# MENGHITUNG BUFFER STOCK
#---------------------------------------------------------------------------------------------------
data['Time_Index'] = range(1, len(data) + 1)
data.set_index('Time_Index', inplace=True)
# Metode auto_arima akan mencari model ARIMA terbaik berdasarkan data yang diberikan
ARIMAbufferstock = auto_arima(BufferStock, seasonal=False, m=1, trace=True)
# Membuat prediksi untuk periode berikutnya
HasilPrediksi3 = ARIMAbufferstock.predict(n_periods=1)
# Menampilkan hasil prediksi
print('Prediksi Buffer Stock periode ke  ' + str(HasilPrediksi3))
#--------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
# MENGHITUNG KEBUTUHAN ROLLING
#----------------------------------------------------------------------------------------------------
data['Time_Index'] = range(1, len(data) + 1)
data.set_index('Time_Index', inplace=True)
# Metode auto_arima akan mencari model ARIMA terbaik berdasarkan data yang diberikan
ARIMAkebutuhanrolling = auto_arima(KebutuhanRolling, seasonal=False, m=1, trace=True)
# Membuat prediksi untuk periode berikutnya
HasilPrediksi4 = ARIMAkebutuhanrolling.predict(n_periods=1)
# Menampilkan hasil prediksi
print('Prediksi Kebutuhan Rolling periode ke  ' + str(HasilPrediksi4))
#----------------------------------------------------------------------------------------------------
#PREDIKSI PERTUMBUHAN CHANEL
#----------------------------------------------------------------------------------------------------
data['Time_Index'] = range(1, len(data) + 1)
data.set_index('Time_Index', inplace=True)
# Metode auto_arima akan mencari model ARIMA terbaik berdasarkan data yang diberikan
ARIMAPertumbuhanChanel = auto_arima(PertumbuhanChanel, seasonal=False, m=1, trace=True)
# Membuat prediksi untuk periode berikutnya
HasilPrediksi5 = ARIMAPertumbuhanChanel.predict(n_periods=1)
# Menampilkan hasil prediksi
print('Prediksi Pertumbuhan Chanel Periode ke  ' + str(HasilPrediksi5))
#----------------------------------------------------------------------------------------------------
# MENGHITUNG JUMLAH TABUNG OPTIMAL
#----------------------------------------------------------------------------------------------------
D = HasilPrediksi4
S = HasilPrediksi2
H = 15000
G = HasilPrediksi5
BS = HasilPrediksi3
RumusEOQ = ((2*(G+S)*S*D)/H)
EOQ = math.sqrt(RumusEOQ)
Optimal = EOQ + BS
print ('Nilai Optimal Pembelian Tabung sebesar ' + str (EOQ))
#----------------------------------------------------------------------------------------
#MENGHITUNG NILAI ERROR
# Data aktual Kebutuhan Tabung Gas
data_aktual_kebutuhan_tabung_gas = data['Kebutuhan Tabung Gas']
# Menghitung selisih antara prediksi dan data aktual
selisih = HasilPrediksi1 - data_aktual_kebutuhan_tabung_gas
# Menghitung MSE
mse = np.mean(selisih**2)
# Menampilkan MSE
print('Mean Squared Error (MSE) untuk Prediksi Kebutuhan Tabung Gas adalah ' + str(mse))
