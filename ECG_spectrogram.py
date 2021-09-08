from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd

ecg_df = pd.read_csv('./Desktop/ECG/AF/MUSE_20180111_155542_84000.csv')

fs = 500
print(ecg_df.head(10))
print(ecg_df.tail(10))
for i in range(0,12):
    x = ecg_df.iloc[:,i]
    i = i+1
    f, t, Sxx = signal.spectrogram(x, fs)
    
    plt.pcolormesh(t, f, Sxx, shading='auto')
    plt.title('Spectrogram(AF)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.xlim(0,10)
    plt.ylim(0,50)
    plt.colorbar(label='')
    plt.show()
    

    
