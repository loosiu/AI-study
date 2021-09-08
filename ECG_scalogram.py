from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ecg_df = pd.read_csv('./ECG/AF/MUSE_20180111_155542_84000.csv')
print(ecg_df.head(10))
print(ecg_df.tail(10))

for i in range(0,12):
    x = ecg_df.iloc[:,i]
    i = i+1
    sig = x
    wavelet = signal.ricker
    widths = np.arange(100,500)
    cwtmatr = signal.cwt(sig, wavelet, widths)
    plt.imshow(cwtmatr, extent=[0, 10, 0, 500], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plt.title('Scalogram(AF)')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.show()