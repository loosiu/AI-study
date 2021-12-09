from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import gc

def scalogram(sig):   
    plt.rcParams['figure.figsize'] = [3.87,3.97]
    plt.rcParams['figure.dpi'] = 100
    wavelet = signal.ricker
    widths = np.arange(1,50)
    cwtmatr = signal.cwt(sig, wavelet, widths)
    plt.imshow(cwtmatr, extent=[0, 10, 0, 50], cmap='PRGn', aspect='auto',
               vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())

      
path = './ECG/AF/'
file_list_AF = os.listdir(path)
# file_list_AF_py = [file for file in os.listdir(path)]

for i in file_list_AF[0:280]:  
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/AFIB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
    del data
    gc.collect()
        
            # del i
            # del j
# del file_list_AF
#     del data

        
for i in file_list_AF[280:350]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/AFIB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data

        
for i in file_list_AF[350:438]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):        
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/AFIB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data 
    # gc.collect()    
del file_list_AF
gc.collect() 
   
path = './ECG/AFIB/'
file_list_AFIB = os.listdir(path)

for i in file_list_AFIB[0:1139]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/AFIB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j  
    # del data
        
for i in file_list_AFIB[1139:1424]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/AFIB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j  
    # del data
        
for i in file_list_AFIB[1424:1780]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/AFIB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j

    # del data 
del file_list_AFIB
gc.collect()

path = './ECG/AT/'
file_list_AT = os.listdir(path)

for i in file_list_AT[0:78]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()  
        if j == 11:
            del data
            del i
            del j 
    # del data
        
for i in file_list_AT[78:97]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
      data = pd.read_csv(path + i)
      scalogram(data.iloc[:,j])
      plt.savefig(f'./ECG/scalogram/lead{j+1}/val/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
      plt.close() 
      if j == 11:
            del data
            del i
            del j
    # del data

for i in file_list_AT[97:121]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()  
        if j == 11:
            del data
            del i
            del j 
    # del data 
del file_list_AT
gc.collect()
        
path = './ECG/AVNRT/'
file_list_AVNRT = os.listdir(path)

for i in file_list_AVNRT[0:10]: 
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()  
        if j == 11:
            del data
            del i
            del j
    # del data
        
for i in file_list_AVNRT[10:13]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close() 
        if j == 11:
            del data
            del i
            del j
    # del data
        
for i in file_list_AVNRT[13:16]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close() 
        if j == 11:
            del data
            del i
            del j
    # del data         
del file_list_AVNRT
gc.collect()   

path = './ECG/AVRT/'
file_list_AVRT = os.listdir(path)
        
for i in file_list_AVRT[0:5]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close() 
        if j == 11:
            del data
            del i
            del j  
    # del data

for i in file_list_AVRT[5:6]:  
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j  
    # del data
        
for i in file_list_AVRT[6:8]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data    
del file_list_AVRT
gc.collect() 
        

path = './ECG/SA/'
file_list_SA = os.listdir(path)

for i in file_list_SA[0:254]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/SR/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data
    
for i in file_list_SA[254:318]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/SR/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data
        
for i in file_list_SA[318:397]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/SR/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data    
del file_list_SA
gc.collect()
        
path = './ECG/SAAWR/'
file_list_SAAWR = os.listdir(path)

for i in file_list_SAAWR[0:5]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j   
    # del data
        
for i in file_list_SAAWR[5:6]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data
       
for i in file_list_SAAWR[6:7]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()   
        if j == 11:
            del data
            del i
            del j
    # del data    
del file_list_SAAWR
gc.collect()
      
path = './ECG/SB/' 
file_list_SB = os.listdir(path)

for i in file_list_SB[0:2488]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):  
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/SB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close() 
        if j == 11:
            del data
            del i
            del j
    # del data
      
for i in file_list_SB[2488:3110]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/SB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close() 
        if j == 11:
            del data
            del i
            del j 
    # del data
      
for i in file_list_SB[3110:3888]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/SB/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close() 
        if j == 11:
            del data
            del i
            del j
    # del data    
del file_list_SB
gc.collect()

path = './ECG/SR/'
file_list_SR = os.listdir(path)

for i in file_list_SR [0:1168]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/SR/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data
    
for i in file_list_SR [1168:1460]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/SR/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j 
    # del data
                
for i in file_list_SR [1460:1825]:  
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/SR/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data    
del file_list_SR
gc.collect()
 
path = './ECG/ST/'        
file_list_ST = os.listdir(path)

for i in file_list_ST[0:1001]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data
    
for i in file_list_ST[1001:1251]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j 
    # del data
        
for i in file_list_ST[1251:1564]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j 
    # del data    
del file_list_ST
gc.collect()
        
path = './ECG/SVT/'      
file_list_SVT = os.listdir('./ECG/SVT/')

for i in file_list_SVT[0:348]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/train/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j    
    # del data
    
for i in file_list_SVT[348:435]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/val/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j 
    # del data
    
for i in file_list_SVT[435:544]:
    data = pd.read_csv(f'{path}{i}', header=None, error_bad_lines=False)
    for j in range(0,12):
        scalogram(data.iloc[:,j])
        plt.savefig(f'./ECG/scalogram/lead{j+1}/test/GSVT/{i}_lead{j+1}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        if j == 11:
            del data
            del i
            del j
    # del data    
del file_list_SVT
gc.collect()
    
        
        
            
        
        
        
