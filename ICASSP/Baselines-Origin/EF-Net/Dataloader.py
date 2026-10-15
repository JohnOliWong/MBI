import os
import  math
import numpy as np
import torch
import scipy.io
import pandas as pd

freq_f=10 # Hz
freq_e=200
slide=1 # sec 
window=5  # sec
downsampled=True # downsample by half

# subject-dependent
'''
outputs:
eeg [360, 500, 30, 1] = [trial, time_sample, channel, 1]
nirs [360, 25, 72, 1]
labels [360,]
'''
def extract_subj_dep(subject,baseline):
    file_path = "../../Data/EF-WG/Raw/VP0"+subject+"-NIRS/cnt_wg.mat"
    file_path2 = "../../Data/EF-WG/Raw/VP0"+subject+"-EEG/cnt_wg.mat"
    fnirsdata = scipy.io.loadmat(file_path, simplify_cells=True)
    eegdata= scipy.io.loadmat(file_path2, simplify_cells=True)

    file_path3 = "../../Data/EF-WG/Raw/VP0"+subject+"-NIRS/mrk_wg.mat"
    file_path4 = "../../Data/EF-WG/Raw/VP0"+subject+"-EEG/mrk_wg.mat"
    fnirs_session_marks = scipy.io.loadmat(file_path3, simplify_cells=True)
    eeg_session_marks = scipy.io.loadmat(file_path4, simplify_cells=True)

    fstart_times_ms=fnirs_session_marks['mrk_wg']['time']
    estart_times_ms=eeg_session_marks['mrk_wg']['time']

    y_raw_fnirs=fnirs_session_marks['mrk_wg']['y']

    X_fnirs=[]
    X_eeg=[]
    Y=[]

    data_deoxy=fnirsdata['cnt_wg']['deoxy']['x']
    data_oxy=fnirsdata['cnt_wg']['oxy']['x']
    data_eeg=eegdata['cnt_wg']['x']

    for i,ftask_idx in enumerate(fstart_times_ms):     
    
        etask_idx=estart_times_ms[i]
         
        ftask_idx= ftask_idx/1000 # msec to sec
        ftask_idx=round(ftask_idx*freq_f) # sec to samples(freq)
		# the first 2 seconds of data is discarded
        task_deoxy=data_deoxy[(ftask_idx+2*freq_f):(ftask_idx+12*freq_f)]
        task_oxy=data_oxy[(ftask_idx+2*freq_f):(ftask_idx+12*freq_f)]
        
        etask_idx= etask_idx/1000
        etask_idx=round(etask_idx*freq_e)
        task_eeg=data_eeg[(etask_idx+2*freq_e):(etask_idx+12*freq_e)]

        tasks_fnirs=np.concatenate((task_deoxy, task_oxy), axis=1)
        print("tasks shape",tasks_fnirs.shape)
        
        e_range=range(0,task_eeg.shape[0],freq_e*slide)
        for k,j in enumerate(range(0,tasks_fnirs.shape[0],freq_f*slide)):
            ftask_sample=tasks_fnirs[j:(j+freq_f*window),:]
            etask_sample=task_eeg[e_range[k]:(e_range[k]+freq_e*window),:]

            # downsample:
            if downsampled:
                ftask_sample=ftask_sample[1::2]
                etask_sample=etask_sample[1::2]

            if baseline:
                # reshaping for resnet
                eegsample=np.zeros([3,200,32])
                eegsample[0,0:200,0:30]=etask_sample[0:200,0:30]
                eegsample[1,0:200,0:30]=etask_sample[150:350,0:30]
                eegsample[2,0:200,0:30]=etask_sample[300:500,0:30]
                eegsample = eegsample.transpose((1, 2, 0))

                fnirssample=np.zeros([3,32,72])
                fnirssample[0,0:25,0:72]=ftask_sample[0:25,0:72]
                fnirssample = fnirssample.transpose((1, 2, 0))

            if baseline:
                    X_fnirs.append(fnirssample)
                    X_eeg.append(eegsample)
            else:
                    # as a features matrix, samplesxchannels - EEG=500,30,1 FNIRS=25,72,1
                    ftask_sample=np.reshape(ftask_sample,(ftask_sample.shape[0],ftask_sample.shape[1],1))
                    etask_sample=np.reshape(etask_sample,(etask_sample.shape[0],etask_sample.shape[1],1))
                    X_fnirs.append(ftask_sample)
                    X_eeg.append(etask_sample)
            # taking binary labels for binary_crossentropy predictions
            Y.append(y_raw_fnirs[:,i])

            # check if window slide is ending
            if((j+freq_f*window)>=(tasks_fnirs.shape[0])):
                break
        print("Length of data samples, ",len(Y))

    X_fnirs=np.array(X_fnirs)
    X_eeg=np.array(X_eeg)
    Y=np.array(Y)

    print(X_fnirs.shape)
    print(X_eeg.shape)
    print(Y.shape)

    return X_eeg, X_fnirs, Y


def read_excel_eeg(subject_id, mode):
	# 1. set directory
	if not mode:
		root_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MI/{subject_id}'
		eeg_key = f'{subject_id}_EEG.xls'
	else:
		root_path = f'../../Dataset/EF-MI-MA/EEG-EXCEL-MA/{subject_id}'
		eeg_key = f'{subject_id}_EEG-MA.xls'
	eeg_path = os.path.join(root_path, eeg_key)
	labels_key = f'{subject_id}_desc.xls'
	labels_path = os.path.join(root_path, labels_key)

	# 2. read excel files
	# raw data format: 60 sheets, each containing a matrix with shape (7000, 36)
	raw_eeg = pd.read_excel(eeg_path, header=None, sheet_name=None)
	desc = pd.read_excel(labels_path, header=None)

	# 3. concatenate all sheets into a single tensor
	eeg = []
	for i in range(1, 61):
		sheet_name = f'Sheet{i}'
		eeg.append(raw_eeg[sheet_name].values)
	eeg = np.array(eeg).transpose((0, 2, 1)) # eeg.shape = (trial, channel, sample_points)
	desc = np.array(desc)

	# 4. extract trials belonging to different categories
	eeg_L = []
	eeg_R = []
	start = 2000
	end = 6000
	for i in range(60):
		if desc[i, 0] == 16:
			eeg_L.append(eeg[i, :, start:end])
		elif desc[i, 0] == 32:
			eeg_R.append(eeg[i, :, start:end])
	
	# 5. concatenate trials one by one and generate labels (1 for left hand, 2 for right hand)
	eeg_L = np.array(eeg_L) # eeg_L.shape = (30, 30, 4000)
	eeg_R = np.array(eeg_R)
	eeg = []
	labels = []
	for i in range(30):
		eeg.append(eeg_L[i, :, :])
		eeg.append(eeg_R[i, :, :])
		labels.append(0)
		labels.append(1)
	eeg = np.array(eeg)
	eeg = np.stack(eeg, axis=0)
	eeg = np.reshape(eeg, (eeg.shape[0], eeg.shape[2], eeg.shape[1], 1))
	labels = np.array(labels)

	eeg = torch.tensor(eeg, dtype=torch.float32)
	labels = torch.tensor(labels, dtype=torch.long)
	print(f'{subject_id}', 'EEG', eeg.shape, 'Labels', labels.shape)

	return eeg, labels

def read_excel_nirs(subject_id, mode):
	if not mode:
		root_path = f'../../Dataset/EF-MI-MA/NIRS-EXCEL-MI/{subject_id}'
	else:
		root_path = f'../../Dataset/EF-MI-MA/NIRS-EXCEL-MA/{subject_id}'
	oxy_key = f'{subject_id}_oxy.xls'
	oxy_path = os.path.join(root_path, oxy_key)
	deoxy_key = f'{subject_id}_deoxy.xls'
	deoxy_path = os.path.join(root_path, deoxy_key)
	labels_key = f'{subject_id}_desc.xls'
	labels_path = os.path.join(root_path, labels_key)

	raw_oxy = pd.read_excel(oxy_path, header=None, sheet_name=None)
	raw_deoxy = pd.read_excel(deoxy_path, header=None, sheet_name=None)
	desc = pd.read_excel(labels_path, header=None)

	HbO = []
	HbR = []
	for i in range(1, 61):
		sheet_name = f'Sheet{i}'
		HbO.append(raw_oxy[sheet_name].values)
		HbR.append(raw_deoxy[sheet_name].values)
	
	HbO = np.array(HbO).transpose((0, 2, 1)) # HbO.shape = (60, 36, 350)
	HbR = np.array(HbR).transpose((0, 2, 1))
	desc = np.array(desc)

	HbO_L = []
	HbO_R = []
	HbR_L = []
	HbR_R = []
	start = 100
	end = 300
	for i in range(60):
		if desc[i, 0] == 1:
			HbO_L.append(HbO[i, :, start:end])
			HbR_L.append(HbR[i, :, start:end])
		elif desc[i, 0] == 2:
			HbO_R.append(HbO[i, :, start:end])
			HbR_R.append(HbR[i, :, start:end])
	
	HbO_L = np.array(HbO_L)
	HbO_R = np.array(HbO_R)

	nirs = []
	labels = []
	for i in range(30):
		nirs.append(HbO_L[i, :, :])
		nirs.append(HbO_R[i, :, :])
		labels.append(0)
		labels.append(1)
	
	nirs = np.array(nirs)
	nirs = np.stack(nirs, axis=0)
	nirs = np.reshape(nirs, (nirs.shape[0], nirs.shape[2], nirs.shape[1], 1))
	labels = np.array(labels)
	
	nirs = torch.tensor(nirs, dtype=torch.float32)
	print(f'{subject_id}', 'fNIRS', nirs.shape, 'Labels', labels.shape)

	return nirs