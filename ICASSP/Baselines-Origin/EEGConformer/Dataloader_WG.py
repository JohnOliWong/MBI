import numpy as np
import math
from scipy.io import loadmat as load
import time
import torch

freq_f = 10 # Hz
freq_e = 200
slide = 1 # sec
window = 5
downsampled=True # dropout_rate = 0.5
baseline = False # baseline = ResNet

def read_wg_data_single(subject,eeg_root, nirs_root, baseline):
    '''
    outputs:
    eeg [360, 500, 30, 1] -> [trials, time_points, channels, 1]
    nirs [360, 25, 72, 1]
    labels [360, 2]
    '''
    subject = f'{subject:02d}'
    nirs_path = nirs_root + 'VP0' + str(subject) + '-NIRS/cnt_wg.mat'
    eeg_path = eeg_root + 'VP0' + str(subject) + '-EEG/cnt_wg.mat'
    fnirsdata = load(nirs_path, simplify_cells=True)
    eegdata= load(eeg_path, simplify_cells=True)

    nirs_mark_path = nirs_root + 'VP0' + str(subject) + '-NIRS/mrk_wg.mat'
    eeg_mark_path = eeg_root + 'VP0' + str(subject) + '-EEG/mrk_wg.mat'
    fnirs_session_marks = load(nirs_mark_path, simplify_cells=True)
    eeg_session_marks = load(eeg_mark_path, simplify_cells=True)

    # len of total NIRs events 
    fstart_times_ms=fnirs_session_marks['mrk_wg']['time']
    estart_times_ms=eeg_session_marks['mrk_wg']['time']
    y_raw_fnirs=fnirs_session_marks['mrk_wg']['y']

    # split data into 60 trials, save extra data set to separate fnirs and eeg
    nirs=[]
    eeg=[]
    labels=[]

    data_deoxy=fnirsdata['cnt_wg']['deoxy']['x']
    data_oxy=fnirsdata['cnt_wg']['oxy']['x']
    data_eeg=eegdata['cnt_wg']['x']

    for i,ftask_idx in enumerate(fstart_times_ms):     
    
        etask_idx=estart_times_ms[i]
         
        # epoching
        # a. extract data of 2-12s into an epoch, concatenate oxy and deoxy f-nirs data along axis=1
        # b. extract sample_frequency * window points from each epoch
        # c. downsample by half
        # d. reshape and append
        ftask_idx= ftask_idx/1000
        ftask_idx=round(ftask_idx*freq_f)
        task_deoxy=data_deoxy[(ftask_idx+2*freq_f):(ftask_idx+12*freq_f)]
        task_oxy=data_oxy[(ftask_idx+2*freq_f):(ftask_idx+12*freq_f)]
        
        etask_idx= etask_idx/1000
        etask_idx=round(etask_idx*freq_e)
        task_eeg=data_eeg[(etask_idx+2*freq_e):(etask_idx+12*freq_e)] # [2000, 30]

        tasks_fnirs=np.concatenate((task_deoxy, task_oxy), axis=1) # [100, 72]
        # print("tasks shape",tasks_fnirs.shape)

        task_eeg = task_eeg[1::2]
        tasks_fnirs = tasks_fnirs[1::2]
        eeg.append(task_eeg)
        nirs.append(tasks_fnirs)
        labels.append(y_raw_fnirs[0, i]) 
        
        # e_range=range(0,task_eeg.shape[0],freq_e*slide) # range(0, 2000, 200)
        # for k,j in enumerate(range(0, tasks_fnirs.shape[0],freq_f*slide)): # range(0, 100, 10)
        #     ftask_sample=tasks_fnirs[j:(j+freq_f*window),:] # [j:j+50, :]
        #     etask_sample=task_eeg[e_range[k]:(e_range[k]+freq_e*window),:] # [j':j'+1000, :]

        #     if downsampled: # downsample by 50%
        #         ftask_sample=ftask_sample[1::2]
        #         etask_sample=etask_sample[1::2]
            
        #     if baseline:
        #         eegsample=np.zeros([3,200,32])
        #         eegsample[0,0:200,0:30]=etask_sample[0:200,0:30]
        #         eegsample[1,0:200,0:30]=etask_sample[150:350,0:30]
        #         eegsample[2,0:200,0:30]=etask_sample[300:500,0:30]
        #         eegsample = eegsample.transpose((1, 2, 0))

        #         fnirssample=np.zeros([3,32,72])
        #         fnirssample[0,0:25,0:72]=ftask_sample[0:25,0:72]
        #         fnirssample = fnirssample.transpose((1, 2, 0))

        #     if baseline:
        #             nirs.append(fnirssample)
        #             eeg.append(eegsample)
        #     else:
        #             ftask_sample=np.reshape(ftask_sample,(ftask_sample.shape[0],ftask_sample.shape[1]))
        #             etask_sample=np.reshape(etask_sample,(etask_sample.shape[0],etask_sample.shape[1]))
        #             nirs.append(ftask_sample)
        #             eeg.append(etask_sample)
        #     labels.append(y_raw_fnirs[:,i])

        #     if((j+freq_f*window)>=(tasks_fnirs.shape[0])): # (60, 110) -> overflow
        #         break
        # print("Length of data samples, ",len(labels))

    eeg=np.array(eeg)
    nirs=np.array(nirs)
    labels=np.array(labels)

    eeg = eeg.transpose(0, 2, 1)
    nirs = nirs.transpose(0, 2, 1)

    print(eeg.shape)
    print(nirs.shape)
    print(labels.shape)

    # eeg = torch.tensor(eeg, dtype=torch.float64)
    # nirs = torch.tensor(nirs, dtype=torch.float64)
    # labels = torch.tensor(labels, dtype=torch.long)

    return eeg, nirs, labels

def read_wg_data_cross(eeg_root,nirs_root, baseline, random_seed):
    subjcount = 26
    train_stop=math.floor(subjcount * 0.6) # 60% for training, was 80%

    # Separating subjects data by:
    # a. making an array of subjects numbers
    # b. shuffling them and then use in the trial extraction for the loop below
    subjects_list=list(range(26))
    np.random.seed(random_seed)
    np.random.shuffle(subjects_list)
    print(subjects_list)

    eeg_train = []
    nirs_train = []
    labels_train = []
    eeg_test = []
    nirs_test = []
    labels_test = []

    for s, subject in enumerate(subjects_list):
        subject += 1
        if subject < 10:
            #fnirs data
            file_path_f = nirs_root+"VP00"+str(subject)+"-NIRS/cnt_wg.mat"
            subjectdataf = load(file_path_f, simplify_cells=True)

            #fnirs marks
            file_path2_f = nirs_root+"VP00"+str(subject)+"-NIRS/mrk_wg.mat"
            session_marksf = load(file_path2_f, simplify_cells=True)

            #eeg data
            file_path_e = eeg_root+"VP00"+str(subject)+"-EEG/cnt_wg.mat"
            subjectdatae = load(file_path_e, simplify_cells=True)

            #eeg marks file
            file_path2_e = eeg_root+"VP00"+str(subject)+"-EEG/mrk_wg.mat"
            session_markse = load(file_path2_e, simplify_cells=True)
        else:
            #fnirsdata file
            file_path_f = nirs_root+"VP0"+str(subject)+"-NIRS/cnt_wg.mat"
            subjectdataf = load(file_path_f, simplify_cells=True)

            #fnirs marks file
            file_path2_f = nirs_root+"VP0"+str(subject)+"-NIRS/mrk_wg.mat"
            session_marksf = load(file_path2_f, simplify_cells=True)

            #eeg data
            file_path_e = eeg_root+"VP0"+str(subject)+"-EEG/cnt_wg.mat"
            subjectdatae = load(file_path_e, simplify_cells=True)

            #eeg marks file
            file_path2_e = eeg_root+"VP0"+str(subject)+"-EEG/mrk_wg.mat"
            session_markse = load(file_path2_e, simplify_cells=True)

        # extract class labels from dataset
        y_raw=session_marksf['mrk_wg']['y']
        y_raw=np.array(y_raw)

        # number of datapoints
        data_len=len(subjectdataf['cnt_wg']['deoxy']['x'])
        print('fnirs len',data_len)
        data_len=len(subjectdatae['cnt_wg']['x'])
        print('eeg len',data_len)

        # index of trial start times
        start_times_msF=session_marksf['mrk_wg']['time']
        time_len=len(session_marksf['mrk_wg']['time'])
        start_times_msE=session_markse['mrk_wg']['time']

        # extract eeg and fnirs signals data from dataset
        data_deoxy=subjectdataf['cnt_wg']['deoxy']['x']
        data_oxy=subjectdataf['cnt_wg']['oxy']['x']

        data_eeg=subjectdatae['cnt_wg']['x']

        for i,task_idxF in enumerate(start_times_msF):

            task_idxE=start_times_msE[i]

            # epoching
            # data shape [100, 36] = [trials, channels]
            # the first 2 seconds of data of each trial are discarded, the subsequent 10 seconds of data are kept
            task_idxF= task_idxF/1000 # msec to sec
            task_idxF=round(task_idxF*freq_f) # seconds to samples(freq)
            task_deoxy=data_deoxy[(task_idxF+2*freq_f):(task_idxF+12*freq_f)]
            task_oxy=data_oxy[(task_idxF+2*freq_f):(task_idxF+12*freq_f)]

            task_idxE= task_idxE/1000
            task_idxE=round(task_idxE*freq_e)
            task_eeg=data_eeg[(task_idxE+2*freq_e):(task_idxE+12*freq_e)]

            # attaching oxy to deoxy on the right.
            tasks_fnirs=np.concatenate((task_deoxy, task_oxy), axis=1)

            # these ranges define sliding window starting indices
            e_range=range(0,task_eeg.shape[0],freq_e*slide)
            f_range=range(0,tasks_fnirs.shape[0],freq_f*slide)

            for k,j in enumerate(f_range):
                ftask_sample=tasks_fnirs[j:(j+freq_f*window),:] # 5s sample
                etask_sample=task_eeg[e_range[k]:(e_range[k]+freq_e*window),:]

                # downsample:
                if downsampled:
                    ftask_sample=ftask_sample[1::2]
                    etask_sample=etask_sample[1::2]

                if baseline:
                    # reshape for resnet: 
                    eegsample=np.zeros([3,200,32])
                    eegsample[0,0:200,0:30]=etask_sample[0:200,0:30]
                    eegsample[1,0:200,0:30]=etask_sample[150:350,0:30]
                    eegsample[2,0:200,0:30]=etask_sample[300:500,0:30]
                    eegsample = eegsample.transpose((1, 2, 0))

                    fnirssample=np.zeros([3,32,72])
                    fnirssample[0,0:25,0:72]=ftask_sample[0:25,0:72] # actual data is only 25x72
                    fnirssample[1,0:25,0:72]=ftask_sample[0:25,0:72]
                    fnirssample[2,0:25,0:72]=ftask_sample[0:25,0:72]
                    fnirssample = fnirssample.transpose((1, 2, 0))

                index = s+1
                if(index>train_stop):
                    # appending samples into an X for testing
                    if baseline: # append data as 3 channels
                        eeg_test.append(eegsample)
                        nirs_test.append(fnirssample)
                    else: # append as a features matrix, samplesxchannels - EEG=500,30 FNIRS=25,72
                        nirs_test.append(ftask_sample)
                        eeg_test.append(etask_sample)
                    labels_test.append(y_raw[:,i])
                else: # appending samples into an X for training
                    if baseline:    # append data as 3 channels
                        eeg_train.append(eegsample)
                        nirs_train.append(fnirssample)
                    else: # append as a features matrix, samplesxchannels - EEG=500,30 FNIRS=25,72
                        nirs_train.append(ftask_sample)
                        eeg_train.append(etask_sample)
                    labels_train.append(y_raw[:,i])

                # check if window slide is ending
                if((j+freq_f*window)>=(tasks_fnirs.shape[0])):
                    break;

        print('Subject ',subject+1,'is done. Length of X_train,X_test is now:',len(eeg_train), len(eeg_test),'and length of labels_train, labels_test is: ',len(labels_train), len(labels_test))

    eeg_train=np.array(eeg_train)
    nirs_train=np.array(nirs_train)
    eeg_test=np.array(eeg_test)
    nirs_test=np.array(nirs_test)
    print("eeg_train",eeg_train.shape)
    print("nirs_train",nirs_train.shape)
    print("eeg_test",eeg_test.shape)
    print("nirs_test",nirs_test.shape)

    labels_train=np.array(labels_train)
    print("labels_train",labels_train.shape)
    print("labels_train samples",labels_train[0:7])

    labels_test=np.array(labels_test)
    print("labels_test",labels_test.shape)
    print("labels_test samples",labels_test[0:7])

    return eeg_train, nirs_train, labels_train, eeg_test, nirs_test, labels_test