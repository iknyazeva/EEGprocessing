__author__ = 'Irina Knyazeva'


import pandas as pd
import numpy as np
from numpy.fft import fft,ifft

class EEGAnalyser:

    wt = []
    data =  pd.DataFrame()
    freq = []
    good_trials = []
    norm_data = []
    min_length = 0
    channelsName = []
    num_trials = 0


    def __init__(self, min_freq ,max_freq, num_freq, srate = 500, num_chan = 19):

        """
        Define sample rate and parameters for frequencies analysis
        """
        self.srate = srate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_freq = num_freq
        self.num_chan = num_chan

    def find_minimal_trial(self, timingPath):

        from os import listdir
        from os.path import isfile, join
        files = [join(timingPath, f) for f in listdir(timingPath) if isfile(join(timingPath, f))]
        min_trial_lens = np.min([int(500 * np.min(np.diff(np.loadtxt(f, skiprows=1, usecols=(0,))))) for f in files])
        self.min_length = min_trial_lens

        return min_trial_lens



    def load_data(self, sig_name, t_name, ftr_name):
        '''
        function for loading, segmenting and cutting out failed trials
        sig_name: name of the csv file with the signal
        t_name: name of the TXT file with timing
        ftr_name: name of the txt file with labels of failed trials
        srate: sampling rate of signal
        '''

        #load csv file
        df = pd.read_csv(sig_name, sep = ',', header=0)

        p =np.loadtxt(ftr_name)
        #загружаем файл с номерами неудачных проб
        #преобразуем номера неудачных проб в int
        p = p.astype(int)
        #загружаем файл с метками начала новой пробы по времени
        t = np.loadtxt(t_name, skiprows=1, usecols=(0,))
        #переводим метки из секунд в отсчеты (2 ms)
        t = t*self.srate
        #преобразуем номера отсчетов в int
        t = t.astype(int)
        #помечаем отсчеты до первой пробы как -1
        df.iloc[:t[0], 19] = -1
        #помечаем все остальные отсчеты номерами проб, к которым они относятся
        for i in range(len(t)-1):
            df.iloc[t[i]:t[i+1], 19] = i+1

        #помечаем последнюю пробу
        df.iloc[t[-1]:, 19] = len(t)
        #копируем данные (не знаю зачем, на всякий случай)

        #удаляем неудачные пробы
        for k in range(len(p)):
            df.drop(df.index[df['LABEL'] == p[k]], inplace = True)

        #удаляем отсчеты до начала первой пробы
        df.drop(df.index[df['LABEL'] == -1], inplace = True)
        df = df.reset_index(drop=True)

        self.data = df
        self.channelsName  = list(df.columns[:-1])


        return  df

    def initializeWavelets(self):
        self.wt = np.zeros(shape = (self.num_freq,self.data.shape[0], self.data.shape[1]), dtype=complex)


    def normalize_data(self, dropOrgignalData = True):
        """
        Function for data normalization: concatenating all trials of equal min_trial_length
        in one series
        """
        sig  = self.data.as_matrix()
        L = sig[:, 19].real.astype(int)
        sig = np.delete(sig, 19, 1)
        listTrials = np.unique(L)
        self.good_trials = listTrials



        #calculating minimal trial length
        if (self.min_length == 0):
            trial_lengths = np.zeros(len(listTrials))
            for i in range(0, len(N)):
                trial_lengths[i] = np.where(L == listTrials[i])[0].shape[0]
            self.min_length = int(np.amin(trial_lengths))



        #Merge all data in one big series with equal trial length
        #check last trial
        lenLastTrial = np.where(L == listTrials[-1:])[0].shape[0]
        if (lenLastTrial<self.min_length):
            listTrials = listTrials[:-1]
        self.num_trials = len(listTrials)

        for k in np.arange(0, listTrials.shape[0]):
            start = np.where(L == listTrials[k])[0][0]

            if (k==0):
                alldata = sig[start:(start+self.min_length), :]
            else:
                alldata = np.vstack((alldata,sig[start:(start+self.min_length), :]))

        alldataDF = pd.DataFrame(data=alldata, columns=self.data.columns[:-1])
        if dropOrgignalData:
            self.data = alldataDF
            self.norm_data = False
        else:
            self.norm_data = alldataDF

        return alldataDF

    def computeERP(self):
        num_trials = self.num_trials
        min_trial_len = self.min_length
        num_chan = self.num_chan
        assert self.data.shape[0] > 0, "You need to define EEG data for wavelet transform first"
        if (self.norm_data == False):
            data = self.data.values
        elif (len(self.norm_data) == 0):
            data = self.data.values
        else:
            data = self.norm_data.values


        ERP = np.mean(np.reshape(data, [min_trial_len, num_trials, num_chan], order='F'), axis=1)

        return ERP


    def wavelet_transform(self, elList = None, print_ = True):

        """
        Function for wavelet transform computing


        elList: list of electrodes, for ex. [0,2,3], if null  - compute all electrodes
        return wt: wavelet transformation for time series or matrix

        freq: vector of used frequencies in Hz
        srate: sampling rate in Hz
        min_freq: minimal frequency of wavelet
        max_freq: max frequency of wavelet
        num_freq: number of frequency in interval from min to max

        """

        assert self.data.shape[0] > 0, "You need to define EEG data for wavelet transform first"
        if (self.norm_data == False) :
            data = self.data.values
        elif (len(self.norm_data)==0) :
            data = self.data.values
        else:
            data = self.norm_data.values


        freq = np.logspace(np.log10(self.min_freq),np.log10(self.max_freq),self.num_freq)
        range_cycles = [4, 8]
        s = np.logspace(np.log10(range_cycles[0]),np.log10(range_cycles[1]),self.num_freq)/(2*np.pi*freq)
        wavtime = np.arange(-1,1+1/self.srate,1/self.srate)
        half_wave =  np.floor((len(wavtime)-1)/2).astype(int)

        nWave = wavtime.shape[0]

        nData = data.shape[0]
        nConv = nWave + nData - 1

        if elList == None:
            elList = list(range(0, data.shape[1]))

        if len(self.wt) == 0:
            self.initializeWavelets()

        for el in elList:
            if print_:
                print("Compute channel: ",el+1)
            dataX = fft(data[:, el],nConv)
            for fi in range(0, self.num_freq):
                wavelet = np.exp(2*1j*np.pi*freq[fi]*wavtime)* np.exp(-np.power(wavtime,2)/(2*np.power(s[fi],2)))
                waveletX = fft(wavelet,nConv)

                convData = ifft(waveletX * dataX)
                convData = convData[half_wave:-half_wave]
                self.wt[fi, :, el] = convData

       # self.wt = wt
        self.freq = freq
        return (freq, self.wt[:,:,elList])

    def baseline_normalization(self, prestimulInterval = 150, typeNorm='Z', trial_average=True):
        """
        wt: array with wavelet transform, where shape  = (frequeencies, time, channels)
        prestimulInterval: interval for baseline,
        typeNorm: string with type normalization, could be 'Z', or 'DB'
        trial_average: if True at first all trials averaged and than baseline computes
        """



        wt = self.wt

        num_trials = self.num_trials
        min_trial_len = self.min_length
        num_chanels = self.num_chan
        num_freq = self.num_freq

        if trial_average:
            NormWT = np.zeros([num_freq, min_trial_len, num_chanels])
        else:
            NormWT = np.zeros([num_freq, min_trial_len, num_trials, num_chanels])

        for ch in range(num_chanels):
            wt_ch = np.squeeze(wt[:, :, ch])

            if abs(wt_ch[0,0])>0:

                if trial_average:

                    temppower = np.median(np.absolute(np.reshape(wt_ch, [num_freq, min_trial_len, num_trials], order="F")) ** 2, axis=2)
                    mean_vector = np.mean(temppower[:, :prestimulInterval], axis=1)
                    if (typeNorm == 'Z'):
                        std_vector = np.std(temppower[:, :prestimulInterval], axis=1)
                        NormWT[:, :, ch] = (temppower - mean_vector[:, None]) / std_vector[:, None]
                    else:
                        NormWT[:, :, ch] = 10 * np.log10(temppower / mean_vector[:, None])
                else:
                    temppower = np.absolute(np.reshape(wt_ch, [num_freq, min_trial_len, num_trials], order="F")) ** 2
                    mean_vector = np.mean(temppower[:, :prestimulInterval, :], axis=1)
                    if (typeNorm == 'Z'):
                        std_vector = np.std(temppower[:, :prestimulInterval, :], axis=1)
                        NormWT[:, :, :, ch] = (temppower - mean_vector[:, None]) / std_vector[:, None]
                    else:
                        NormWT[:, :, :, ch] = 10 * np.log10(temppower / mean_vector[:, None])

        return NormWT


    def phase_coherence(self, chanel1_id, chanel2_id, num_points = None, trial_average = True):
        """

        num_points: num_points for computing
        Function for wavelet coherence  computing
        based on experiment results
        now accept data after wavelet transform
        """

        #length of time window for phase averaging from 1.5 for lowest freq to 3 cycles
        timewindow = np.linspace(1.5,3,self.num_freq)
        #length of the largest time window in points
        time_window_largest = np.floor((1000/self.freq[0])*timewindow[0]/(1000/self.srate)).astype(int)
        if num_points == None:
            times2saveidx = np.arange(time_window_largest,self.min_length-time_window_largest)
        else:
            times2saveidx = np.linspace(time_window_largest,self.min_length-time_window_largest,num_points).astype(int)

        if trial_average:
            ispc = np.zeros(shape = (self.num_freq,len(times2saveidx)), dtype=float)
        else:
            ispc = np.zeros(shape = (self.num_freq,len(times2saveidx), self.num_trials), dtype=float)


        if len(self.wt) == 0:
            self.wavelet_transform(elList = [chanel1_id, chanel2_id])
        else:
            if np.sum(abs(self.wt[:,:,chanel1_id]))==0:
                self.wavelet_transform(elList=[chanel1_id])
            if np.sum(abs(self.wt[:, :, chanel2_id])) == 0:
                self.wavelet_transform(elList=[chanel2_id])

        data1 = self.wt[:, :, chanel1_id]
        data2 = self.wt[:, :, chanel2_id]

        for fi in range(0, self.num_freq):

            phase_sig1 = np.angle(data1[fi, :].reshape(self.min_length,self.num_trials,order='F'))
            phase_sig2 = np.angle(data2[fi, :].reshape(self.min_length,self.num_trials,order='F'))

            #Phase difference between the signals
            phase_diffs = phase_sig1-phase_sig2

            #Averaging in the sliding window
            #compute time window in indices for this frequency and average inside the window
            time_window_idx = np.floor((1000/self.freq[fi])*timewindow[fi]/(1000/self.srate)).astype(int)
            for ti in range(0,len(times2saveidx)):
                phasesynch = abs(np.mean(np.exp(1j*phase_diffs[times2saveidx[ti]-time_window_idx:times2saveidx[ti]+time_window_idx,:]),axis = 0))

                # then average over trials
                if trial_average:
                    ispc[fi,ti] = np.mean(phasesynch)
                else:
                    ispc[fi,ti,:] = phasesynch

        phase_syncr = {'ispc': ispc, 'chanIds': [chanel1_id, chanel2_id], 'times': times2saveidx}
        return phase_syncr


    #Computing phase coherence

    # Spectral Coherence (Magnitude-Squared Coherence)
    def spectral_coher(self, chanel1_id, chanel2_id, num_points = 300):

        """
        compute spectral coherence by wavelet transformed data
        """

        spectcoher = np.zeros(shape=(self.num_freq, num_points), dtype=float)
        times2saveidx = np.linspace(0,self.min_length-1,num_points).astype(int)

        data1 = self.wt[:,:,chanel1_id]
        data2 = self.wt[:,:,chanel2_id]

        for fi in range(0, self.num_freq):
            sig1 = data1[fi, :].reshape(self.min_length,self.num_trials,order='F')
            sig2 = data2[fi, :].reshape(self.min_length,self.num_trials,order='F')

            spec1 = np.mean(np.power(abs(sig1),2), 1)
            spec2 = np.mean(np.power(abs(sig1),2), 1)
            cross_spec = np.power(abs(np.mean(sig1*np.conj(sig2), 1)), 2)
            spectcoher[fi, :] = cross_spec[times2saveidx]/(spec1[times2saveidx] * spec2[times2saveidx])

        self.spect_syncr  = {'spc':spectcoher,'chanIds':[chanel1_id,chanel2_id],'times': times2saveidx}



