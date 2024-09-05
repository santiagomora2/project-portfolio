#------------------------------------------------------------------------------------------------------------------
#   Online speech classification
#------------------------------------------------------------------------------------------------------------------

import time
import queue
import numpy as np
import threading
import sounddevice as sd
from scipy import signal
from python_speech_features import mfcc
import joblib

##### Classifier loading #####

clf, ref_cols, target = joblib.load('CLF.pkl')

##### Data acquisition configuration #####

# Device configuration
fs=44100    
channels = 2
recording_time = 2

# Buffers for data aquisition
buffer_size = int(2 * fs * recording_time)
circular_buffer = np.random.rand(buffer_size, channels)
audio_queue = queue.Queue()

# Flag to stop de audio recording
stop_recording_flag = threading.Event()

# Callback for audio recording
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# Function that runs in another thread for audio acquisition
def record_audio():
    with sd.InputStream(samplerate=fs, channels=channels, callback=audio_callback):
        while not stop_recording_flag.is_set():
            try:
                indata = audio_queue.get(timeout=0.1)
                if indata is not None:                    
                    global circular_buffer
                    circular_buffer = np.roll(circular_buffer, -len(indata), axis=0)
                    circular_buffer[-len(indata):, :] = indata                     
            except queue.Empty:
                continue

# Function for stopping the audio acquisition
def stop_recording():
    stop_recording_flag.set()
    recording_thread.join()
    
# Helper function that obtains the last N seconds of audio recording
def get_last_n_seconds(n_seconds):
    samples = int(fs * n_seconds)
    return circular_buffer[-samples:]

# Start data acquisition
recording_thread = threading.Thread(target=record_audio, daemon=True)
recording_thread.start()

##### Online classification #####
preparation_time = 0.3

while (True):

    input("<< Press enter to record audio >>")

    # Preparation time
    time.sleep(preparation_time)    

    # Record audio
    time.sleep(recording_time)
    new_record = get_last_n_seconds(recording_time)
    #print(new_record)

    ####################################################
    # Process and classify new record                  #
    ####################################################

    filt = signal.iirfilter(4, [10, 15000], rs=60, btype='band',
                       analog=False, ftype='cheby2', fs=fs,
                       output='ba')
    filtered = []

    ff1 = signal.filtfilt(filt[0], filt[1],x = new_record[:, 0], method='gust')
    ff2 = signal.filtfilt(filt[0], filt[1],x = new_record[:, 1], method='gust')
    filtered.append(np.column_stack((ff1, ff2)))

    features = []
    for tr in filtered:
        mfcc_feat = mfcc(tr, fs, nfft = 2048)
        features.append(mfcc_feat.flatten())

    # Build x array
    x = np.array(features)

    print(clf.predict(x)[0])
    results = {1: 'Siguiente', 2: 'Anterior', 3: 'Mas', 4: 'Menos', 5:'Pausa', 6:'Continua'}

    print('Dijiste: ' + results[clf.predict(x)[0]])

# Stop audio acquisition
stop_recording()
#------------------------------------------------------------------------------------------------------------------
#   End of file
#------------------------------------------------------------------------------------------------------------------
