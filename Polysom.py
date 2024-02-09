#################################################
### José Javier Gutierrez Gil
### jogugil@gmail.com
#################################################
import pickle
from scipy.signal import welch
import os
import sys

import warnings
import pandas as pd
import time


import numpy as np
import matplotlib.pyplot as plt
import mne
import yasa
from mne.datasets.brainstorm import bst_auditory
from mne.io import read_raw_ctf
from mne.preprocessing import annotate_movement, compute_average_dev_head_t
import numpy as np
from scipy.signal import resample
from scipy.signal import welch

class Polysom ():
    subject        = None # Sujeto de estudio asociado al fichero que contiene su PSG
    #raw_data      = None # Datos en bruto de los canales del PSG. --123-- vercomo serializar y no mantener en memoria
    #raw_cropped   = None # Datos en bruto cortados del PSG.  --123-- vercomo serializar y no mantener en memoria
 
    crop           = False # Indica que el PSG se ha reducido en tiempo. En vez de 8h tener 2 o 4h
    poly_path      = None  # Path con nombre de fichero donde se guarda el fichero que contieneel PSG del sujeto
    config_poly    = None  # Objeto que contiene los parámetros de configuración del trabajo
    channels       = None  # Nombre de lso canales que se tienen en el objeto raw:data o en su defecto raw_cropped
    decim          = None  # Parámetro para diezzmar la señal una vez segmentada. (Es para remuestrear la señal y reducirla)
    current_sfreq  = None  # Frecuencia de meustreo del PSG
    desired_sfreq  = 100   # Frecuencia de meustreo deseada para el estudio sobre las señales del PSG. Por defecto 100 ya que la fs deberá ser <50 y para el objetivo del trabajo es perfecta (reducimos señal y eliminamos frecuencias altas)
    obtained_sfreq = None  # parámetro para la segmentación de las señales del PSG
    
    paper_optim_features      = None # DataFrame con las caracteristicas del paper seleccionadas automaticamente (datos normalizados)
    yasa_optim_features       = None # DataFrame con las caracteristicas del api yasa seleccionadas automaticamente (datos normalizados)
    
    eeg_raw_resam             = None # Señales de los canales eeg cargados del fichero edf
    eeg_raw_resam_nms         = None # Señales de los canales eeg y eog cargados del fichero edf
    eeg_eog_raw_resam         = None # conjunto de señales de lso canales eeg y eog cargados del fichero edf
    eeg_eog_raw_resam_nms     = None # conjunto de señales de lso canales eeg y eog cargados del fichero edf
    paper_feature_norm_matrix = None # matriz de carateríssiticas del paper normaiizada. Datos de entrada del modelo
    yasa_feature_norm_matrix  = None # matriz de carateríssiticas del api yasa normaiizada. Datos de entrada del modelo
    
    serialized_data      = None # Serielización datos fichero edf
    serialized_crop_data = None #serialización de las señales limitadas en tiempo 
    
    labels = None # Lista que contiene los labels de los dos expertos para un sujeto determinado
    
    ####################################################
    ## 
    ##  events = mne.find_events(raw_data)
    ##  epochs = mne.Epochs(raw_data, events, decim = decim)  --> con esta  función podemos diezmar la señal y remuestrearla a la frecuencia deseada
    ##
    ##################################################
    
    ##### NOTAS:
    ## Cuando se realiza cualquier tipo de filtro de la señal se modifica el objeto raw_data
    ##
    ## Los datos deben ser reescalados a 1e6 para trabajar en microV. data = raw.get_data () * 1e6 (no se modifica el objeto raw_data, ni se mantiene en memoria. Cada vez que se coja el conjunto de datos se realiza la operacion)
    ##
    ## Eliminamos las 30 últimas muestras --123-- porque lo indica el artículo de referencia   data = data [:,:-30*30*100] (no se modifica el objeto raw_data, ni se mantiene en memoria. Cada vez que se coja el conjunto de datos se realiza la operacion)
    ## 
    
    def __init__(self, file_path, subject, config):
        try:
            print ('file_path:',file_path)
            print ('subject:',subject) 
            path = "%s/%s.edf" % (file_path, subject)
            #if (DEBUG): 
                #print (path)
            print ('path:',path)
           # Leer el archivo EDF con MNE-Python
            raw_data      = mne.io.read_raw_edf (path, preload = True) # cargamos los datos en bruto de los canales del PSG del subject 
            
            
            self.config_poly   = config
            self.subject       = subject
            self.poly_path     = path
            self.channels      = raw_data.ch_names
            
            # Para la decimación y remuestreo posterior de las señales
            self.desired_sfreq  = self.config_poly ['desired_fs']
            self.current_sfreq  = raw_data.info ["sfreq"]
            self.decim          = np.round (self.current_sfreq / self.desired_sfreq).astype (int)
            self.obtained_sfreq = self.current_sfreq / self.decim
            
            # Filtro paso baja para despues diezmar la señal. Remuestreamos y eliminamos las altas
            # frecuencias de posibles artefactos
            lowpass_freq  = self.obtained_sfreq / 3.0
            raw_data = raw_data.copy ().filter (l_freq = None, h_freq = lowpass_freq)
            
           
            self.serialized_raw_data (raw_data)
            
            
            #if (DEBUG):
                # Obtener información sobre las señales y el muestreo
                #print (raw_data.info)   # Imprimir información general sobre las señales

        except Exception as e:
            print (f" An error occurred while reading the EDF file {str (e)}")
            
    ####################################################
    def serialized_raw_data (self, raw_data):
        if raw_data is not None:
            # Especifica el nombre del archivo en el que deseas guardar el objeto serializado
            self.serialized_data = f'mne_data_{self.subject}.pkl'

            # Serializar el objeto mne y guardarlo en el archivo
            with open (self.serialized_data, 'wb') as data_file:
                pickle.dump (raw_data, data_file)
                
            self.raw_data = None

            
    ####################################################
    def get_raw_oData (self):
        '''
            Datos en bruto de las señales del PSG. Buscaremos la manera de serializar este objeto y no tenerlo
            en memoria siempre. --123-- Cuando se serialice, el objeto raw_data será privado y sóo se podrá
            acceder desde las funciones getXXX ()
        '''
        mne_data = None
        # Cargar el objeto serializado desde el archivo
        with open(self.serialized_data, 'rb') as data_file:
            mne_data = pickle.load (data_file)
            
        return mne_data
    
    def get_raw_eData (self):
        '''
            Datos modificados de las señales del PSG. Buscaremos la manera de serializar este objeto y no tenerlo
            en memoria siempre. Datos mne serializados
        '''
        
        mne_data = None
          
        if self.crop: # La funcionalidad crop aún no está implementada
            # Cargar el objeto serializado desde el archivo
            with open(self.serialized_crop_data, 'rb') as data_file:
                mne_data = pickle.load (data_file)
        else:
            # Cargar el objeto serializado desde el archivo
            with open(self.serialized_data, 'rb') as data_file:
                mne_data = pickle.load (data_file)
                
        return mne_data
          
    def getData (self):
        '''
            CUIDADO! si se accede directamente a los datos hay que cambiar la escala!
            Se trabajará en microVolts (porque mne trabaja en V). Y eliminamos las 30
            últimas porque lo indica el artículo de referencia. Aqui no se realia dichas operaciones
            para poder trabajar en bruto y ser más flexibles.
        '''
        #data = self.get_raw_eData ().get_data () * 1e6 # microVolts (porque mne trabaja en V)
        #data = data [:,:-30*30*100] # eliminamos las 30 últimas porque lo indica el artículo de referencia
        return self.get_raw_eData ().get_data ()
 
    def getDataDF (self):
        '''
            Devolvemos las señales en formato DataFrame de Pandas. Cada columna tendrá un canal
            Notar que hemos reescalado los datos 1e6 y eliminado las 30 últimas muestras pro prescripción
            del paper de referencia. Aqui no se realiza dicha operación, loq ue significa que 
            en algún momento se debe trasnformar la escala y eliminar las 30 últimas muestras.
        '''
        data = self.getData ()
        return (pd.DataFrame(data.T, columns = self.get_raw_eData ().ch_names)) 
    
    def getChannels (self):
        return self.get_raw_eData ().info['ch_names']
    
    def getDataType (self, channel_names):
        '''
              Obtenermos el conjunto de canales directamente del objeto raw_data
        '''
        #print (channel_names)
        raw  = self.get_raw_eData ()
        #print (raw)
        return raw.copy().pick (channel_names)
    
    def doWSegment (self, data_raw, sf = 200, sw = 30):
        '''
             Devuelve un objeto que contiene las señales segmentada por ventanas. 
             A la función se le pasa un objeto con las señales que desea segmentar.
             ojo!!! No es un objeto mns-python sino ndarray.numpy 3D  (epochs, canales, muestras).  
        '''
        #dt = data_raw.get_data () * 1e6 # microVolts (porque mne trabaja en V)
        #dt = dt [:,:-30*30*100] # eliminamos las 30 últimas porque lo indica el artículo de referencia
        # Segmentamos en oposcs de sw = 30 segundos- En este punto aún no se remuestreó por lo que 
        # la frecuencia de muestreo sigue siendo sf = 200.
        times, data_win = yasa.sliding_window(data_raw.get_data (), sf = sf, window = sw)

        return times, data_win
    
    def getDataWSegment (self, sw = 30):
        '''
             Devuelve un objeto que contiene las señales segmentada por ventanas. 
             ojo!!! No es un objeto mns-python sino ndarray.numpy 3D  (epochs, canales, muestras)
        '''
      	# Segmentamos en oposcs de sw = 30 segundos- En este punto aún no se remuestreó por lo que 
        # la frecuencia de muestreo sigue siendo sf = 200 
        # (obtenemos la frecuencia de meustreo del fichero de configuración).
        sf = self.info () ['sfreq'] 
         
       
        _, data = yasa.sliding_window(self.getData (), sf = sf, window = sw)
         
        return data
    
    def doSegmentResample (self, data_raw, of = 200, sf = 100, sw = 30):
        '''
            La función  primero segmenta cada una de las señales de cada canal en epochs de tamaño de ventana sw.
            Posteriormente hace un remuestreo de las señales con la decimación deseada al crear el objeto. 
            Devuelve una matriz  (epochs, canales, muestras) remuestreada a la frecuencia deseada.
        '''
        
        
        # segmentamos la señal en epochs con tamaño de ventaana sw
        times, data_win = self.doWSegment (data_raw, sf = of, sw = sw)
        
        # Definir la nueva cantidad de muestras por ventana deseada
        nuevas_muestras_por_ventana = int (data_win.shape [2] / self.decim)  # Ajusta según tus necesidades

        # Realizar el re-muestreo para cada ventana y canal
        data_resample = np.apply_along_axis (lambda x: resample (x, nuevas_muestras_por_ventana), 2, data_win)
        
        return times, data_resample
    
    def getDataSegmentResample (self, sw = 30):
        '''
            La función  primero segmenta cada una de las señales de cada canal en epochs de tamaño de ventana sw.
            Posteriormente hace un remuestreo de las señales con la decimación deseada al crear el objeto. 
            Devuelve una matriz  (epochs, canales, muestras) remuestreada a la frecuencia deseada.
        '''
        # segmentamos la señal en epochs con tamaño de ventaana sw
        data_segment = self.getDataWSegment (sw = sw)
        
        # Definir la nueva cantidad de muestras por ventana deseada
        nuevas_muestras_por_ventana = int (data_segment.shape [2] / self.decim)  # Ajusta según tus necesidades

        # Realizar el re-muestreo para cada ventana y canal
        data_resample = np.apply_along_axis (lambda x: resample (x, nuevas_muestras_por_ventana), 2, data_segment)
        
        return data_resample
    
    def getWelchDataSegmentResample (self, sw = 30, origin = False):
        ''' 
            Devuelve el espectro de las ventanas de cada señal. Se calcula
            el conjunto de datos segmentado por ventana y remuestreado ('data_resample'). 
            Si origin es True devuelve el espectro de los datos originales sin segmentar.
            Devuelve freqs, psd_signal
        '''
        data = self.getData ()
        if (origin):
            data = self.getDataSegmentResample (sw = sw)
        
        sf         = self.info () ['sfreq'] #4
        win        = int (sw * sf)   
        freqs, psd = welch (data, sf, nperseg = win, axis = -1) 

        return freqs, psd
    
    def getBandPowertWelch (self, sw = 30, origin = False):
        '''
            Calculamos la potencia en bandas de cada canal (periodograma de Welch)
        '''  
        data = self.getData ()
        if (origin):
            data = self.getDataSegmentResample (sw = sw)

        sf       = self.info () ['sfreq'] #1
        channels = self.get_raw_eData ().ch_names
        return yasa.bandpower (data, sf = sf, ch_names = channels, bandpass = True)
    
    def getBandPowertDataSegmentResample (self, sw = 30, origin = False):
        '''
            Devuelve la potencia de las ventanas de cada señal. Se crea
            el conjunto de datos segmentado por epochs y remuestreado. 
            Si origin = True, devuelve la banda de potencia de los datos originales
        '''
        data = self.getData ()
        
        if (origin):
            data = self.getDataSegmentResample (sw = sw)
            
        sf  = self.info () ['sfreq'] #2
        win = int (sw * sf)  
        
        # Bandas e intervalos de frecuencia de las mismas
        bands = [(0.5, 4, 'Delta'), (4, 8, 'Theta'), (8, 12, 'Alpha'), 
                    (12, 16, 'Sigma'), (16, 30, 'Beta')]

        freqs, psd = welch (data, sf, nperseg = win, axis = -1) 
        
        # Calculate the bandpower on 3-D PSD array
        bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs, bands)
        
        return bandpower
    
    ################################################
    def set_labels (self,labels):
        self.labels = labels
        
    def get_labels (self):
        return self.labels
    
    def set_eeg_raw_resam  (self, eeg_raw_resam, eeg_raw_resam_nms):
        self.eeg_raw_resam     = eeg_raw_resam
        self.eeg_raw_resam_nms = eeg_raw_resam_nms
        
    def set_eeg_eog_raw_resam (self, eeg_eog_raw_resam, eeg_eog_raw_resam_nms):
        self.eeg_eog_raw_resam     = eeg_eog_raw_resam   
        self.eeg_eog_raw_resam_nms = eeg_eog_raw_resam_nms
        
    def set_paper_feature_norm_matrix (self, paper_feature_norm_matrix):
        self.paper_feature_norm_matrix = paper_feature_norm_matrix
    
    def set_yasa_feature_norm_matrix (self, yasa_feature_norm_matrix):
        self.yasa_feature_norm_matrix = yasa_feature_norm_matrix
    
    def get_eeg_raw_resam (self):
        return self.eeg_raw_resam, self.eeg_raw_resam_nms
    
    def get_eeg_eog_raw_resam (self):
        return self.eeg_eog_raw_resam, self.eeg_eog_raw_resam_nms  
    
    def get_paper_feature_norm_matrix (self):
        return self.paper_feature_norm_matrix
    
    def get_yasa_feature_norm_matrix (self):
        return self.yasa_feature_norm_matrix 
    
    def set_paper_optim_features (self, paper_optim_features):
        self.paper_optim_features = paper_optim_features
    
    def set_yasa_optim_features (self, yasa_optim_features):
        self.yasa_optim_features = yasa_optim_features
    
    def get_paper_optim_features (self):
        df = self.get_paper_feature_norm_matrix ().iloc[:, self.paper_optim_features]
        return df
    
    def get_yasa_optim_features (self):
        df = self.get_yasa_feature_norm_matrix ().iloc[:, self.yasa_optim_features]
        return df
 
    def get_tupla_paper_feeatures_labels (self):
        return [(self.get_paper_feature_norm_matrix (), self.get_labels()[0]), (self.get_paper_feature_norm_matrix (), self.get_labels()[1])]
    
    def get_tupla_yasa_feeatures_labels (self):
        return [(self.get_yasa_feature_norm_matrix (), self.get_labels()[0]), (self.get_yasa_feature_norm_matrix (), self.get_labels()[1])]

    #####################
    def getEEG (self):
        #print (self.config_poly ['channels_eeg'].split(','))
        rraw =  self.getDataType (self.config_poly ['channels_eeg'].strip().split(','))
        #rraw.filter(0.3, 35) #no hay que filtrarlos los datos. Ya vienen filtrados 
        return rraw 
    
    def getEOG (self):
        rraw =  self.getDataType (self.config_poly ['channels_eog'].strip().split(','))
        #rraw.filter(0.3, 35) ) #no hay que filtrarlos os datos. Ya vienen filtrados 
        return rraw 
 
    def getEMG (self):
        rraw =  self.getDataType (self.config_poly ['channels_emg'].strip().split(','))
        #rraw.filter (10, 70) ) #no hay que filtrarlos o los datos. ya vienen filtrados 
        return rraw 
 
    def getECG (self):
        rraw =  self.getDataType (self.config_poly.channel_ecg)
        return rraw 
    
    def getABDOM (self):
        rraw =  self.getDataType (self.config_poly ['channels_abdom'].strip().split(','))
        return rraw 
    
    def getChnlMain (self):
        rraw =  self.getDataType (self.config_poly ['channels_main'].strip().split(','))
        return rraw 
    
    def getChannels (self, eeg = False, eog = False, emg = False, ecg = False,  sao2 = False, abdom = False):
        chn = []
        if eeg: chn.extend (self.config_poly ['channels_eeg'].strip().split(','))
        if eog: chn.extend ( self.config_poly ['channels_eog'].strip().split(',')) 
        if emg: chn.extend ( self.config_poly ['channels_emg'].strip().split(',')) 
        if ecg: chn.extend (self.config_poly.channel_ecg)
        if abdom: chn.extend (self.config_poly ['channels_abdom'].strip().split(','))
        if sao2: chn.extend  ('SaO2') 
        
        return self.getDataType (chn)
         
    ##### Obtención de información del fichero PSG
    def info (self):
         return self.get_raw_eData ().info
        
    def miscellaneous (self):
        
        raw  = self.get_raw_eData ()
        
        n_time_samps = raw.n_times
        time_secs    = raw.times
        ch_names     = raw.ch_names
        n_chan       = len (ch_names)  # note: there is no raw.n_channels attribute
        print (
            "the (cropped) sample data object has {} time samples and {} channels."
            "".format (n_time_samps, n_chan)
        )
        print ("The last time sample is at {} seconds.".format (time_secs[-1]))
        print ("The first few channel names are {}.".format (", ".join(ch_names[:3])))
        print ()  # insert a blank line in the output

        # some examples of raw.info:
        print ("bad channels:", raw.info ["bads"])  # chs marked "bad" during acquisition
        print (raw.info ["sfreq"], "Hz")  # sampling frequency
        print (raw.info ["description"], "\n")  # miscellaneous acquisition info

        print (raw.info)
    
    #### Visualización:
    
    def plot_signal (self, fs = 100):   
        '''
            Muestra las señales presentes en cada uno de los canales del PSG
        '''
        df   = self.getDataDF () 
        time = np.arange (len (df)) / fs

        fig, axes = plt.subplots (len (df.columns), 1, sharex = True, sharey = False, figsize = (20, len (df.columns)*2))
        for (ax, column) in zip (axes, df):
            ax.plot (time, df [column])
            ax.set_title (column)

        plt.xlim (time[[0,-1]])
        plt.show ()  
        
    def plot_signal_seg (self, seg = 60, fs = 100):   
        '''
            Muestra los seg de las señales presentes en cada uno de los canales del PSG
        '''    
        df   = self.getDataDF () 
        time = np.arange (len (df)) / fs
        
        fig, axes = plt.subplots (len (df.columns), 1, sharex = True, sharey = False, figsize = (20, len (df.columns)*2))
        for (ax, column) in zip (axes, df):
            ax.plot (time [0:(seg*fs)], df [column][0:(seg*fs)])
            ax.set_title (column)

        #plt.xlim (time [[0,-1]])
        plt.show ()  
        
    def plot_polysom (self, seg = 60):
        '''
             Visualizar los primeros segundos de las señales de cada uno de los
             canales presentes en el objeto raw_data. Se utiliza el api mne-python 
             para su representaciñon
        '''
        raw = self.get_raw_eData ()
        raw.plot (duration = seg, scalings = 'auto')  
    
   
    def plot_signal_chn (self, channels):
        '''
        	NOTA: EL API mne-python tiene fallos ya que el conjunto de datos no tiene
            bien registrado el tipo de señal que corresponde. Se emte una consulta a mne
            y sólo indican que deben de modificarse y anotarse los registros de los datos 
            para que la API mne pueda funcionar. 
            Se muestra el PSG sólo con los canales pasados en el parámetro 'channels'.
            Se utiliza el API mne-python para representar el PSG --123-- error, no funciona
        '''
  
        data, time = self.get_raw_eData () [channels] #seleccionamos canal chnl

            
        time_h = time / 3600
        plt.figure (figsize = (20,5))
        plt.plot (time_h, data.ravel ()*1e6) #pasamos de V a uV
        plt.xlabel ("time (h)")
        plt.ylabel (f"channel  {self.channels [channels]} (uV)")
        plt.show ()
        
    def plot_signal_psd (self, average = False):
        '''
            Se muestra el espectro de frecuencia de las señales. Se utiliza
            el API mne-python para su representación
        '''
          # Configura Matplotlib para el modo inline
 
        raw = self.get_raw_eData ()
 
        spectrum = raw.compute_psd ()
        # Configura el título de la figura con el nombre del archivo
        fig = spectrum.plot(average=average, picks="data", exclude="bads")
        fig.suptitle(f'Nombre del archivo:  {self.subject}.edf', fontsize=12)

        # Muestra la figura
        plt.show()
            
    #### Varios
    def crop_data (self, tmin, tmax):
        '''
            Recortamos la señal entre los segundos tmin y tmax. Se crea un nuevo objeto 
            raw_cropped
        '''
        raw = self.get_raw_eData ().copy ()
        raw.crop (tmin = 1*tmin, tmax = 2*tmax)
        
        self.serialized_crop_data = f'mne_crop_data_{self.subject}.pkl'

        # Serializar el objeto mne y guardarlo en el archivo
        with open (self.serialized_crop_data, 'wb') as data_file:
            pickle.dump (raw, data_file)
        
        self.crop = True
 
        return self.raw_cropped  