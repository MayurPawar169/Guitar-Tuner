import copy
import os
import numpy as np
import scipy.fftpack
import sounddevice as sd
import time

from tkinter import * 

from threading import *

class Tuner(Thread):
    def run(self):

        SAMPLE_FREQ = 48000
        WINDOW_SIZE = 48000
        WINDOW_STEP = 12000
        NUM_HPS = 5  
        POWER_THRESH = 1e-6  
        CONCERT_PITCH = 440  
        WHITE_NOISE_THRESH = 0.2

        WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ  
        SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ  
        
        DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE
        OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

        ALL_NOTES = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]


        def find_closest_note(pitch):
            i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
            closest_note = ALL_NOTES[i % 12] + str(4 + (i + 9) // 12)
            closest_pitch = CONCERT_PITCH*2**(i/12)
            return closest_note, closest_pitch


        HANN_WINDOW = np.hanning(WINDOW_SIZE)


        def callback(indata, frames, time, status):
            if not hasattr(callback, "window_samples"):
                callback.window_samples = [0 for _ in range(WINDOW_SIZE)]
            if not hasattr(callback, "noteBuffer"):
                callback.noteBuffer = ["1", "2"]

            if status:
                print(status)
                return
            if any(indata):
                callback.window_samples = np.concatenate(
                    (callback.window_samples, indata[:, 0]))  # append new samples
                callback.window_samples = callback.window_samples[len(
                    indata[:, 0]):]  # remove old samples

                # skip if signal power is too low
                signal_power = (np.linalg.norm(callback.window_samples,
                                ord=2)**2) / len(callback.window_samples)
                if signal_power < POWER_THRESH:
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Closest note: ...")
                    return

                hann_samples = callback.window_samples * HANN_WINDOW
                magnitude_spec = abs(scipy.fftpack.fft(
                    hann_samples)[:len(hann_samples)//2])

                for i in range(int(62/DELTA_FREQ)):
                    magnitude_spec[i] = 0

                for j in range(len(OCTAVE_BANDS)-1):
                    ind_start = int(OCTAVE_BANDS[j]/DELTA_FREQ)
                    ind_end = int(OCTAVE_BANDS[j+1]/DELTA_FREQ)
                    ind_end = ind_end if len(
                        magnitude_spec) > ind_end else len(magnitude_spec)
                    avg_energy_per_freq = (np.linalg.norm(
                        magnitude_spec[ind_start:ind_end], ord=2)**2) / (ind_end-ind_start)
                    avg_energy_per_freq = avg_energy_per_freq**0.5
                    for i in range(ind_start, ind_end):
                        magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH * \
                            avg_energy_per_freq else 0

                mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS), np.arange(0, len(magnitude_spec)),
                                        magnitude_spec)
                mag_spec_ipol = mag_spec_ipol / \
                    np.linalg.norm(mag_spec_ipol, ord=2)  

                hps_spec = copy.deepcopy(mag_spec_ipol)

                for i in range(NUM_HPS):
                    tmp_hps_spec = np.multiply(
                        hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
                    if not any(tmp_hps_spec):
                        break
                    hps_spec = tmp_hps_spec

                max_ind = np.argmax(hps_spec)
                max_freq = max_ind * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

                closest_note, closest_pitch = find_closest_note(max_freq)
                max_freq = round(max_freq, 1)
                closest_pitch = round(closest_pitch, 1)

                callback.noteBuffer.insert(0, closest_note)
                callback.noteBuffer.pop()

                os.system('cls' if os.name == 'nt' else 'clear')
                if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
                    o = open("randomfile.txt", "w")
                    o.write(f"Note: {closest_note} {max_freq}/{closest_pitch}")
                    o.close
                    print(f"Note: {closest_note} {max_freq}/{closest_pitch}")
                else:
                    print(f"Note: ...")

            else:
                o = open("randomfile.txt", "w")
                o.write(f"Note: ... ")
                o.close


        try:
            print("Starting HPS guitar tuner...")
            with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
                while True:
                    time.sleep(0.5)
        except Exception as exc:
            print(str(exc))

class GUI(Thread):
    def run(self):
        root = Tk()
        root.geometry('550x400')
        root.title('Translator')
        root.iconbitmap('guitar.ico')
        root.resizable(False, False)
        root.configure(bg='cyan')
        
        def change():
            o = open("randomfile.txt")
            word = o.readlines()
            label3.configure(text = word)
            o.close()
        
        label3= Label(root, text="", font=('Comic Sans MS', 15, 'italic bold'), bg='cyan')
        label3.place(x=160, y=250)

        btn1 = Button(root, text="Get Note", bd=10, bg='White', activebackground='red', width=10,
              font=('Comic Sans MS', 15, 'italic bold'), compound=RIGHT, command = change)
        btn1.place(x=200, y=170)

        root.mainloop()


t1 = Tuner()
t2 = GUI()

t1.start()
t2.start()
