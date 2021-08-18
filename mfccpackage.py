# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 21:23:41 2021

@author: asus
"""

from python_speech_features import mfcc
from python_speech_features import delta
from python_speech_features import logfbank
#import scipy.io.wavfile as wav
#import wave
import soundfile
import numpy as np

#(rate,sig) = wav.read("SA1.WAV")
def read_wave_data(filename):
    '''wav=wave.open(filename,'rb')
    num_frame = wav.getnframes() # 获取帧数
    num_channel=wav.getnchannels() # 获取声道数
    framerate=wav.getframerate() # 获取帧速率
    num_sample_width=wav.getsampwidth() # 获取实例的比特宽度，即每一帧的字节数
    str_data = wav.readframes(num_frame) # 读取全部的帧
    wav.close() # 关闭流
    wave_data = np.fromstring(str_data, dtype = np.short) # 将声音文件数据转换为数组矩阵形式
    wave_data.shape = -1, num_channel # 按照声道数将数组整形，单声道时候是一列数组，双声道时候是两列的矩阵
    wave_data = wave_data.T # 将矩阵转置
    wave_data = wave_data'''
    wave_data,framerate=soundfile.read(filename)
    return wave_data, framerate

sig,rate=read_wave_data(r'./SA1.WAV')
mfcc_feat = mfcc(sig,rate)
d_mfcc_feat = delta(mfcc_feat, 2)
fbank_feat = logfbank(sig,rate)

print(fbank_feat[1:3,:])