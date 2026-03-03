#!/usr/bin/env python3

import numpy as np
import sounddevice as sd

sr = 44100  # sample rate
f = 440.0  # frecuencia (Hz)
dur = 2.0  # segundos

t = np.linspace(0, dur, int(sr * dur), endpoint=False)
# seno
x = 0.2 * np.sin(2 * np.pi * f * t)  # 0.2 = volumen (evita clipping)

# cuadrada
x = 0.2 * np.sign(np.sin(2 * np.pi * f * t))

# diente de sierra
phase = (f * t) % 1.0
x = 0.2 * (2 * phase - 1)

sd.play(x, sr)
sd.wait()
