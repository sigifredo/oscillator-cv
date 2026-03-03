import numpy as np
import sounddevice as sd

BLOCK_SIZE = 2048
SAMPLE_RATE = 44100


def map_range(value: float, in_min: float, in_max: float, out_min: float, out_max: float, clamp=False):
    result = out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)
    return max(out_min, min(out_max, result)) if clamp else result


class Oscillator:
    def __init__(
        self,
        frequency: float = 440.0,
        amplitude: float = 0.4,
    ):
        self.frequency = frequency
        self.amplitude = amplitude
        self._phase: float = 0.0
        self._stream = None

    def set_frequency(self, f: float) -> None:
        '''Actualiza la frecuencia en Hz. Seguro llamarlo con el stream activo.'''
        self.frequency = float(f)

    def set_amplitude(self, a: float) -> None:
        '''Actualiza la amplitud (0.0 a 1.0).'''
        self.amplitude = float(np.clip(a, 0.0, 1.0))

    def play(self) -> None:
        '''Inicia el stream de audio en segundo plano. No bloqueante.'''

        if self._stream is not None and self._stream.active:
            return
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype='float32',
            callback=self._callback,
        )

        self._stream.start()

    def stop(self) -> None:
        '''Detiene y cierra el stream.'''

        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._phase = 0.0

    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    def _callback(self, outdata, frames, time, status) -> None:
        freq = self.frequency
        amp = self.amplitude
        t = np.arange(frames) / SAMPLE_RATE

        # wave = amp * np.sin(2 * np.pi * freq * t + self._phase) # Sinosoidal
        phase_vec = (self._phase + 2 * np.pi * freq * t) % (2 * np.pi)

        # Diente de sierra: mapeo lineal de [0, 2π] → [-1, 1]
        wave = amp * (phase_vec / np.pi - 1.0)

        # Acumular fase para evitar discontinuidades entre bloques
        self._phase = (self._phase + 2 * np.pi * freq * frames / SAMPLE_RATE) % (2 * np.pi)

        outdata[:, 0] = wave.astype(np.float32)
