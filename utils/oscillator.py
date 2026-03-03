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
        type: str = 'sine',
    ):
        self.frequency = frequency
        self.amplitude = amplitude
        self.type = type
        self._phase: float = 0.0
        self._stream = None

    def set_frequency(self, f) -> None:
        '''Acepta float u Oscillator.'''
        self.frequency = f

    def set_amplitude(self, a) -> None:
        '''Acepta float u Oscillator.'''
        self.amplitude = a

    def render(self, frames: int) -> np.ndarray:
        '''
        Genera `frames` muestras sin stream de audio.
        Avanza la fase interna para mantener continuidad entre llamadas.
        Retorna array float32 en [-1, 1] (sin aplicar amplitud propia).
        '''
        freq = self._resolve(self.frequency, frames)
        amp = self._resolve(self.amplitude, frames)

        # Fase acumulada sample a sample para soportar freq variable (FM)
        if np.isscalar(freq):
            phase_inc = 2 * np.pi * freq / SAMPLE_RATE
            phase_vec = self._phase + phase_inc * np.arange(frames)
        else:
            phase_inc = 2 * np.pi * freq / SAMPLE_RATE
            phase_vec = self._phase + np.cumsum(phase_inc)

        if self.type == 'sawtooth':
            wave = amp * (phase_vec % (2 * np.pi) / np.pi - 1.0)
        else:
            t = np.arange(frames) / SAMPLE_RATE
            wave = amp * np.sin(2 * np.pi * freq * t + self._phase)  # Sinosoidal

        # Avanzar fase
        self._phase = phase_vec[-1] % (2 * np.pi)

        return wave.astype(np.float32)

    def play(self) -> None:
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
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            self._phase = 0.0

    def is_playing(self) -> bool:
        return self._stream is not None and self._stream.active

    def _resolve(self, param, frames: int):
        '''
        Si param es Oscillator, llama render() para obtener un array.
        Si es escalar, lo devuelve tal cual.
        '''
        if isinstance(param, Oscillator):
            return param.render(frames)
        return float(param)

    def _callback(self, outdata, frames, time, status) -> None:
        wave = self.render(frames)
        outdata[:, 0] = wave
