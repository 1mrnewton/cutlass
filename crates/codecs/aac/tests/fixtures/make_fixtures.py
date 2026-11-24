#!/usr/bin/env python3
"""Rebuild the AAC decoder fixtures in this folder.

Needs numpy and an `ffmpeg` with the native `aac` encoder. The source is a
synthetic stereo mix: plucked notes with harmonics at different pan
positions, a kick, snare and hi-hat pattern for transients, and a quiet
independent hiss in each channel. FFmpeg encodes it to ADTS and decodes that
again to 32-bit float PCM, which is the reference the tests compare with.

The encoder's output can change between FFmpeg versions, so a rebuild may
change the fixtures; the README beside this file records the version used.
"""
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np


def source(rate, seconds, seed):
    rng = np.random.default_rng(seed)
    n = int(rate * seconds)
    t = np.arange(n) / rate
    left = np.zeros(n)
    right = np.zeros(n)

    def add(signal, start, pan):
        begin = int(start * rate)
        end = min(n, begin + len(signal))
        left[begin:end] += signal[: end - begin] * np.cos(pan * np.pi / 2)
        right[begin:end] += signal[: end - begin] * np.sin(pan * np.pi / 2)

    def pluck(freq, length, level):
        tt = np.arange(int(length * rate)) / rate
        tone = np.zeros(len(tt))
        for harmonic in range(1, 9):
            if freq * harmonic > rate * 0.45:
                break
            decay = np.exp(-tt * (3.0 + 1.5 * harmonic))
            tone += np.sin(2 * np.pi * freq * harmonic * tt) * decay / harmonic
        attack = np.minimum(1.0, tt / 0.002)
        return tone * attack * level

    def kick(level):
        tt = np.arange(int(0.18 * rate)) / rate
        phase = 2 * np.pi * (45 * tt + 60 * (1 - np.exp(-tt * 30)) / 30)
        return np.sin(phase) * np.exp(-tt * 18) * level

    def noise_hit(length, decay, low, level):
        count = int(length * rate)
        tt = np.arange(count) / rate
        burst = rng.standard_normal(count)
        spectrum = np.fft.rfft(burst)
        freqs = np.fft.rfftfreq(count, 1 / rate)
        spectrum[freqs < low] = 0
        return np.fft.irfft(spectrum, count) * np.exp(-tt * decay) * level

    notes = [220.0, 277.18, 329.63, 440.0, 554.37, 659.25, 880.0, 1318.5]
    step = seconds / 8
    for beat in range(8):
        add(pluck(notes[beat], 0.6, 0.22), beat * step, 0.15 + 0.1 * beat)
        add(pluck(notes[(beat * 3 + 2) % 8] / 2, 0.6, 0.18), beat * step, 0.85 - 0.1 * beat)
        if beat % 2 == 0:
            add(kick(0.5), beat * step, 0.5)
        else:
            add(noise_hit(0.12, 35, 400, 0.12), beat * step, 0.45)
        add(noise_hit(0.04, 120, 6000, 0.06), beat * step + step / 2, 0.7)
    left += rng.standard_normal(n) * 0.004
    right += rng.standard_normal(n) * 0.004
    fade = np.minimum(1.0, np.minimum(t, seconds - t) / 0.01)
    pcm = np.stack([left * fade, right * fade], axis=1)
    return np.clip(pcm, -0.95, 0.95)


def build(ffmpeg, name, rate, seconds, seed, options):
    pcm = source(rate, seconds, seed)
    wav = os.path.join(tempfile.gettempdir(), f"{name}.wav")
    with wave.open(wav, "wb") as out:
        out.setnchannels(2)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes((pcm * 32767).round().astype("<i2").tobytes())
    quiet = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    subprocess.run(
        quiet + ["-i", wav, "-c:a", "aac", *options, "-bitexact", "-f", "adts", f"{name}.aac"],
        check=True,
    )
    subprocess.run(
        quiet + ["-c:a", "aac", "-i", f"{name}.aac", "-f", "f32le", f"{name}.f32"],
        check=True,
    )


def main():
    ffmpeg = sys.argv[1] if len(sys.argv) > 1 else "ffmpeg"
    build(ffmpeg, "tools_44100_128k_nopns", 44100, 0.5, 1, ["-b:a", "128k", "-aac_pns", "0"])
    build(ffmpeg, "tools_48000_64k_nopns", 48000, 0.5, 2, ["-b:a", "64k", "-aac_pns", "0"])
    build(ffmpeg, "tools_44100_96k_pns", 44100, 0.5, 3, ["-b:a", "96k"])


if __name__ == "__main__":
    main()
