import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import scipy.fft

# === 設定 ===
# 入力ファイルのパス（例：ノイズ.m4a）
file_path = "noise.m4a"

# === 音声読み込み ===
audio = AudioSegment.from_file(file_path)
samples = np.array(audio.get_array_of_samples())
sample_rate = audio.frame_rate

# ステレオ→モノラルに変換（平均化）
if audio.channels == 2:
    samples = samples.reshape((-1, 2))
    samples = samples.mean(axis=1)

# 分析用に最初の2秒を抽出（長すぎるとFFT分解能が落ちる）
segment = samples[:sample_rate * 2]

# === FFT（高速フーリエ変換） ===
frequencies = scipy.fft.fftfreq(len(segment), 1 / sample_rate)
fft_magnitude = np.abs(scipy.fft.fft(segment))

# 正の周波数成分だけ取り出す
positive_freqs = frequencies[:len(frequencies)//2]
positive_magnitude = fft_magnitude[:len(frequencies)//2]

# === グラフ表示 ===
plt.figure(figsize=(10, 4))
plt.plot(positive_freqs, positive_magnitude)
plt.title("Frequency Spectrum of Noise")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 1000)  # 主に50〜100Hz帯域を観察
plt.grid(True)
plt.tight_layout()
#plt.show()
plt.savefig("spectrum.png")
print("スペクトル画像を spectrum.png として保存しました。")


