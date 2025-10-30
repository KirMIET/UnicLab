import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def read_complex_bin(filename):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.float64)

    # Преобразуем в комплексные числа
    complex_data = data[0::2] + 1j * data[1::2]

    return complex_data


def plot_signal(signal, filename):
    N = len(signal)
    t = np.arange(N)

    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Анализ сигнала из файла: {filename}', fontsize=16)

    # 1. Действительная и мнимая части
    axes[0, 0].plot(t, signal.real, 'b-', linewidth=1, label='Действительная часть')
    axes[0, 0].plot(t, signal.imag, 'r-', linewidth=1, label='Мнимая часть')
    axes[0, 0].set_xlabel('Отсчеты')
    axes[0, 0].set_ylabel('Амплитуда')
    axes[0, 0].set_title('Действительная и мнимая части')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Модуль сигнала
    axes[0, 1].plot(t, np.abs(signal), 'g-', linewidth=1)
    axes[0, 1].set_xlabel('Отсчеты')
    axes[0, 1].set_ylabel('Амплитуда')
    axes[0, 1].set_title('Модуль сигнала')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Фаза сигнала
    axes[1, 0].plot(t, np.angle(signal), 'm-', linewidth=1)
    axes[1, 0].set_xlabel('Отсчеты')
    axes[1, 0].set_ylabel('Радианы')
    axes[1, 0].set_title('Фаза сигнала')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Спектр амплитуд (БПФ)
    spectrum = np.fft.fft(signal)
    freq = np.fft.fftfreq(N)
    axes[1, 1].plot(freq[:N // 2], np.abs(spectrum[:N // 2]), 'c-', linewidth=1)
    axes[1, 1].set_xlabel('Нормированная частота')
    axes[1, 1].set_ylabel('Амплитуда')
    axes[1, 1].set_title('Амплитудный спектр (БПФ)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'/workspaces/UnicLab/Lab2/OtherFiles/signal_analysis_{os.path.basename(filename)}.png')
    plt.show()

    return fig

if len(sys.argv) > 1:
    filename = sys.argv[1]
else:
    filename = "/workspaces/UnicLab/Lab2/OtherFiles/input.bin"

signal = read_complex_bin(filename)

# Построение графиков
plot_signal(signal, filename)
