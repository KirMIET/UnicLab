import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def read_complex_binary(filename):
    data = np.fromfile(filename, dtype=np.float64)
    return data[::2] + 1j * data[1::2]

def calculate_mse(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2) ** 2)

def calculate_norm_difference(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def print_summary(mse_direct_numpy, mse_fft_numpy, mse_direct_fft,norm_direct_numpy, norm_fft_numpy, norm_direct_fft):
    print("Таблица результатов")

    results = [
        ("Прямая свертка vs numpy.convolve", mse_direct_numpy, norm_direct_numpy),
        ("FFT свертка vs numpy.convolve", mse_fft_numpy, norm_fft_numpy),
        ("Прямая свертка vs FFT свертка", mse_direct_fft, norm_direct_fft)
    ]

    print("=" * 80)
    print("\t"*5, "MSE     \t Норма разности")
    for description, mse, norm in results:
        print(f"{description:<35} {mse:<20.2e} {norm:<20.2e}")

    print("=" * 80)


print("Сравнение методов свертки")

# Чтение входных сигналов
x_signal = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/x_signal.bin')
y_signal = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/y_signal.bin')

# Чтение результатов свертки
direct_result = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/direct_convolution_result.bin')
fft_result = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/fft_convolution_result.bin')

print(f"Размер сигнала x: {len(x_signal)}")
print(f"Размер сигнала y: {len(y_signal)}")
print(f"Размер прямой свертки: {len(direct_result)}")
print(f"Размер FFT свертки: {len(fft_result)}")

# Вычисление свертки с помощью numpy
print("\nВычисление свертки с помощью numpy.convolve")
numpy_result = np.convolve(x_signal, y_signal, mode='full')
print(f"Размер numpy свертки: {len(numpy_result)}")

# Сравнение прямой свертки с numpy
print("\n1. Сравнение прямой свертки с numpy.convolve:")

mse_direct_numpy = calculate_mse(direct_result, numpy_result)
norm_direct_numpy = calculate_norm_difference(direct_result, numpy_result)
print(f"  Среднеквадратичная ошибка (MSE): {mse_direct_numpy:.10e}")
print(f"  Норма разности: {norm_direct_numpy:.10e}")

# Сравнение FFT свертки с numpy
print("\n2. Сравнение БПФ свертки с numpy.convolve:")

mse_fft_numpy = calculate_mse(fft_result, numpy_result)
norm_fft_numpy = calculate_norm_difference(fft_result, numpy_result)
print(f"  Среднеквадратичная ошибка (MSE): {mse_fft_numpy:.10e}")
print(f"  Норма разности: {norm_fft_numpy:.10e}")


# Сравнение прямой и FFT свертки
print("\n3. Сравнение прямой и БПФ сверток:")
min_size_direct_fft = min(len(direct_result), len(fft_result))
mse_direct_fft = calculate_mse(direct_result[:min_size_direct_fft], fft_result[:min_size_direct_fft])
norm_direct_fft = calculate_norm_difference(direct_result[:min_size_direct_fft], fft_result[:min_size_direct_fft])

print(f"  Среднеквадратичная ошибка (MSE): {mse_direct_fft:.10e}")
print(f"  Норма разности: {norm_direct_fft:.10e}")

# Вывод итоговой таблицы
print_summary(mse_direct_numpy, mse_fft_numpy, mse_direct_fft,
              norm_direct_numpy, norm_fft_numpy, norm_direct_fft)