import numpy as np
import matplotlib.pyplot as plt

def read_complex_binary(filename):
    data = np.fromfile(filename, dtype=np.float64)
    return data[::2] + 1j * data[1::2]

def calculate_mse(arr1, arr2):
    return np.mean(np.abs(arr1 - arr2) ** 2)

def print_summary(mse_idft, mse_ifft, mse_dft_fft, mse_numpy_fft):
    print("Таблица результатов")

    results = [
        ("X = ОДПФ(ДПФ(X))", mse_idft),
        ("X = ОБПФ(БПФ(X))", mse_ifft),
        ("ДПФ(X) = БПФ(X)", mse_dft_fft),
        ("Наш БПФ = numpy.fft.fft", mse_numpy_fft)
    ]

    for description, mse in results:
        print(f"{description:30} | MSE: {mse:.2e} ")

X_original = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/input.bin')
DFT_result = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/dft_result.bin')
IDFT_result = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/idft_result.bin')
FFT_result = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/fft_result.bin')
IFFT_result = read_complex_binary('/workspaces/UnicLab/Lab2/OtherFiles/ifft_result.bin')


print(f"Размер исходного сигнала: {len(X_original)}")
print(f"Размер ДПФ: {len(DFT_result)}")
print(f"Размер БПФ: {len(FFT_result)}")

# 1. Проверка равенств X = ОДПФ(ДПФ(X)) и X = ОБПФ(БПФ(X))
print("1. Проверка восстановления сигнала")

# Проверка для ДПФ
mse_idft = calculate_mse(X_original, IDFT_result)

print(f"Сравнение X и ОДПФ(ДПФ(X)):")
print(f"  Среднеквадратичная ошибка (MSE): {mse_idft:.10e}")

# Проверка для БПФ
mse_ifft = calculate_mse(X_original, IFFT_result)

print(f"Сравнение X и ОБПФ(БПФ(X)):")
print(f"  Среднеквадратичная ошибка (MSE): {mse_ifft:.10e}")

# 2. Сравнение результатов ДПФ(X) и БПФ(X)
print("2. Сравнение ДПФ и БПФ")

mse_dft_fft = calculate_mse(DFT_result, FFT_result)

print(f"Сравнение ДПФ(X) и БПФ(X):")
print(f"  Среднеквадратичная ошибка (MSE): {mse_dft_fft:.10e}")

# 3. Сравнение с встроенной функцией FFT в Python
print("3. Сравнение с numpy.fft.fft")

numpy_fft = np.fft.fft(X_original)

mse_numpy_fft = calculate_mse(numpy_fft, FFT_result)

print(f"Сравнение нашего БПФ и numpy.fft.fft:")
print(f"  Среднеквадратичная ошибка (MSE): {mse_numpy_fft:.10e}")

# Вывод итоговой таблицы
print_summary(mse_idft, mse_ifft, mse_dft_fft, mse_numpy_fft)