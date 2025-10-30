import numpy as np
import matplotlib.pyplot as plt

def generate_input_file(filename, N):
    # Генерация тестового сигнала
    t = np.linspace(0, 1, N)

    # Создаем комплексный сигнал
    # Действительная часть: сумма двух синусоид
    real_part = (0.7 * np.sin(2 * np.pi * 5 * t) +
                 0.3 * np.sin(2 * np.pi * 15 * t) +
                 0.1 * np.sin(2 * np.pi * 30 * t))

    # Мнимая часть: сумма синусоиды и косинусоиды
    imag_part = (0.5 * np.sin(2 * np.pi * 8 * t) +
                 0.4 * np.cos(2 * np.pi * 20 * t))

    # Комплексный сигнал
    signal = real_part + 1j * imag_part

    # Добавляем немного шума
    noise = 0.05 * (np.random.normal(0, 1, N) +
                    1j * np.random.normal(0, 1, N))
    signal += noise

    # Сохраняем в бинарный файл, действительные и мнимые части попеременно
    with open(filename, 'wb') as f:
        data_to_write = np.empty(2 * N, dtype=np.float64)
        data_to_write[0::2] = signal.real
        data_to_write[1::2] = signal.imag
        data_to_write.tofile(f)

    print(f"Файл {filename} успешно создан")
    print(f"Записано {N} комплексных чисел")

    # Визуализация сигнала
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, signal.real, 'b-', linewidth=1, label='Действительная часть')
    plt.plot(t, signal.imag, 'r-', linewidth=1, label='Мнимая часть')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Сгенерированный комплексный сигнал')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, np.abs(signal), 'g-', linewidth=1, label='Модуль')
    plt.plot(t, np.angle(signal), 'm-', linewidth=1, label='Фаза')
    plt.xlabel('Время')
    plt.ylabel('Амплитуда')
    plt.title('Модуль и фаза сигнала')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('/workspaces/UnicLab/Lab2/OtherFiles/generated_signal.png', dpi=150)
    plt.show()

    return signal

N = 2**10
signal = generate_input_file("/workspaces/UnicLab/Lab2/OtherFiles/input.bin", N)


