import numpy as np

def generate_convolution_test_signals(x_filename="x_signal.bin", y_filename="y_signal.bin"):
    M = 1024
    L = 1024

    t_x = np.linspace(0, 2, M)
    t_y = np.linspace(0, 2, L)

    x_real = (0.7 * np.sin(2 * np.pi * 3 * t_x) +
              0.3 * np.sin(2 * np.pi * 8 * t_x) +
              0.1 * np.sin(2 * np.pi * 15 * t_x))
    x_imag = (0.5 * np.sin(2 * np.pi * 5 * t_x) +
              0.2 * np.cos(2 * np.pi * 12 * t_x))

    y_real = (0.6 * np.sin(2 * np.pi * 4 * t_y) +
              0.4 * np.sin(2 * np.pi * 10 * t_y))
    y_imag = (0.3 * np.cos(2 * np.pi * 6 * t_y) +
              0.7 * np.cos(2 * np.pi * 7 * t_y))

    x_real += 0.05 * np.random.normal(0, 1, M)
    x_imag += 0.05 * np.random.normal(0, 1, M)
    y_real += 0.05 * np.random.normal(0, 1, L)
    y_imag += 0.05 * np.random.normal(0, 1, L)

    x_signal = x_real + 1j * x_imag
    y_signal = y_real + 1j * y_imag

    save_complex_binary(x_filename, x_signal)
    save_complex_binary(y_filename, y_signal)

    return x_signal, y_signal


def save_complex_binary(filename, complex_signal):
    data_to_write = np.empty(2 * len(complex_signal), dtype=np.float64)
    data_to_write[0::2] = complex_signal.real
    data_to_write[1::2] = complex_signal.imag

    with open(filename, 'wb') as f:
        data_to_write.tofile(f)


x, y = generate_convolution_test_signals("/workspaces/UnicLab/Lab2/OtherFiles/x_signal.bin", "/workspaces/UnicLab/Lab2/OtherFiles/y_signal.bin")

