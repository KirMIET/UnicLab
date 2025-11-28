import numpy as np
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import os

M_final = 21
N_final = 2 * M_final
w_p = [0, 0.60 * np.pi]
w_s = [0.70 * np.pi, np.pi]
omega_x = 0.30 * np.pi

def syntez(x_j, w_p, w_s, M):
    w_j = np.pi * (np.arange(0, M + 1) + 0.5) / (M + 1)
    flag_p = (w_j >= w_p[0]) & (w_j <= w_p[1])
    flag_s = (w_j >= w_s[0]) & (w_j < w_s[1])
    flag_K_d = np.zeros(w_j.size)
    flag_K_d[flag_p] = 1.0
    flag_K_d[flag_s] = 0.0
    flag_K_d[~(flag_s | flag_p)] = x_j

    h = np.zeros(M + 1)
    for k in range(M + 1):
        h[M - k] = (1 / (M + 1)) * np.sum(flag_K_d * np.cos(w_j * k))

    def calc_A(w_arr):
        sum_val = np.zeros_like(w_arr)
        for k in range(1, M + 1):
            sum_val += 2 * h[M - k] * np.cos(w_arr * k)
        return h[M] + sum_val

    W_s_check = np.arange(w_s[0], w_s[1], 0.001)
    W_p_check = np.arange(w_p[0], w_p[1], 0.001)

    A_s = calc_A(W_s_check)
    A_p = calc_A(W_p_check)

    err_s = np.max(np.abs(A_s - 0))
    err_p = np.max(np.abs(A_p - 1))

    Er = max(err_s, err_p)
    return Er, h

w_j = np.pi * (np.arange(0, M_final + 1) + 0.5) / (M_final + 1)
mask_trans = ~((w_j >= w_p[0]) & (w_j <= w_p[1]) | ((w_j >= w_s[0]) & (w_j < w_s[1])))
x0 = np.full(np.sum(mask_trans), 0.5)

x_opt = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M_final)[0], x0, disp=False)
_, h_coeffs = syntez(x_opt, w_p, w_s, M_final)

b_unique = h_coeffs[::-1]
b = np.concatenate((b_unique, b_unique[-2::-1]))

T = 150
n = np.arange(T)
x_n = np.sin(omega_x * n)

# Фильтрация
y_n = scipy.signal.lfilter(b, 1.0, x_n)

RESULTS_DIR = 'Results/task4'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.figure(figsize=(12, 6))

plt.plot(n, x_n, 'b', label=r'Входной сигнал $x(n)=\sin(0.30\pi n)$', alpha=0.7, linewidth=1.5)
plt.plot(n, y_n, 'r', label=f'Выходной сигнал $y(n)$', linewidth=2)

plt.title(f'Определение задержки $alpha$ (КИХ-НЧФ, $N={N_final}, M=11$)')
plt.xlabel('Номер отсчета $n$')
plt.ylabel('Амплитуда')
plt.grid(True, linestyle='--')
plt.legend()

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, 'task4.png')
plt.savefig(save_path)
plt.show()

print(f"\n Результат Задания 4 ")
print(f"1. Частота гармонического сигнала: omega_x = {omega_x/np.pi:.2f}pi")
print(f"2. Задержка числа отсчётов alpha равна половине порядка фильтра M.")
print(f"   alpha = M_min = {M_final} отсчетов.")