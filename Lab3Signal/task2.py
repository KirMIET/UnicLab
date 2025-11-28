import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import os


def syntez(x_j, w_p, w_s, M):
    w_j = np.pi * (np.arange(0, M + 1) + 0.5) / (M + 1)

    flag_p = (w_j >= w_p[0]) & (w_j <= w_p[1])  # Полоса пропускания
    flag_s = (w_j >= w_s[0]) & (w_j < w_s[1])  # Полоса подавления

    flag_K_d = np.zeros(w_j.size)
    flag_K_d[flag_p] = 1.0
    flag_K_d[flag_s] = 0.0

    # Заполнение переходной зоны оптимизируемыми параметрами
    mask_trans = ~(flag_s | flag_p)
    flag_K_d[mask_trans] = x_j

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


M = 9
w_p = [0, 0.60 * np.pi]
w_s = [0.70 * np.pi, np.pi]

w_j = np.pi * (np.arange(0, M + 1) + 0.5) / (M + 1)
mask_trans = ~((w_j >= w_p[0]) & (w_j <= w_p[1]) | ((w_j >= w_s[0]) & (w_j < w_s[1])))
num_trans_points = np.sum(mask_trans)

print(f"Количество точек в переходной зоне: {num_trans_points}")
x0 = np.full(num_trans_points, 0.5)

x_opt = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M)[0], x0, disp=True)

print(f"Оптимизированное значение x: {x_opt}")

Er_final, h_final = syntez(x_opt, w_p, w_s, M)

w = np.arange(0, 2 * np.pi, 0.001)

sum_part = sum(2 * h_final[M - k] * np.cos(w * k) for k in range(1, M + 1))
A = h_final[M] + sum_part

H = A * np.exp(-1j * w * M)
H_abs = np.abs(H)

# Идеальная АЧХ
Ideal_H_mod = (w <= w_p[1]).astype(float)

# Идеальная ФЧХ
Ideal_Phase = np.zeros_like(w)
Ideal_Phase[w <= w_p[1]] = -w[w <= w_p[1]] * M

RESULTS_DIR = 'Results/task2'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, H_abs, label='Синтезированная АЧХ (Оптим.)', linewidth=2)
plt.plot(w / np.pi, Ideal_H_mod, 'r--', label='Идеальная АЧХ', alpha=0.7)

plt.title(f'АЧХ фильтра после оптимизации (M={M}, Error={Er_final:.4f})')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(H), label='Синтезированная ФЧХ')
plt.plot(w / np.pi, Ideal_Phase, 'g--', label=r'Идеальная ФЧХ $(-\omega M)$', alpha=0.7)
plt.title('ФЧХ фильтра')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel('Фаза (рад)')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.figure(figsize=(10, 4))

min_value = 1e-7

Abs_H = H_abs
Abs_H[Abs_H < min_value] = min_value
H_dB = 20 * np.log10(Abs_H)

Abs_Ideal_H = Ideal_H_mod
Abs_Ideal_H[Abs_Ideal_H < min_value] = min_value
Ideal_H_dB = 20 * np.log10(Abs_Ideal_H)

plt.plot(w / np.pi, H_dB, label='Синтезированная АЧХ (дБ)', linewidth=2)
plt.plot(w / np.pi, Ideal_H_dB, 'r--', label='Идеальная АЧХ (дБ)', alpha=0.7)
plt.title(f'АЧХ фильтра (M={M}) в децибелах')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel(r'$20 \log_{10}|H(e^{j\omega})|$ (дБ)')
plt.grid(True)
plt.legend()

plt.tight_layout()

save_path_combined = os.path.join(RESULTS_DIR, 'task2.png')
plt.figure(1)
plt.savefig(save_path_combined)

save_path_db = os.path.join(RESULTS_DIR, 'task2_db.png')
plt.figure(2)
plt.savefig(save_path_db)

plt.show()