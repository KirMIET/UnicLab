import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import os

M_start = 9
w_p = [0, 0.60 * np.pi]
w_s = [0.70 * np.pi, np.pi]
delta_p = 0.0150
delta_s = 0.0350

def syntez(x_j, w_p, w_s, M):
    w_j = np.pi * (np.arange(0, M + 1) + 0.5) / (M + 1)
    flag_p = (w_j >= w_p[0]) & (w_j <= w_p[1])
    flag_s = (w_j >= w_s[0]) & (w_j < w_s[1])
    flag_K_d = np.zeros(w_j.size)
    flag_K_d[flag_p] = 1.0
    flag_K_d[flag_s] = 0.0

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


M = M_start
M_min = None
print("Поиск минимального порядка фильтра")

try:
    w_j_start = np.pi * (np.arange(0, M_start + 1) + 0.5) / (M_start + 1)
    mask_trans_start = ~((w_j_start >= w_p[0]) & (w_j_start <= w_p[1]) | ((w_j_start >= w_s[0]) & (w_j_start < w_s[1])))
    x0_start = np.full(np.sum(mask_trans_start), 0.5)

    x_opt_start = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M_start)[0], x0_start, disp=False)
    Er_start, _ = syntez(x_opt_start, w_p, w_s, M_start)
    print(f"Начальный M={M_start}, Ошибка Er={Er_start:.5f}. Требование: Er <= {delta_p}")

    if Er_start > delta_p:
        print("M=9 не удовлетворяет требованиям. Ищем, увеличивая M.")
        M_min_candidate = M_start + 1

        while True:
            M = M_min_candidate
            w_j = np.pi * (np.arange(0, M + 1) + 0.5) / (M + 1)
            mask_trans = ~((w_j >= w_p[0]) & (w_j <= w_p[1]) | ((w_j >= w_s[0]) & (w_j < w_s[1])))
            x0 = np.full(np.sum(mask_trans), 0.5)

            x_opt = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M)[0], x0, disp=False)
            Er, _ = syntez(x_opt, w_p, w_s, M)

            print(f"Проверка M={M}: Er={Er:.5f}")

            if Er <= delta_p:
                M_min = M
                print(f"Минимальный M={M_min}")
                break

            M_min_candidate += 1
            if M_min_candidate > 50:
                print("Слишком большой M. Остановлено.")
                break

    else:
        print("M=9 удовлетворяет требованиям. Ищем, уменьшая M.")
        M_min_candidate = M_start

        while M_min_candidate >= 0:
            M = M_min_candidate

            w_j = np.pi * (np.arange(0, M + 1) + 0.5) / (M + 1)
            mask_trans = ~((w_j >= w_p[0]) & (w_j <= w_p[1]) | ((w_j >= w_s[0]) & (w_j < w_s[1])))

            if np.sum(mask_trans) > 0:
                x0 = np.full(np.sum(mask_trans), 0.5)
                x_opt = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M)[0], x0, disp=False)
                Er, _ = syntez(x_opt, w_p, w_s, M)
            else:
                Er = 1  # Гарантируем, что не удовлетворяет

            print(f"Проверка M={M}: Er={Er:.5f}")

            if Er <= delta_p:
                M_min = M
                M_min_candidate -= 1
            else:
                M_min = M + 1
                print(f"Минимальный M={M_min}")
                break

except Exception as e:
    M_min = M_start

M_final = M_min
N_final = 2 * M_final

w_j = np.pi * (np.arange(0, M_final + 1) + 0.5) / (M_final + 1)
flag_p = (w_j >= w_p[0]) & (w_j <= w_p[1])
flag_s = (w_j >= w_s[0]) & (w_j < w_s[1])
mask_trans = ~ (flag_p | flag_s)

x0 = np.full(np.sum(mask_trans), 0.5)

x_opt = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M_final)[0], x0, disp=True)
Er_final, h_final = syntez(x_opt, w_p, w_s, M_final)

w = np.arange(0, np.pi, 0.0001)

sum_part = sum(2 * h_final[M_final - k] * np.cos(w * k) for k in range(1, M_final + 1))
A = h_final[M_final] + sum_part
H = A * np.exp(-1j * w * M_final)
H_abs = np.abs(H)

Ideal_H_mod = (w <= w_p[1]).astype(float)

Ideal_Phase = np.zeros_like(w)
Ideal_Phase[w <= w_p[1]] = -w[w <= w_p[1]] * M_final

RESULTS_DIR = 'Results/task3'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, H_abs, label='Синтезированная АЧХ')
plt.plot(w / np.pi, Ideal_H_mod, 'r--', label='Идеальная АЧХ', alpha=0.7)

plt.hlines([1 + delta_p, 1 - delta_p], w_p[0] / np.pi, w_p[1] / np.pi, color='g', linestyle=':', label=r'$\pm\delta_p$')
plt.hlines([delta_s], w_s[0] / np.pi, w_s[1] / np.pi, color='orange', linestyle=':', label=r'$\delta_s$')
plt.title(f'АЧХ фильтра (N={N_final}, M={M_final}). Er={Er_final:.5f}')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.ylim(-0.05, 1.1)
plt.grid(True)
plt.legend(loc='lower left')

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(H), label='Синтезированная ФЧХ')
plt.plot(w / np.pi, Ideal_Phase, 'g--', label=r'Идеальная ФЧХ $(-\omega M)$', alpha=0.7)
plt.title('ФЧХ фильтра')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel('Фаза (рад)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.figure(1)
plt.savefig(os.path.join(RESULTS_DIR, 'task3.png'))

plt.figure(figsize=(10, 4))

min_value = 1e-7

Abs_H_dB = H_abs
Abs_H_dB[Abs_H_dB < min_value] = min_value
H_dB = 20 * np.log10(Abs_H_dB)

Abs_Ideal_H_dB = Ideal_H_mod
Abs_Ideal_H_dB[Abs_Ideal_H_dB < min_value] = min_value
Ideal_H_dB = 20 * np.log10(Abs_Ideal_H_dB)

plt.plot(w / np.pi, H_dB, label='Синтезированная АЧХ (дБ)', linewidth=2)
plt.plot(w / np.pi, Ideal_H_dB, 'r--', label='Идеальная АЧХ (дБ)', alpha=0.7)

plt.title(f'АЧХ фильтра (N={N_final}, M={M_final}) в децибелах')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel(r'$20 \log_{10}|H(e^{j\omega})|$ (дБ)')
plt.grid(True)
plt.legend(loc='lower left')

plt.tight_layout()
plt.figure(2)
plt.savefig(os.path.join(RESULTS_DIR, 'task3_db.png'))

plt.show()