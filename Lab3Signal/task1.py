import numpy as np
import matplotlib.pyplot as plt
import os

w_p = [0, 0.6 * np.pi]
M = 9

w_j = np.pi * (np.arange(0, M+1) + 0.5) / (M + 1)
K_d = (w_p[0] <= w_j) & (w_j < w_p[1])
K_d = K_d.astype(float)

h = np.zeros(M + 1)
for k in range(M + 1):
    h[M - k] = (1 / (M + 1)) * np.sum(K_d * np.cos(w_j * k))

w = np.arange(0, 2 * np.pi, 0.001)

sum_part = sum(2 * h[M - k] * np.cos(w * k) for k in range(1, M + 1))
A = h[M] + sum_part

H = A * np.exp(-1j * w * M)

Ideal_H_mod = (w <= w_p[1]).astype(float)

Ideal_Phase = np.zeros_like(w)
Ideal_Phase[w <= w_p[1]] = -w[w <= w_p[1]] * M

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(w / np.pi, np.abs(H), label='Синтезированная АЧХ')
plt.plot(w / np.pi, Ideal_H_mod, label='Идеальная АЧХ', linestyle='--')
plt.title(f'АЧХ фильтра (M={M})')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel(r'$|H(e^{j\omega})|$')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(w / np.pi, np.angle(H), label='Синтезированная ФЧХ')
plt.plot(w / np.pi, Ideal_Phase, label=r'Идеальная ФЧХ $(-\omega M)$', linestyle='--')
plt.title('ФЧХ фильтра')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel('Фаза (рад)')
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.figure(figsize=(10, 4))


min_value = 1e-7

Abs_H = np.abs(H)
Abs_H[Abs_H < min_value] = min_value
H_dB = 20 * np.log10(Abs_H)

Abs_Ideal_H = Ideal_H_mod
Abs_Ideal_H[Abs_Ideal_H < min_value] = min_value
Ideal_H_dB = 20 * np.log10(Abs_Ideal_H)

plt.plot(w / np.pi, H_dB, label='Синтезированная АЧХ (дБ)')
plt.plot(w / np.pi, Ideal_H_dB, label='Идеальная АЧХ (дБ)', linestyle='--')
plt.title(f'АЧХ фильтра (M={M}) в децибелах')
plt.xlabel(r'Нормированная частота $\omega / \pi$')
plt.ylabel(r'$20 \log_{10}|H(e^{j\omega})|$ (дБ)')
plt.grid(True)
plt.legend()

plt.tight_layout()

output_dir = 'Results/task1'
os.makedirs(output_dir, exist_ok=True)

save_path_combined = os.path.join(output_dir, 'task1_plot.png')
plt.figure(1)
plt.savefig(save_path_combined)

save_path_db = os.path.join(output_dir, 'task1_plot_db.png')
plt.figure(2)
plt.savefig(save_path_db)

plt.show()