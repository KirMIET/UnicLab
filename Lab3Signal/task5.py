import numpy as np
import scipy.optimize
import scipy.signal
import matplotlib.pyplot as plt
import os
from PIL import Image

M_final = 21
N_final = 2 * M_final
w_p = [0, 0.60 * np.pi]
w_s = [0.70 * np.pi, np.pi]


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

    return 0, h


w_j = np.pi * (np.arange(0, M_final + 1) + 0.5) / (M_final + 1)
mask_trans = ~((w_j >= w_p[0]) & (w_j <= w_p[1]) | ((w_j >= w_s[0]) & (w_j < w_s[1])))
x0 = np.full(np.sum(mask_trans), 0.5)

x_opt = scipy.optimize.fmin(lambda t: syntez(t, w_p, w_s, M_final)[0], x0, disp=False)
_, h_coeffs = syntez(x_opt, w_p, w_s, M_final)

b_unique = h_coeffs[::-1]
b = np.concatenate((b_unique, b_unique[-2::-1]))

I_original = np.array(Image.open('/workspaces/UnicLab/Lab3Signal/var10.png').convert('L')) / 255.0

# размеры изображения
R, C = I_original.shape

print(f"Размер исходного изображения: {I_original.shape}")
print(f"Порядок фильтра N={N_final}, задержка M={M_final}.")

# Фильтрация строк
I_row_filtered = np.zeros_like(I_original)
for i in range(R):
    I_row_filtered[i, :] = scipy.signal.lfilter(b, 1.0, I_original[i, :])

# Фильтрация столбцов
I_final = np.zeros_like(I_row_filtered)
for j in range(C):
    I_final[:, j] = scipy.signal.lfilter(b, 1.0, I_row_filtered[:, j])

RESULTS_DIR = 'Results/task5'
os.makedirs(RESULTS_DIR, exist_ok=True)

plt.figure(figsize=(12, 5))

# Исходное изображение
plt.subplot(1, 2, 1)
plt.imshow(I_original, cmap='gray')
plt.title('Исходное изображение')
plt.axis('off')

# Отфильтрованное изображение
plt.subplot(1, 2, 2)
plt.imshow(I_final[M_final:, M_final:], cmap='gray')
plt.title(f'Отфильтрованное изображение ($N=42, M=21$)')
plt.axis('off')

plt.tight_layout()
save_path = os.path.join(RESULTS_DIR, 'task5.png')
plt.savefig(save_path)
plt.show()