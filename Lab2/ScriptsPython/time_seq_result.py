import pandas as pd
import matplotlib.pyplot as plt

# Чтение данных из CSV файла
df = pd.read_csv("/workspaces/UnicLab/Lab2/OtherFiles/timing_results.csv")

# Первый случай — x фиксирован (N_x = 512)
case1 = df[df["N_x"] == 512]
direct1 = case1[case1["method"] == "direct"]
fft1 = case1[case1["method"] == "fft"]

plt.figure(figsize=(8, 5))
plt.loglog(direct1["N_y"], direct1["time_ms"], 'o-', label="Прямая свертка", base=2)
plt.loglog(fft1["N_y"], fft1["time_ms"], 's-', label="БПФ свертка", base=2)
plt.xlabel("Длина второй последовательности N_y")
plt.ylabel("Время, мс")
plt.title("Случай 1: N_x=512, N_y варьируется (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/workspaces/UnicLab/Lab2/OtherFiles/case1_size.png")
plt.show()

# Второй случай — N_x = N_y
case2 = df[df["method"].str.contains("_same")]
direct2 = case2[case2["method"] == "direct_same"]
fft2 = case2[case2["method"] == "fft_same"]

plt.figure(figsize=(8, 5))
plt.loglog(direct2["N_x"], direct2["time_ms"], 'o-', label="Прямая свертка", base=2)
plt.loglog(fft2["N_x"], fft2["time_ms"], 's-', label="БПФ свертка", base=2)
plt.xlabel("Длина N (N_x = N_y)")
plt.ylabel("Время, мс")
plt.title("Случай 2: N_x = N_y = N (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig("/workspaces/UnicLab/Lab2/OtherFiles/case2_size.png")
plt.show()