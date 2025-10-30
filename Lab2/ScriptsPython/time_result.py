import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Чтение данных из CSV файла
df = pd.read_csv('/workspaces/UnicLab/Lab2/OtherFiles/execution_times.csv')

# Создание графика
plt.figure(figsize=(10, 6))

# Построение линий для DFT и FFT
plt.plot(df['N'], df['DFT_Time_us'], 'ro-', linewidth=2, markersize=6, label='DFT Time')
plt.plot(df['N'], df['FFT_Time_us'], 'bo-', linewidth=2, markersize=6, label='FFT Time')

# Настройка графика
plt.xlabel('Array Size (N)')
plt.ylabel('Time (microseconds)')
plt.title('ДПФ vs БПФ Time')
plt.legend()
plt.grid(True, alpha=0.3)

# Логарифмическая шкала для оси X
plt.xscale('log', base=2)
plt.xticks(df['N'])

# Сохранение и отображение графика
plt.tight_layout()
plt.savefig('/workspaces/UnicLab/Lab2/OtherFiles/time_plot.png')
plt.show()

