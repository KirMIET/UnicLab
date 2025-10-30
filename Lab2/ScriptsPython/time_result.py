import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Чтение данных из CSV файла
df = pd.read_csv('/workspaces/UnicLab/Lab2/OtherFiles/execution_times.csv')

# Создание графика
plt.figure(figsize=(10, 6))

# Построение линий для DFT и FFT с логарифмическим масштабом
plt.loglog(df['N'], df['DFT_Time_us'], 'ro-', linewidth=2, markersize=6, label='DFT Time', base=2)
plt.loglog(df['N'], df['FFT_Time_us'], 'bo-', linewidth=2, markersize=6, label='FFT Time', base=2)

# Настройка графика
plt.xlabel('Array Size (N)')
plt.ylabel('Time (microseconds)')
plt.title('ДПФ vs БПФ Time (Log-Log Scale)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.3)

# Установка меток по оси X
plt.xticks(df['N'])

# Сохранение и отображение графика
plt.tight_layout()
plt.savefig('/workspaces/UnicLab/Lab2/OtherFiles/time_plot.png')
plt.show()