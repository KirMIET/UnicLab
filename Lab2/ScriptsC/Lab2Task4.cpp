#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <chrono>
#include <random>
#include <fstream>
#include <string>

using namespace std;
using namespace std::chrono;

// ДПФ
vector<complex<double>> direct_dft(const vector<complex<double>>& x) {
    int N = x.size();
    vector<complex<double>> y(N);
    
    for (int k = 0; k < N; k++) {
        complex<double> sum(0, 0);
        for (int j = 0; j < N; j++) {
            double angle = -2 * M_PI * j * k / N;
            complex<double> w(cos(angle), sin(angle));
            sum += x[j] * w;
        }
        y[k] = sum;
    }
    return y;
}

// БПФ с прореживанием по частоте
vector<complex<double>> fft(const vector<complex<double>>& input) {
    vector<complex<double>> x = input;
    int N = x.size();

    for (int step = N / 2; step >= 1; step /= 2) {
        int jump = step * 2;
        for (int start = 0; start < N; start += jump) {
            for (int k = 0; k < step; k++) {
                double angle = -2.0 * M_PI * k / jump;
                complex<double> w(cos(angle), sin(angle));

                complex<double> a = x[start + k];
                complex<double> b = x[start + k + step];

                x[start + k] = a + b;
                x[start + k + step] = (a - b) * w;
            }
        }
    }

    return x;
}

// Генерация случайного комплексного сигнала
vector<complex<double>> generate_signal(int N) {
    vector<complex<double>> signal(N);
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (int i = 0; i < N; i++) {
        signal[i] = complex<double>(dis(gen), dis(gen));
    }
    return signal;
}

// Измерение времени выполнения 
void measure_performance() {
    ofstream time_file("/workspaces/UnicLab/Lab2/OtherFiles/execution_times.csv");
    
    // Заголовок для CSV файла
    time_file << "N,DFT_Time_us,FFT_Time_us" << endl;
    
    cout << "N\tDFT (mksec)\tFFT (mksec)" << endl;
    
    // Размеры данных для тестирования 
    vector<int> sizes;
    for (int n = 6; n <= 12; n++) {
        sizes.push_back(1 << n); 
    }
    
    // Тестирование для каждого размера
    for (int N : sizes) {
        cout << N << "\t";
        
        auto signal = generate_signal(N);
        
        // Измерение времени ДПФ
        auto start = high_resolution_clock::now();
        auto dft_result = direct_dft(signal);
        auto stop = high_resolution_clock::now();
        auto dft_time_us = duration_cast<microseconds>(stop - start).count();
        
        // Измерение времени БПФ
        start = high_resolution_clock::now();
        auto fft_result = fft(signal);
        stop = high_resolution_clock::now();
        auto fft_time_us = duration_cast<microseconds>(stop - start).count();
        
        // Вывод в консоль
        cout << dft_time_us << "\t\t" << fft_time_us << endl;
        
        // Запись в CSV файл
        time_file << N << "," << dft_time_us << "," << fft_time_us << endl;
    }
    
    time_file.close();
}

int main() {
    try {
        measure_performance();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}