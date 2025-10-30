#include <iostream>
#include <vector>
#include <complex>
#include <chrono>
#include <fstream>
#include <random>
#include <cmath>

using namespace std;
using namespace chrono;

// Вычисление прямой свертки по формуле (3)
vector<complex<double>> direct_convolution(const vector<complex<double>>& x, const vector<complex<double>>& y) {
    int M = x.size();
    int L = y.size();
    int N = M + L - 1;  // Длина результата свертки
    
    vector<complex<double>> result(N, 0.0);
    
    // Вычисление свертки по прямой формуле
    for (int n = 0; n < N; n++) {
        complex<double> sum = 0.0;
        for (int k = 0; k <= n; k++) {
            // Проверка границ массивов
            if (k < M && (n - k) < L) {
                sum += x[k] * y[n - k];
            }
        }
        result[n] = sum;
    }
    
    return result;
}
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

                // Основные "бабочки" 
                x[start + k] = a + b;
                x[start + k + step] = (a - b) * w;
            }
        }
    }

    return x;
}

vector<complex<double>> ifft(const vector<complex<double>>& input) {
    vector<complex<double>> x = input;
    int N = x.size();

    for (int step = 1; step < N; step *= 2) {
        int jump = step * 2;
        for (int start = 0; start < N; start += jump) {
            for (int k = 0; k < step; k++) {
                double angle = 2.0 * M_PI * k / jump;
                complex<double> w(cos(angle), sin(angle));

                complex<double> a = x[start + k];
                complex<double> b = x[start + k + step] * w;

                x[start + k] = a + b;
                x[start + k + step] = a - b;
            }
        }
    }

    // нормализация
    for (auto &v : x) v /= N;

    return x;
}

// Вычисление свертки на основе БПФ
vector<complex<double>> fft_convolution(const vector<complex<double>>& x, const vector<complex<double>>& y) {
    int M = x.size();
    int L = y.size();
    int conv_size = M + L - 1;

    // N_fft: наименьшая степень двойки >= conv_size
    int N_fft = 1;
    while (N_fft < conv_size) N_fft <<= 1;

    // Дополнение нулями
    vector<complex<double>> x_padded(N_fft, 0.0);
    vector<complex<double>> y_padded(N_fft, 0.0);
    copy(x.begin(), x.end(), x_padded.begin());
    copy(y.begin(), y.end(), y_padded.begin());

    // БПФ
    vector<complex<double>> X_hat = fft(x_padded);
    vector<complex<double>> Y_hat = fft(y_padded);

    // Поэлементное умножение 
    vector<complex<double>> U_hat(N_fft);
    for (int i = 0; i < N_fft; ++i) {
        U_hat[i] = X_hat[i] * Y_hat[i];
    }

    // Обратный БПФ
    vector<complex<double>> u_padded = ifft(U_hat);

    // Обрезаем до conv_size
    vector<complex<double>> result(conv_size);
    copy(u_padded.begin(), u_padded.begin() + conv_size, result.begin());

    return result;
}

// Генерация случайных комплексных чисел 
vector<complex<double>> generate_signal(int N) {
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(-1.0, 1.0);
    vector<complex<double>> v(N);
    for (int i = 0; i < N; ++i)
        v[i] = complex<double>(dist(gen), dist(gen));
    return v;
}

int main() {
    vector<int> Ns = {64, 128, 256, 512, 1024, 2048, 4096}; 
    ofstream fout("/workspaces/UnicLab/Lab2/OtherFiles/timing_results.csv");
    fout << "N_x,N_y,method,time_ms\n";

    // Первый случай — x фиксированной длины (512), y меняется
    int fixed_N = 512;
    for (int N : Ns) {
        auto x = generate_signal(fixed_N);
        auto y = generate_signal(N);

        auto t1 = high_resolution_clock::now();
        direct_convolution(x, y);
        auto t2 = high_resolution_clock::now();
        double direct_time = duration<double, milli>(t2 - t1).count();

        auto t3 = high_resolution_clock::now();
        fft_convolution(x, y);
        auto t4 = high_resolution_clock::now();
        double fft_time = duration<double, milli>(t4 - t3).count();

        fout << fixed_N << "," << N << ",direct," << direct_time << "\n";
        fout << fixed_N << "," << N << ",fft," << fft_time << "\n";

        cout << "1 | Nx=" << fixed_N << " Ny=" << N
             << " : direct=" << direct_time << " ms, fft=" << fft_time << " ms\n";
    }

    // Второй случай — длины одинаковые
    for (int N : Ns) {
        auto x = generate_signal(N);
        auto y = generate_signal(N);

        auto t1 = high_resolution_clock::now();
        direct_convolution(x, y);
        auto t2 = high_resolution_clock::now();
        double direct_time = duration<double, milli>(t2 - t1).count();

        auto t3 = high_resolution_clock::now();
        fft_convolution(x, y);
        auto t4 = high_resolution_clock::now();
        double fft_time = duration<double, milli>(t4 - t3).count();

        fout << N << "," << N << ",direct_same," << direct_time << "\n";
        fout << N << "," << N << ",fft_same," << fft_time << "\n";

        cout << "2 | N=" << N
             << " : direct=" << direct_time << " ms, fft=" << fft_time << " ms\n";
    }

    fout.close();
    return 0;
}
