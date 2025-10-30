#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

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

// Функция для чтения комплексных чисел из бинарного файла
vector<complex<double>> read_complex_binary(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    
    int num_complex = size / (2 * sizeof(double));
    vector<complex<double>> result(num_complex);
    
    vector<double> raw_data(2 * num_complex);
    file.read(reinterpret_cast<char*>(raw_data.data()), size);
    
    // Преобразование пар double в комплексные числа
    for (int i = 0; i < num_complex; i++) {
        result[i] = complex<double>(raw_data[2 * i], raw_data[2 * i + 1]);
    }
    
    return result;
}

// Функция для записи комплексных чисел в бинарный файл
void write_complex_binary(const string& filename, const vector<complex<double>>& data) {
    ofstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    // Преобразование комплексных чисел в массив double
    vector<double> raw_data(2 * data.size());
    for (size_t i = 0; i < data.size(); i++) {
        raw_data[2 * i] = data[i].real();
        raw_data[2 * i + 1] = data[i].imag();
    }
    
    file.write(reinterpret_cast<const char*>(raw_data.data()), raw_data.size() * sizeof(double));
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

int main(int argc, char* argv[]) {
    string x_filename = "/workspaces/UnicLab/Lab2/OtherFiles/x_signal.bin";
    string y_filename = "/workspaces/UnicLab/Lab2/OtherFiles/y_signal.bin";
    string output_filename = "/workspaces/UnicLab/Lab2/OtherFiles/fft_convolution_result.bin";
    
    // Обработка аргументов командной строки
    if (argc > 1) x_filename = argv[1];
    if (argc > 2) y_filename = argv[2];
    if (argc > 3) output_filename = argv[3];
    
    try {
        // Чтение входных сигналов
        vector<complex<double>> x = read_complex_binary(x_filename);
        vector<complex<double>> y = read_complex_binary(y_filename);
        
        // Вычисление свертки на основе БПФ
        vector<complex<double>> convolution_result = fft_convolution(x, y);
        
        // Сохранение результата
        write_complex_binary(output_filename, convolution_result);
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}