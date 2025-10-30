#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <string>

using namespace std;

// Функция для чтения комплексных чисел из бинарного файла
vector<complex<double>> read_complex_binary(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    // Определение размера файла
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    
    // Вычисление количества комплексных чисел в файле
    int num_complex = size / (2 * sizeof(double));
    vector<complex<double>> result(num_complex);
    
    // Чтение сырых данных как массива double
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
    
    vector<double> raw_data(2 * data.size());
    for (size_t i = 0; i < data.size(); i++) {
        raw_data[2 * i] = data[i].real();
        raw_data[2 * i + 1] = data[i].imag();
    }
    
    // Запись данных в файл
    file.write(reinterpret_cast<const char*>(raw_data.data()), raw_data.size() * sizeof(double));
}

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

int main(int argc, char* argv[]) {
    string x_filename = "/workspaces/UnicLab/Lab2/OtherFiles/x_signal.bin";
    string y_filename = "/workspaces/UnicLab/Lab2/OtherFiles/y_signal.bin";
    string output_filename = "/workspaces/UnicLab/Lab2/OtherFiles/direct_convolution_result.bin";
    
    if (argc > 1) x_filename = argv[1];
    if (argc > 2) y_filename = argv[2];
    if (argc > 3) output_filename = argv[3];
    
    try {
        // Чтение входных сигналов
        vector<complex<double>> x = read_complex_binary(x_filename);
        vector<complex<double>> y = read_complex_binary(y_filename);
        
        // Вычисление прямой свертки
        vector<complex<double>> convolution_result = direct_convolution(x, y);
        
        // Сохранение результата
        write_complex_binary(output_filename, convolution_result);
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}