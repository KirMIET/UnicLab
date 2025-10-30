#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>

using namespace std;

// ДПФ
vector<complex<double>> dft(const vector<complex<double>>& x) {
    int N = x.size();
    vector<complex<double>> y(N);
    double factor = 1.0 / sqrt(N);
    
    for (int k = 0; k < N; k++) {
        complex<double> sum(0, 0);
        for (int j = 0; j < N; j++) {
            double angle = - 2 * M_PI * j * k / N;
            complex<double> w(cos(angle), sin(angle));
            sum += x[j] * w;
        }
        y[k] = sum * factor;
    }
    return y;
}

// ОДПФ
vector<complex<double>> idft(const vector<complex<double>>& y) {
    int N = y.size();
    vector<complex<double>> x(N);
    double factor = 1.0 / sqrt(N);
    
    for (int j = 0; j < N; j++) {
        complex<double> sum(0, 0);
        for (int k = 0; k < N; k++) {
            double angle = 2 * M_PI * j * k / N;
            complex<double> w(cos(angle), sin(angle));
            sum += y[k] * w;
        }
        x[j] = sum * factor;
    }
    return x;
}

// Чтение комплексных чисел из бинарного файла
vector<complex<double>> read_complex_vector(const string& filename) {
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    // Определяем размер файла
    file.seekg(0, ios::end);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    
    // Читаем данные
    vector<double> data(size / sizeof(double));
    file.read(reinterpret_cast<char*>(data.data()), size);
    
    // Преобразуем в комплексные числа (real, imag пары)
    vector<complex<double>> result;
    for (size_t i = 0; i < data.size(); i += 2) {
        result.emplace_back(data[i], data[i + 1]);
    }
    
    return result;
}

// Запись комплексных чисел в бинарный файл
void write_complex_vector(const string& filename, const vector<complex<double>>& vec) {
    ofstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    // Записываем как последовательность double (real, imag)
    vector<double> data;
    for (const auto& c : vec) {
        data.push_back(c.real());
        data.push_back(c.imag());
    }
    
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(double));
}

int main() {
    try {
        // Чтение входных данных
        vector<complex<double>> X = read_complex_vector("/workspaces/UnicLab/Lab2/OtherFiles/input.bin");
        cout << "Read " << X.size() << " complex numbers" << endl;
        
        // ДПФ, ОДПФ
        vector<complex<double>> Y = dft(X);
        cout << "DFT computed" << endl;
        
        vector<complex<double>> X_reconstructed = idft(Y);
        cout << "IDFT computed" << endl;
        
        // Сохранение результатов
        write_complex_vector("/workspaces/UnicLab/Lab2/OtherFiles/dft_result.bin", Y);
        write_complex_vector("/workspaces/UnicLab/Lab2/OtherFiles/idft_result.bin", X_reconstructed);
        cout << "Results saved to files" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}