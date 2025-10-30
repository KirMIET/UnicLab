#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <fstream>
#include <string>

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

// Чтение комплексных чисел из бинарного файла 
vector<complex<double>> read_complex_binary(const string& filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    
    int num_complex = size / (2 * sizeof(double));
    vector<complex<double>> result(num_complex);
    
    // Чтение всех данных как double
    vector<double> raw_data(2 * num_complex);
    file.read(reinterpret_cast<char*>(raw_data.data()), size);
    
    // Конвертация в комплексные числа (пары действительная/мнимая части)
    for (int i = 0; i < num_complex; i++) {
        result[i] = complex<double>(raw_data[2 * i], raw_data[2 * i + 1]);
    }
    
    return result;
}

// Запись комплексных чисел в бинарный файл
void write_complex_binary(const string& filename, const vector<complex<double>>& data) {
    ofstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Cannot open file: " + filename);
    }
    
    // Конвертация комплексных чисел в массив double
    vector<double> raw_data(2 * data.size());
    for (size_t i = 0; i < data.size(); i++) {
        raw_data[2 * i] = data[i].real();
        raw_data[2 * i + 1] = data[i].imag();
    }
    
    file.write(reinterpret_cast<const char*>(raw_data.data()), raw_data.size() * sizeof(double));
}

int main(int argc, char* argv[]) {
    string input_filename = "/workspaces/UnicLab/Lab2/OtherFiles/input.bin";
    string output_fft_filename = "/workspaces/UnicLab/Lab2/OtherFiles/fft_result.bin";
    string output_ifft_filename = "/workspaces/UnicLab/Lab2/OtherFiles/ifft_result.bin";
    
    if (argc > 1) {
        input_filename = argv[1];
    }
    if (argc > 2) {
        output_fft_filename = argv[2];
    }
    if (argc > 3) {
        output_ifft_filename = argv[3];
    }
    
    try {
        vector<complex<double>> signal = read_complex_binary(input_filename);
        
        vector<complex<double>> spectrum = fft(signal);
        
        vector<complex<double>> reconstructed = ifft(spectrum);
        
        // Сохранение результатов
        write_complex_binary(output_fft_filename, spectrum);
        write_complex_binary(output_ifft_filename, reconstructed);
        cout << "Results saved" << endl;
        
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    return 0;
}