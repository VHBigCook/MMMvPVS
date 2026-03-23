#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>

using namespace std;

// --------------------------
// 1. Гауссово ядро
// --------------------------
double gaussian_kernel(double u) {
    return exp(-0.5 * u * u) / sqrt(2 * M_PI);
}

// --------------------------
// 2. Оценка Парзена
// --------------------------
double parzen_regression(const vector<double>& x,
                         const vector<double>& y,
                         double x0,
                         double beta,
                         int exclude_index = -1) {
    double num = 0.0;
    double den = 0.0;

    for (int i = 0; i < x.size(); i++) {
        if (i == exclude_index) continue;

        double u = (x0 - x[i]) / beta;
        double w = gaussian_kernel(u);

        num += w * y[i];
        den += w;
    }

    if (den == 0) return 0;
    return num / den;
}

// --------------------------
// 3. LOOCV
// --------------------------
double loocv_mse(const vector<double>& x,
                 const vector<double>& y,
                 double beta) {
    int n = x.size();
    double mse = 0.0;

    for (int i = 0; i < n; i++) {
        double y_pred = parzen_regression(x, y, x[i], beta, i);
        double err = y[i] - y_pred;
        mse += err * err;
    }

    return mse / n;
}

// --------------------------
// 4. Главная функция
// --------------------------
int main() {
    const int n = 80;

    vector<double> x(n), y(n);

    // Генераторы случайных чисел
    random_device rd;
    mt19937 gen(rd());

    uniform_real_distribution<> dist_x(-3.0, 7.0);
    normal_distribution<> dist_noise(0.0, 0.3);

    // --------------------------
    // Генерация данных
    // --------------------------
    for (int i = 0; i < n; i++) {
        x[i] = dist_x(gen);
        double noise = dist_noise(gen);

        // предполагаем, что "cox(x)" = cos(x)
        y[i] = 0.1 * (x[i] - 4) * cos(x[i]) + 0.5 * x[i] + noise;
    }

    // --------------------------
    // Подбор beta
    // --------------------------
    double best_beta = 0.1;
    double best_mse = 1e9;

    cout << "Beta\tMSE\n";

    for (double beta = 0.1; beta <= 2.0; beta += 0.1) {
        double mse = loocv_mse(x, y, beta);
        cout << beta << "\t" << mse << endl;

        if (mse < best_mse) {
            best_mse = mse;
            best_beta = beta;
        }
    }

    cout << "\nЛучшее beta: " << best_beta << endl;
    cout << "Минимальное MSE: " << best_mse << endl;

    // --------------------------
    // Прогноз (пример)
    // --------------------------
    cout << "\nПример предсказаний:\n";

    for (double xi = -3; xi <= 7; xi += 1.0) {
        double yi = parzen_regression(x, y, xi, best_beta);
        cout << "x=" << xi << " y_hat=" << yi << endl;
    }

    return 0;
}
