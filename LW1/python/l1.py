import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Генерация данных
# --------------------------
np.random.seed(42)

n = 80
x = np.random.uniform(-3, 7, n)

# шум N(0, 0.3)
noise = np.random.normal(0, 0.3, n)

# функция (предполагаю, что "cox(x)" = cos(x))
y = 0.1 * (x - 4) * np.cos(x) + 0.5 * x + noise


# --------------------------
# 2. Ядро (гауссово)
# --------------------------
def gaussian_kernel(u):
    return np.exp(-0.5 * u**2) / np.sqrt(2 * np.pi)


# --------------------------
# 3. Оценка Розенблатта–Парзена
# --------------------------
def parzen_regression(x_train, y_train, x0, beta):
    u = (x0 - x_train) / beta
    weights = gaussian_kernel(u)
    
    if np.sum(weights) == 0:
        return 0
    
    return np.sum(weights * y_train) / np.sum(weights)


# --------------------------
# 4. LOOCV (скользящий экзамен)
# --------------------------
def loocv_mse(x, y, beta):
    n = len(x)
    errors = []
    
    for i in range(n):
        x_train = np.delete(x, i)
        y_train = np.delete(y, i)
        
        y_pred = parzen_regression(x_train, y_train, x[i], beta)
        errors.append((y[i] - y_pred)**2)
    
    return np.mean(errors)


# --------------------------
# 5. Подбор beta
# --------------------------
betas = np.arange(0.1, 2.1, 0.1)
mse_values = []

for beta in betas:
    mse = loocv_mse(x, y, beta)
    mse_values.append(mse)

best_beta = betas[np.argmin(mse_values)]
best_mse = np.min(mse_values)

print(f"Лучшее beta: {best_beta}")
print(f"MSE: {best_mse}")


# --------------------------
# 6. Построение регрессии
# --------------------------
x_grid = np.linspace(-3, 7, 200)
y_pred = [parzen_regression(x, y, xi, best_beta) for xi in x_grid]


# --------------------------
# 7. Визуализация
# --------------------------
plt.scatter(x, y, label="Данные")
plt.plot(x_grid, y_pred, linewidth=2, label=f"Регрессия (beta={best_beta})")
plt.legend()
plt.title("Непараметрическая регрессия (Розенблатт–Парзен)")
plt.show()


# --------------------------
# 8. График ошибки
# --------------------------
plt.plot(betas, mse_values, marker='o')
plt.xlabel("beta")
plt.ylabel("MSE")
plt.title("Подбор параметра beta")
plt.show()
