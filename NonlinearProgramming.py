# 1変数の場合のバックトラック法を用いた最急降下法
def f(x):
    return 3*x**2
# 微分
def df(x):
    return 6*x

def GradientDescentMethod(x):
    for i in range(n):
        delta = - df(x)**2
        if i == 0:
            alpha = alpha_bar
        # Armijoの条件
        while f(x + alpha * delta) > f(x) + c * alpha * delta:
            alpha = rho * alpha
        x = x + alpha * delta
    return f(x)

if __name__ == '__main__':
    x = 10
    n = 10
    rho = 0.39
    c = 0.08
    alpha_bar = rho**2 - x*c
    gamma = 0.4
    print(GradientDescentMethod(x))