# 1変数の場合のバックトラック法を用いた最急降下法
def f(x):
    return 3*x**2

def df(x):
    return 6*x

def GradientDescent(x):
    for i in range(n):
        delta = - df(x)**2
        if i == 0:
            alpha = alpha_bar
        # Armijoの条件
        while f(x + alpha * delta) > f(x) + c * alpha * delta:
            alpha = rho * alpha
            print(alpha)
        x = x  + alpha * delta
    return f(x)


if __name__ = '__main__'

x = 10
n = 10
alpha_bar = 0.1
c = 0.8
rho = 0.3
gamma = 0.4

print(GradientDescent(x))