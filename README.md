# slp-XUYING
SLP
import random
import numpy as np
import matplotlib.pyplot as plt

xh_number=1000


class A:
    def __init__(self, lr=0.1):
        self.lr = lr
        self.errors = []
        self.b = random.random()
        self.w = [0.13, -0.144]

    def train(self, Px, t):
        update = self.lr * (t - self.predict(Px))
        self.b += update
        self.w[0] += update*Px[0]
        self.w[1] += update*Px[1]

    def predict(self, Px):
        number = self.w[0]*Px[0] + self.w[1]*Px[1] + self.b
        return np.where(number >=0, 1, 0)


def main():

    P = [[-0.5, -0.5], [-0.5, 0.5], [0.3, -0.5], [0, 1]]
    T = [1.0, 1.0, 0, 0]
    a = A (0.1)
    plt.figure()
    plt.plot([-0.5, -0.5], [-0.5, 0.5], 'bo')
    plt.plot([0.3, 0], [-0.5, 1],'bo' )
    for i in range(xh_number):
        for i in range(4):
            Px = P[i]
            t = T[i]
            a.train(Px, t)
        print(-a.w[0]/a.w[1])
        print(-a.b/a.w[1])
    x = np.arange(-1, 1)
    y = -a.w[0]/a.w[1]*x-a.b/a.w[1]
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":
    main()
