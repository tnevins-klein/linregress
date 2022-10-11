import numpy as np


class LinearRegression:
    def __init__(self, xs=np.zeros(10), ys=np.zeros(10)):
        self.xs = xs
        self.ys = ys
        self.slope = 0
        self.intercept = 0

    def loss(self, slope=None, intercept=None):
        """
        Returns the sum of the square of the residuals with a given slope and intercept
        """
        if slope is None:
            slope = self.slope

        if intercept is None:
            intercept = self.intercept

        return np.sum((slope * self.xs + intercept - self.ys) ** 2)

    def fit(self, delta=1e-5, gamma=1**-5, max_iter=3457807, debug_log=False):
        """
        Uses gradient descent to find the slope and y-intercept values (i.e. regression
        values) that minimize the loss function.

        This is NOT an efficient way to do linear regression. There are FAR better
        ways to do it.

        Parameters
        ----------
        delta : float
            The interval used to approximate the instantaneous slope at a given point.
        gamma : float
            The starting learning rate for the descent.
        max_iter : int
            The maximum number of iterations allowed for the descent. Going above the
            default value (3457807) will yield no additional accuracy, because gamma
            reaches 0.
        debug_log : bool
            Determines if debug log messages should be printed to standard output at
            each iteration
        """

        for it in range(1, max_iter + 1):
            slope_gradient = (self.loss(self.slope + delta) - self.loss(self.slope - delta)) / (2 * delta)
            intercept_gradient = (self.loss(intercept=self.intercept + delta) - self.loss(self.slope - delta)) / (
                        2 * delta)

            new_slope = self.slope - gamma * slope_gradient
            new_intercept = self.intercept - gamma * intercept_gradient

            if debug_log:
                print(
                    f"{it}, grad: (s={slope_gradient}, "
                    f"i={intercept_gradient}), slope: {new_slope}, int: {self.intercept}, gamma: {gamma}"
                )

                if gamma == 0.0:
                    print(f"Gamma reached 0. Stopping after {it} iterations.")
                    break

            if self.loss(new_slope, intercept=new_intercept) >= self.loss(self.slope, intercept=self.intercept):
                gamma /= 2
            else:
                self.slope = new_slope
                self.intercept = new_intercept

        return self.slope, self.intercept

    def predict(self, xs) -> [float]:
        return xs * self.slope


def main():
    adelie_bill_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=0)
    adelie_flipper_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=1)

    regression = LinearRegression(adelie_bill_len_mm, adelie_flipper_len_mm)
    print(f"Starting loss: f{regression.loss()}")

    slope, intercept = regression.fit(debug_log=True)
    print(f"Final values: {slope=}, {intercept=}, loss={regression.loss()}")


if __name__ == '__main__':
    main()
