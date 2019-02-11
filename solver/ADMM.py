import numpy as np
import numba


class ADMM(object):
    def __init__(self, A, y, regularization_strength, rho=1e-3):
        """ constructor """
        self.A = A.copy()
        self.y = y.copy()
        self.l = regularization_strength
        self.pre_l = self.l.copy()
        self.rho = rho
        self.M, self.N = self.A.shape

        self.y_tilde = self.A.T @ self.y
        self.inv_ = np.linalg.inv(self.A.T @ self.A + np.eye(self.N) * self.rho)

        self.x = self.A.T @ self.y
        self.z = self.x.copy()
        self.b = np.zeros(self.N)

        self.variable_history_list = []
        self.regularization_changed_flag = []

    @numba.jit(nogil=True)
    def solve(self, max_iteration=1000, tol=1e-3):
        """ solver """
        convergence_flag = False
        diff = 999999
        for iteration_index in range(max_iteration):
            self.variable_history_list.append(self.return_variables())
            if np.any(self.pre_l != self.l) and iteration_index == 0:
                self.regularization_changed_flag.append(1)
            else:
                self.regularization_changed_flag.append(0)
            pre_x = self.x.copy()
            self._update_x()
            self._update_z()
            self._update_b()
            # diff = np.linalg.norm(pre_x - self.x) / self.N ** 0.5
            diff = max(np.abs(pre_x - self.x))
            if diff < tol and iteration_index > 3:
                convergence_flag = True
                print("converged")
                print("diff = {0}".format(diff))
                print("iteration num = {0}".format(iteration_index + 1))
                break

        if not convergence_flag:
            print("doesn't converged")
            print("diff = {0}".format(diff))

        self.pre_l = self.l.copy()

    def _update_x(self):
        self.x = self.inv_ @ (self.y_tilde + self.rho * self.z - self.b)

    def _update_z(self):
        self.z = soft_threshold(self.x + self.b / self.rho, self.l / self.rho)

    def _update_b(self):
        self.b = self.b + self.rho * (self.x - self.z)

    def show_me(self):
        print("x.mean = ", self.x.mean())
        print("z.mean = ", self.z.mean())
        print("b.mean = ", self.b.mean())

    def return_variables(self):
        return [
            self.x.mean(),
            self.z.mean(),
            self.b.mean()
        ]


def soft_threshold(h, l):
    return (h - l * np.sign(h)) * np.heaviside(np.abs(h) - l, 0.5)
