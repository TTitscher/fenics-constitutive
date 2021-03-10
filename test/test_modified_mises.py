import unittest
from constitutive.cpp import ModMisesEeq, Constraint, q_dim
import numpy as np


def cdf(f, x, delta):
    f_cdf = np.empty_like(x)
    for i in range(len(x.T)):
        d = np.zeros_like(x)
        d[i] = delta
        f_cdf[i] = (f(x + d) - f(x - d)) / (2 * delta)
    return f_cdf


class TestMises(unittest.TestCase):
    def random_cdf(self, constraint, k=10.0, nu=0.2):
        np.random.seed(6174)
        norm = ModMisesEeq(k, nu, constraint)

        def only_eeq(x):
            return norm.evaluate(x)[0]

        for i in range(100):
            strain = np.random.random(q_dim(constraint))
            eeq, deeq = norm.evaluate(strain)
            deeq_cdf = cdf(only_eeq, strain, 1.0e-6)
            self.assertLess(np.linalg.norm(deeq - deeq_cdf), 1.0e-6)

    def test_uniaxial_strain(self):
        for c in [
            Constraint.UNIAXIAL_STRAIN,
            Constraint.UNIAXIAL_STRESS,
            Constraint.PLANE_STRESS,
            Constraint.PLANE_STRAIN,
            Constraint.FULL,
        ]:
            self.random_cdf(c)

    def test_zero(self):
        norm = ModMisesEeq(10, 0.2, Constraint.PLANE_STRESS)
        eeq, deeq = norm.evaluate([0, 0, 0])
        self.assertLess(eeq, 1.0e-10)
        self.assertFalse(np.any(np.isnan(deeq)))

    def test_1D(self):
        norm = ModMisesEeq(10, 0.2, Constraint.UNIAXIAL_STRESS)
        eeq, _ = norm.evaluate([42])
        self.assertAlmostEqual(eeq, 42.0)

        eeq_compression, _ = norm.evaluate([-42.0])
        self.assertAlmostEqual(eeq_compression, 42.0 / 10.0)

    def test_2D(self):
        norm = ModMisesEeq(10, 0.2, Constraint.PLANE_STRESS)
        eeq, _ = norm.evaluate([42.0, -0.2 * 42.0, 0])
        self.assertAlmostEqual(eeq, 42.0)

        eeq_compression, _ = norm.evaluate([-42.0, 0.2 * 42, 0])
        self.assertAlmostEqual(eeq_compression, 42.0 / 10.0)

    def test_3D(self):
        k, nu = 10.0, 0.2
        norm = ModMisesEeq(k, nu, Constraint.FULL)
        eeq, _ = norm.evaluate([42.0, -nu * 42.0, -nu * 42, 0, 0, 0])
        self.assertAlmostEqual(eeq, 42.0)

        eeq_compression, _ = norm.evaluate([nu * 42.0, nu * 42, -42, 0, 0, 0])
        self.assertAlmostEqual(eeq_compression, 42.0 / k)


if __name__ == "__main__":
    unittest.main()
