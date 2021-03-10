import unittest
import dolfin as df
import numpy as np
from fenics_helpers import boundary
from fenics_helpers.timestepping import TimeStepper
import constitutive as c


def cdf2(f, x, delta):
    N = len(x)
    f_cdf = np.empty((N, N))
    for i in range(N):
        d = np.zeros_like(x)
        d[i] = delta
        f_cdf[:, i] = (f(x + d) - f(x - d)) / (2 * delta)
    return f_cdf


def law_from_prm(prm):
    return c.LocalDamage(
        prm.E,
        prm.nu,
        prm.constraint,
        c.DamageLawExponential(prm.ft / prm.E, prm.alpha, prm.ft / prm.gf),
        c.ModMisesEeq(prm.k, prm.nu, prm.constraint),
    )


class TestUniaxial(unittest.TestCase):
    def test_tangent(self):
        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        law = law_from_prm(prm)
        law.resize(1)

        def only_sigma(x):
            return law.evaluate(x, 0)[0]

        np.random.seed(6174)
        for i in range(42):
            strain = np.random.random(3)
            sigma, dsigma = law.evaluate(strain, 0)
            dsigma_cdf = cdf2(only_sigma, strain, 1.0e-6)

            self.assertLess(np.linalg.norm(dsigma - dsigma_cdf), 1.0e-4)

    def test_zero(self):
        prm = c.Parameters(c.Constraint.PLANE_STRESS)
        law = law_from_prm(prm)
        law.resize(1)
        sigma, dsigma = law.evaluate([0, 0, 0])
        self.assertLess(np.max(np.abs(sigma)), 1.0e-10)
        self.assertFalse(np.any(np.isnan(dsigma)))

    def test_uniaxial(self):
        prm = c.Parameters(c.Constraint.UNIAXIAL_STRESS)
        prm.deg_d = 1
        prm.alpha = 0.999
        prm.gf = 0.01
        k0 = 2.0e-4
        prm.ft = k0*prm.E
        law = law_from_prm(prm)

        mesh = df.UnitIntervalMesh(1)
        problem = c.MechanicsProblem(mesh, prm, law)

        bc0 = df.DirichletBC(problem.Vd, [0], boundary.plane_at(0))
        bc_expr = df.Expression(["u"], u=0, degree=0)
        bc1 = df.DirichletBC(problem.Vd, bc_expr, boundary.plane_at(1))

        problem.set_bcs([bc0, bc1])

        ld = c.helper.LoadDisplacementCurve(bc1)

        # ld.show()

        solver = df.NewtonSolver()
        solver.parameters["linear_solver"] = "mumps"
        solver.parameters["maximum_iterations"] = 10
        solver.parameters["error_on_nonconvergence"] = False

        u_max = 100 * k0
        for u in np.linspace(0, u_max, 101):
            bc_expr.u = u
            converged = solver.solve(problem, problem.u.vector())
            assert converged
            problem.update()
            ld(u, df.assemble(problem.R))

        GF = np.trapz(ld.load, ld.disp)
        self.assertAlmostEqual(GF, 0.5 * k0 ** 2 * prm.E + prm.gf, delta=prm.gf / 100)


if __name__ == "__main__":
    unittest.main()
