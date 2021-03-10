import unittest
import constitutive as c
from fenics_helpers import boundary
from dolfin import *
import math

class PlateWithHoleSolution:
    def __init__(self, E, nu, radius=1.0, L=10.0, load=10.0):
        self.radius = radius
        self.L = L
        self.load = load
        self.E = E
        self.nu = nu

    def polar(self, x):
        r = math.hypot(x[0], x[1])
        theta = math.atan2(x[1], x[0])
        return r, theta

    def displacement(self, x):
        r, theta = self.polar(x)
        a = self.radius

        T = self.load
        Ta_8mu = T * a / (4 * self.E / (1.0 + 1.0 * self.nu))
        k = (3.0 - self.nu) / (1.0 + self.nu)

        ct = math.cos(theta)
        c3t = math.cos(3 * theta)
        st = math.sin(theta)
        s3t = math.sin(3 * theta)

        fac = 2 * math.pow(a / r, 3)

        ux = Ta_8mu * (
            r / a * (k + 1.0) * ct + 2.0 * a / r * ((1.0 + k) * ct + c3t) - fac * c3t
        )

        uy = Ta_8mu * (
            (r / a) * (k - 3.0) * st + 2.0 * a / r * ((1.0 - k) * st + s3t) - fac * s3t
        )

        return ux, uy

    def stress(self, x):
        r, theta = self.polar(x)
        T = self.load
        a = self.radius
        cos2t = math.cos(2 * theta)
        cos4t = math.cos(4 * theta)
        sin2t = math.sin(2 * theta)
        sin4t = math.sin(4 * theta)

        fac1 = (a * a) / (r * r)
        fac2 = 1.5 * fac1 * fac1

        sxx = T - T * fac1 * (1.5 * cos2t + cos4t) + T * fac2 * cos4t
        syy = -T * fac1 * (0.5 * cos2t - cos4t) - T * fac2 * cos4t
        sxy = -T * fac1 * (0.5 * sin2t + sin4t) + T * fac2 * sin4t

        return sxx, syy, sxy

    def build_mesh(self, resolution=20):
        from mshr import Rectangle, Circle, generate_mesh

        domain = Rectangle(Point(0.0, 0.0), Point(self.L, self.L)) - Circle(
            Point(0.0, 0.0), self.radius
        )
        return generate_mesh(domain, resolution)

"""
When subclassing from ``dolfin.UserExpression``, make sure to override
``value_shape`` for non-scalar expressions.
"""

class StressSolution(UserExpression):
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution

    def eval(self, value, x):
        sxx, syy, sxy = self.solution.stress(x)

        value[0] = sxx
        value[1] = sxy
        value[2] = sxy
        value[3] = syy

    def value_shape(self):
        return (2, 2)


class DisplacementSolution(UserExpression):
    def __init__(self, solution, **kwargs):
        super().__init__(**kwargs)
        self.solution = solution

    def eval(self, value, x):
        ux, uy = self.solution.displacement(x)
        value[0] = ux
        value[1] = uy

    def value_shape(self):
        return (2,)


class TestPlate(unittest.TestCase):
    def test_plate(self):
        L, radius = 4.0, 1.0

        prm = c.Parameters(c.Constraint.PLANE_STRESS)

        plate_with_hole = PlateWithHoleSolution(L=L, E=prm.E, nu=prm.nu, radius=radius)
        mesh = plate_with_hole.build_mesh(resolution=50)

        n = FacetNormal(mesh)
        stress = StressSolution(plate_with_hole, degree=2)
        traction = dot(stress, n)
  
        problem = c.MechanicsProblem(mesh, prm, c.LinearElastic(prm.E, prm.nu, prm.constraint))
        bc0 = DirichletBC(problem.Vd.sub(0), 0.0, boundary.plane_at(0, "x"))
        bc1 = DirichletBC(problem.Vd.sub(1), 0.0, boundary.plane_at(0, "y"))
        problem.set_bcs([bc0, bc1])
        problem.add_force_term(dot(TestFunction(problem.Vd), traction)*ds)
        u = problem.solve() 
        
        disp = DisplacementSolution(plate_with_hole, degree=2)
        error = errornorm(disp, u)
        self.assertLess(error, 1.e-6)

if __name__ == "__main__":
    unittest.main()
