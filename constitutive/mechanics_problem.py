import dolfin as df
from . import helper as h
from .cpp import *

class Parameters:
    def __init__(self, constraint):
        self.constraint = constraint

        # interpolation degree of displacement field
        self.deg_d = 2

        # quadrature degree
        self.deg_q = 2

        self.E = 20000.
        self.nu = 0.2
        self.ft = 4.
        self.alpha = 0.99
        self.gf = 0.1
        self.k = 10.


class MechanicsProblem(df.NonlinearProblem):
    def __init__(self, mesh, prm, law, iploop=None):
        df.NonlinearProblem.__init__(self)

        self.mesh = mesh
        self.prm = prm

        self.law = law
        self.iploop = iploop or IpLoop()
        self.iploop.add_law(self.law)

        if mesh.geometric_dimension() != g_dim(prm.constraint):
            raise RuntimeError(
                f"The geometric dimension of the mesh does not match the {prm.constraint} constraint."
            )

        metadata = {"quadrature_degree": prm.deg_q, "quadrature_scheme": "default"}
        self.dxm = df.dx(metadata=metadata)

        # solution field
        self.V = df.VectorFunctionSpace(mesh, "CG", degree=prm.deg_d)
        self.d = df.Function(self.V, name="displacement field")

        # generic quadrature function spaces
        VQF, VQV, VQT = h.spaces(mesh, prm.deg_q, q_dim(prm.constraint))

        # quadrature function
        self.q_sigma = df.Function(VQV, name="current stresses")
        self.q_eps = df.Function(VQV, name="current strains")
        self.q_dsigma_deps = df.Function(VQT, name="stress-strain tangent")

        n_gauss_points = len(self.q_eps.vector().get_local()) // q_dim(prm.constraint)
        self.iploop.resize(n_gauss_points);

        dd, d_ = df.TrialFunction(self.V), df.TestFunction(self.V)

        eps = self.eps
        self.R = df.inner(eps(d_), self.q_sigma) * self.dxm
        self.dR = df.inner(eps(dd), self.q_dsigma_deps * eps(d_)) * self.dxm

        self.calculate_eps = h.LocalProjector(eps(self.d), VQV, self.dxm)

        self._assembler = None
        self._bcs = None

    def add_force_term(self, term):
        self.R -= term
        if self._bcs is not None:
            # update it to the new self.R!
            self.set_bcs(self._bcs)

    @property
    def Vd(self):
        """
        Return the function space for the displacements. This distinction is
        required for mixed problems where V != Vd.
        """
        return self.V

    @property
    def u(self):
        """
        Return the full (for all dofs) solution field. 
        """
        return self.d

    def eps(self, u):
        e = df.sym(df.grad(u))
        dim = self.mesh.geometric_dimension()
        if dim == 1:
            return df.as_vector([e[0, 0]])

        if dim == 2:
            return df.as_vector([e[0, 0], e[1, 1], 2 * e[0, 1]])

        if dim == 3:
            return df.as_vector(
                [e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]]
            )

    def evaluate_material(self):
        # project the strain and the nonlocal equivalent strains onto
        # their quadrature spaces and ...
        self.calculate_eps(self.q_eps)
        self.iploop.evaluate(self.q_eps.vector().get_local())

        # ... and write the calculated values into their quadrature spaces.
        h.set_q(self.q_sigma, self.iploop.get(Q.SIGMA))
        h.set_q(self.q_dsigma_deps, self.iploop.get(Q.DSIGMA_DEPS))

    def update(self):
        self.calculate_eps(self.q_eps)
        self.iploop.update(self.q_eps.vector().get_local())


    def set_bcs(self, bcs):
        # Only now (with the bcs) can we initialize the _assembler
        self._bcs = bcs
        self._assembler = df.SystemAssembler(self.dR, self.R, bcs)

    def F(self, b, x):
        if not self._assembler:
            raise RuntimeError("You need to `.set_bcs(bcs)` before the solve!")
        self.evaluate_material()
        self._assembler.assemble(b, x)

    def J(self, A, x):
        self._assembler.assemble(A)

    def solve(self, solver=None):
        if solver is None:
            solver = df.NewtonSolver()
        solver.solve(self, self.u.vector())
        return self.u
