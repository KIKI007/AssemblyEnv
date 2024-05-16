import numpy as np
import scipy.sparse
from AssemblyEnv.py_rigidblock import Assembly, Part
import AssemblyEnv.py_rigidblock as pyrb
import pickle
import polyscope as ps
import polyscope.imgui as psim
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csc_matrix, coo_matrix
import mosek.fusion as mo
from scipy.sparse import vstack
import scipy as sp
import mosek
import sys
import torch
from sksparse.cholmod import cholesky
# Reset since we are using a different mode.
from time import perf_counter



class AssemblyChecker:
	def __init__(self, boundaries=None, rank = 0):
		self.assembly = None
		self.analyzer = None
		self.contacts = None
		self.part_colors = []
		self.inf = 1E5
		self.rank = rank

		if boundaries != None:
			self.init(boundaries)

	def n_part(self):
		return self.assembly.n_part()

	def load_from_file(self, path):
		self.assembly = Assembly()
		self.assembly.from_file(path)
		plane = self.assembly.ground()
		self.assembly.set_boundary(plane, "add")
		self.create_analyzer()

	def init(self, boundaries):
		self.assembly = Assembly()
		for boundary in boundaries:
			block = Part.polygon(np.array(boundary, dtype=float), 1.0)
			self.assembly.add_part(block)

		plane = self.assembly.ground()
		self.assembly.set_boundary(plane, "add")
		self.create_analyzer()

	def create_analyzer(self):
		if self.assembly != None:
			partList = [x for x in range(self.assembly.n_part())]
			self.contacts = self.assembly.contacts(partList, 1.0)
			self.analyzer = self.assembly.analyzer(self.contacts, False)
			self.analyzer.friction = 0.5
			self.analyzer.compute()
			self.K = csc_matrix(self.analyzer.matEq)
			self.g = self.analyzer.vecG
			self.Kf = csc_matrix(self.analyzer.matFr)
			self.create_solver()

	def create_solver(self):
		pass

	def reset(self):
		pass

	def check_stability(self, status):
		[loind, lobnd] = self.analyzer.lobnd(np.array(status))
		[upind, upbnd] = self.analyzer.upbnd(np.array(status))
		varf = cp.Variable(self.analyzer.n_var())
		prob = cp.Problem(cp.Minimize(0), # cp.Minimize((1 / 2) * cp.quad_form(varf, P)),
	                  [self.K @ varf + self.g == 0,
	                   self.Kf @ varf <= 0,
	                   lobnd - varf[loind] <= 0,
	                   varf[upind] - upbnd <= 0])
		try:
			prob.solve(verbose = 1, solver = "SCIPY")
		except cp.SolverError:
			return None
		if prob.status != 'optimal':
			return None
		return prob.value

class AssemblyCheckerADMM(AssemblyChecker):

	def __init__(self, boundaries):
		super(AssemblyCheckerADMM, self).__init__(boundaries)
		self.sigma = 1E-6
		self.rho = 0.1
		self.alpha = 1.6

		self.prime_inf = 1E-4
		self.prime_fea = 1E-4

		self.xk = None
		self.yk = None
		self.inf = 1

		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.data_type = torch.float32
		torch.set_float32_matmul_precision("highest")

	def admm_func(self, xk, yk, zk, A, AT, inv, sigma, rho, alpha, lb, ub):
		with torch.no_grad():
			rhs = sigma * xk + AT @ (rho * zk - yk)
			xt = inv @ rhs
			zt = A @ xt
			x = (xt * alpha + xk * (1 - alpha))
			z = (torch.clip(alpha * zt + (1 - alpha) * zk + yk / rho, lb, ub))
			y = (yk + rho * (alpha * zt + (1 - alpha) * zk - z))
			dy = y - yk
		return [x, y, z, dy]

	def create_solver(self):

		# lhs = σI + ρA'A
		# A = [K, Kf, I]
		I = sp.sparse.identity(self.analyzer.n_var())
		self.A = vstack([self.K, self.Kf, I])
		self.lhs = csc_matrix(self.sigma * sp.sparse.identity(self.A.shape[1]) +(self.A.transpose() @ self.A) * self.rho)

		# factorization
		self.lhs_factor = cholesky(self.lhs)
		inv = self.lhs_factor.inv()

		# cuda
		self.inv = torch.tensor(inv.todense(), device=self.device, dtype=self.data_type)
		self.AT = torch.tensor(self.A.transpose().todense(), device=self.device, dtype=self.data_type)
		self.A = torch.tensor(self.A.todense(), device=self.device, dtype=self.data_type)
	def inf_norm(self, x):
		return torch.max(torch.abs(x))
	def check_stability(self, status):

		[lbind, lbval] = self.analyzer.lobnd(np.array(status))
		[ubind, ubval] = self.analyzer.upbnd(np.array(status))
		lb_f = np.ones(self.analyzer.n_var(), dtype=float) * -self.inf
		lb_f[lbind] = lbval
		ub_f = np.ones(self.analyzer.n_var(), dtype=float) * self.inf
		ub_f[ubind] = ubval
		lb = np.hstack([-self.g, np.ones(self.Kf.shape[0]) * -self.inf, lb_f])
		ub = np.hstack([-self.g, np.zeros(self.Kf.shape[0]), ub_f])
		lb = torch.tensor(lb, device=self.device, dtype=self.data_type)
		ub = torch.tensor(ub, device=self.device, dtype=self.data_type)

		if self.xk == None:
			self.xk = torch.zeros(self.A.shape[1], device=self.device, dtype=self.data_type)
			self.yk = torch.zeros(self.A.shape[0], device=self.device, dtype=self.data_type)
			zk = torch.zeros(self.A.shape[0], device=self.device, dtype=self.data_type)
		else:
			zk = self.A @ self.xk

		start = perf_counter()
		for k in range(5000):
			[self.xk, self.yk, zk, dy] = self.admm_func(self.xk, self.yk, zk, self.A, self.AT, self.inv, self.sigma, self.rho, self.alpha, lb, ub)
		torch.cuda.synchronize()
		end = perf_counter()

		pinf0 = (torch.dot(ub, dy + torch.abs(dy)) / 2.0 + torch.dot(lb, torch.abs(dy) - dy) / 2.0) / self.inf_norm(dy)
		pinf1 = (torch.linalg.norm(self.AT @ dy, ord=torch.inf)) / self.inf_norm(dy)

		#tol_prim = self.tol_abs + self.tol_rel * torch.max(self.inf_norm(self.A @ self.xk), self.inf_norm(zk))
		r_prim = self.inf_norm(self.A @ self.xk - zk)
		print(f"time {end - start}, prim_inf {pinf1}, prim_fea {r_prim.cpu()}")

		if pinf0 < self.prime_inf and pinf1 < self.prime_inf:
			return None
		elif r_prim < self.prime_fea:
			return r_prim
		return None

class AssemblyCheckerGurobi(AssemblyChecker):

	def __init__(self, boundaries):
		super(AssemblyCheckerGurobi, self).__init__(boundaries)

	def create_solver(self):
		self.solver = gp.Model()
		self.solver.setParam('LogFile', "")
		self.solver.setParam('LogToConsole', 0)
		self.solver.setParam('OutputFlag', 0)
		self.varf = self.solver.addMVar(shape=self.analyzer.n_var(), vtype=GRB.CONTINUOUS, name="varf")
		self.solver.addConstr(self.K @ self.varf + self.g == 0)
		self.solver.addConstr(self.Kf @ self.varf <= 0)
		self.solver.setObjective(0, GRB.MINIMIZE)

	def check_stability(self, status):
		[loind, lobnd] = self.analyzer.lobnd(np.array(status))
		[upind, upbnd] = self.analyzer.upbnd(np.array(status))
		for i in range(self.analyzer.n_var()):
			self.varf[i].UB = +GRB.INFINITY
			self.varf[i].LB = -GRB.INFINITY

		for i in range(len(loind)):
			self.varf[loind[i]].LB = lobnd[i]

		for i in range(len(upind)):
			self.varf[upind[i]].UB = upbnd[i]

		self.solver.optimize()
		if self.solver.status == GRB.OPTIMAL:
			return 0
		else:
			return None

class AssemblyCheckerMosek(AssemblyChecker):

	def __init__(self, boundaries = None):
		super(AssemblyCheckerMosek, self).__init__(boundaries)

	def create_solver(self):

		self.solver = mo.Model()
		K_coo = scipy.sparse.coo_matrix(self.K)
		Kf_coo = scipy.sparse.coo_matrix(self.Kf)
		K = mo.Matrix.sparse(K_coo.shape[0], K_coo.shape[1], K_coo.row, K_coo.col, K_coo.data)
		Kf = mo.Matrix.sparse(Kf_coo.shape[0], Kf_coo.shape[1], Kf_coo.row, Kf_coo.col, Kf_coo.data)

		self.varf = self.solver.variable("varf", self.analyzer.n_var())
		self.solver.constraint("eq", mo.Expr.add(mo.Expr.mul(K, self.varf), self.g), mo.Domain.equalsTo(0.0))
		self.solver.constraint("fr", mo.Expr.mul(Kf, self.varf), mo.Domain.lessThan(0.0))
		self.solver.objective("obj", mo.ObjectiveSense.Minimize, 0)

		zero = np.zeros(self.analyzer.n_var())
		self.lb_con = self.solver.constraint('lb', self.varf, mo.Domain.greaterThan(0))
		self.ub_con = self.solver.constraint('ub', self.varf, mo.Domain.lessThan(0))
		#self.solver.setSolverParam("optimizer", "primalSimplex")
		#self.solver.setLogHandler(sys.stdout)

	def check_stability(self, status):
		[lbind, lbval] = self.analyzer.lobnd(np.array(status))
		[ubind, ubval] = self.analyzer.upbnd(np.array(status))
		lb = np.ones(self.analyzer.n_var(), dtype=float) * -self.inf
		lb[lbind] = lbval

		ub = np.ones(self.analyzer.n_var(), dtype=float) * self.inf
		ub[ubind] = ubval

		self.lb_con.update(-lb)
		self.ub_con.update(-ub)
		self.solver.solve()
		if self.solver.getPrimalSolutionStatus() == mo.SolutionStatus.Optimal:
			return 0
		else:
			return None

class AssemblyGUI(AssemblyCheckerMosek):

	def __init__(self, boundaries = None):
		super(AssemblyGUI, self).__init__(boundaries)
		self.text = ""
		self.update_render = False

	def get_status(self):
		status = []
		for part_id in range(self.assembly.n_part()):
			part = self.assembly.part(part_id)
			obj = ps.get_surface_mesh("part{}".format(part_id))
			if part.fixed:
				status.append(2)
			elif obj.is_enabled():
				status.append(1)
			else:
				status.append(0)
		print(status)
		return status

	def update_status(self, status):
		for part_id in range(self.assembly.n_part()):
			part = self.assembly.part(part_id)
			obj = ps.get_surface_mesh("part{}".format(part_id))
			if status[part_id] == 2:
				part.fixed = True
				obj.set_color((0, 0, 0))
				obj.set_enabled(True)
			elif status[part_id] == 1:
				obj.set_enabled(True)
				part.fixed = False
				obj.set_color(self.part_colors[part_id])
			else:
				part.fixed = False
				obj.set_enabled(False)
			#self.update_render = True

	def interface(self):
		if self.update_render:
			self.render()
			self.update_render = False

		if psim.Button("Check Stability"):
			prob_result = self.check_stability(self.get_status())

			if prob_result != None:
				self.text = "{:.3e}".format(prob_result)
			else:
				self.text = "None"

		psim.SameLine()
		psim.Text(self.text)

	def render(self):
		# blocks
		ps.remove_group("assembly", False)
		assembly_group = ps.create_group("assembly")
		self.part_colors = []
		for part_id in range(self.assembly.n_part()):
			part = self.assembly.part(part_id)
			if part.fixed == True:
				color = (0, 0, 0)
				obj = ps.register_surface_mesh("part{}".format(part_id), part.V, part.F, color=color)
			else:
				obj = ps.register_surface_mesh("part{}".format(part_id), part.V, part.F)

			self.part_colors.append(obj.get_color())
			obj.set_edge_width(1)
			obj.add_to_group(assembly_group)

		assembly_group.set_hide_descendants_from_structure_lists(True)

		# contacts
		if self.contacts != None:
			[V, F] = pyrb.ContactFace.mesh(self.contacts)
			contact = ps.register_surface_mesh("contact mesh", V, F)
			contact.set_enabled(False)
