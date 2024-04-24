import numpy as np
from AssemblyEnv.py_rigidblock import Assembly, Part
import AssemblyEnv.py_rigidblock as pyrb
import pickle
import polyscope as ps
import polyscope.imgui as psim
import cvxpy as cp
import gurobipy as gp
from gurobipy import GRB
from scipy.sparse import csc_matrix
class AssemblyChecker:
	def __init__(self, boundaries=None):
		self.assembly = None
		self.analyzer = None
		self.contacts = None
		self.solver = gp.Model()
		self.solver.setParam('LogFile', "")
		self.solver.setParam('LogToConsole', 0)
		self.solver.setParam('OutputFlag', 0)
		self.part_colors = []

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
		#with open(path, 'rb') as fp:
			#boundries = pickle.load(fp)
			#elf.init(boundries)

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

			self.varf = self.solver.addMVar( shape=self.analyzer.n_var(), vtype = GRB.CONTINUOUS, name = "varf")
			self.solver.addConstr(self.K @ self.varf + self.g == 0)
			self.solver.addConstr(self.Kf @ self.varf <= 0)
			self.solver.setObjective(0, GRB.MINIMIZE)

	def reset(self):
		self.solver.reset()

	def close(self):
		self.reset()

	def check_stability_gurobi(self, status):
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

	def check_stability(self, status):
		[loind, lobnd] = self.analyzer.lobnd(np.array(status))
		[upind, upbnd] = self.analyzer.upbnd(np.array(status))
		varf = cp.Variable(self.analyzer.n_var())
		#P = self.analyzer.obj_ceoff()
		prob = cp.Problem(cp.Minimize(0), # cp.Minimize((1 / 2) * cp.quad_form(varf, P)),
	                  [self.K @ varf + self.g == 0,
	                   self.Kf @ varf <= 0,
	                   lobnd - varf[loind] <= 0,
	                   varf[upind] - upbnd <= 0])
		try:
			prob.solve(verbose = 0, solver = "MOSEK")
		except cp.SolverError:
			return None
		if prob.status != 'optimal':
			return None
		return prob.value

class AssemblyGUI(AssemblyChecker):

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
