import time

import numpy as np
import pydot
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from IPython.display import SVG, display

from pydrake.symbolic import Variable
from pydrake.systems.primitives import SymbolicVectorSystem, LogVectorOutput
from pydrake.systems.analysis import Simulator, ExtractSimulatorConfig, ApplySimulatorConfig
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.geometry import StartMeshcat, Sphere, Box, Cylinder
from pydrake.math import RigidTransform, RotationMatrix

def create_system(x_0, a):
    def plot():
        log = logger.FindLog(context)
        plt.plot(log.sample_times(), log.data().transpose())
        plt.xlabel('t')
        plt.ylabel('x(t)')

    # DiagramBuilder - like Simulink but without GUI
    builder = DiagramBuilder()

    x = Variable('x')
    continuous_vector_system = SymbolicVectorSystem(state=[x], dynamics=[a * x], output=[x])
    system = builder.AddSystem(continuous_vector_system)
    logger= LogVectorOutput(system.get_output_port(0), builder)

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    context.SetContinuousState([x_0])

    return diagram, context, plot

def analytical_solution(x_0, a, t):
    z = np.einsum('i, j -> ij', a, t)
    return x_0*np.exp(z)

x_0 = 2
a_space = np.linspace(-1,1,100) # TODO: zmień na przedział od -1 do 1 i 100 próbek, można skorzystać numpy.linspace
t = np.linspace(0,3, 100) # TODO: zmień na przedział (0, 3) i 100 próbek
X_t = analytical_solution(x_0, a_space, t)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for i in range(X_t.shape[0]):
    ax.plot(a_space[i] * np.ones_like(t), t, X_t[i])
ax.view_init(elev=10., azim=140)
ax.set_xlabel('a')
ax.set_ylabel('t')
ax.set_zlabel('x(t)')
plt.show()

x_0 = 5.0
a_lst = [-0.5, 0, 0.5] # TODO: ustaw 3 wartości uzyskując 3 rózne jakościowo przebiegi
T = 10.0 # simulation time [seconds]
for a in a_lst:
    diagram, context, plot = create_system(x_0=x_0, a=a)
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(T)
    plot()
plt.legend(list(map(lambda a: f'a={a}', a_lst)))
plt.title('Numerical solution form simulator x(t)')
plt.show()

t = np.linspace(0, T, 1000)
X_t = analytical_solution(x_0, a_lst, t) # |a_lst| x len(t) array of x_t for different parameter a
t_broadcasted = np.broadcast_to(t, X_t.shape) # broadcast t to X_t shape for plotting
plt.plot(t_broadcasted.T, X_t.T)
plt.legend(list(map(lambda a: f'a={a}', a_lst)))
plt.title('Analytical solution x(t)')
plt.show()

#1_3
# Symulacja układu z pobudzeniem

# x_dot = ax + u
class Linear1DWithInput(LeafSystem):
    def __init__(self, a):
        LeafSystem.__init__(self)

        state_index = self.DeclareContinuousState(1)  # One state variable.
        self.input = self.DeclareVectorInputPort("u", 1)
        self.a = a
        self.DeclareStateOutputPort("y", state_index)  # One output: y=x.

    def DoCalcTimeDerivatives(self, context, derivatives):
        x = context.get_continuous_state_vector().GetAtIndex(0)
        u = self.input.Eval(context)
        xdot = self.a * x + u
        derivatives.get_mutable_vector().SetAtIndex(0, xdot)


def create_system_with_input(x_0, a, controller):
    def plot():
        log = logger.FindLog(context)
        plt.plot(log.sample_times(), log.data().transpose())
        plt.xlabel('t')
        plt.ylabel('x(t)')

    builder = DiagramBuilder()

    system = builder.AddSystem(Linear1DWithInput(a))
    logger = LogVectorOutput(system.get_output_port(), builder)

    controller = builder.AddSystem(controller)

    builder.Connect(controller.get_output_port(), system.get_input_port())
    builder.Connect(system.get_output_port(), controller.get_input_port())

    diagram = builder.Build()

    context = diagram.CreateDefaultContext()
    context.SetContinuousState(np.array([x_0]))
    return diagram, context, plot, system

x_0 = 5.0
a = -0.2
T = 100

class DummyController(LeafSystem):
    def __init__(self, a):
        LeafSystem.__init__(self)

        self.a = a
        self.input = self.DeclareVectorInputPort("x", 1)
        self.output = self.DeclareVectorOutputPort("y", 1, self.control)

    def control(self, context, output):
        x = self.input.Eval(context)
        t = context.get_time()
        u = 0.0
        output.get_mutable_value()[:] = u

controller = DummyController(a)
diagram, context, plot, system = create_system_with_input(x_0, a, controller)

class FeedbackCancellationController(LeafSystem):
    def __init__(self, a):
        LeafSystem.__init__(self)

        self.a = a
        self.input = self.DeclareVectorInputPort("x", 1)
        self.output = self.DeclareVectorOutputPort("y", 1, self.control)

    def control(self, context, output):
        x = self.input.Eval(context)
        t = context.get_time()
        u = -self.a * x # TODO: Ustaw odpowiednią wartość sterowania
        output.get_mutable_value()[:] = u

controller = FeedbackCancellationController(a)
diagram, context, plot, system = create_system_with_input(x_0, a, controller)
simulator = Simulator(diagram, context)
simulator.AdvanceTo(T)
plot()
diagram.set_name("1D with dynamics cancellation")
display(SVG(pydot.graph_from_dot_data(
    diagram.GetGraphvizString(max_depth=2))[0].create_svg()))

class PeriodicOutputController(LeafSystem):
    def __init__(self, a):
        LeafSystem.__init__(self)

        self.a = a
        self.input = self.DeclareVectorInputPort("x", 1)
        self.output = self.DeclareVectorOutputPort("y", 1, self.control)

    def control(self, context, output):
        x = self.input.Eval(context)
        t = context.get_time()
        u = -self.a * x + np.cos(t) # TODO: Ustaw odpowiednią wartość sterowania
        output.get_mutable_value()[:] = u

controller = PeriodicOutputController(a)
diagram, context, plot, system = create_system_with_input(x_0, a, controller)
simulator = Simulator(diagram, context)
simulator.AdvanceTo(T)
plot()
plt.show()

class Controller(LeafSystem):
    def __init__(self, a, u_lim):
        LeafSystem.__init__(self)

        self.a = a
        self.u_lim = u_lim
        self.input = self.DeclareVectorInputPort("x", 1)
        self.output = self.DeclareVectorOutputPort("y", 1, self.control)

    def control(self, context, output):
        x = self.input.Eval(context)
        t = context.get_time()
        u = -self.a * x + np.cos(t)# Ustaw takie samo sterowanie jak w PeriodicOutputController

        # NIE ZMIENIAĆ
        u = np.clip(u, -self.u_lim, self.u_lim)
        output.get_mutable_value()[:] = u

x_0 = 5.0
u_lim_lst = np.linspace(0.1, 2, 4)
for u_lim in u_lim_lst:
    controller = Controller(a, u_lim)
    diagram, context, plot, system = create_system_with_input(x_0, a, controller)
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(T)
    plot()
plt.legend(list(map(lambda u_lim: f'|u| <= {u_lim}', u_lim_lst)))
plt.show()

x_0 = 2.0
u_lim_lst = np.linspace(0.1, 2, 4)
for u_lim in u_lim_lst:
    controller = Controller(a, u_lim)
    diagram, context, plot, system = create_system_with_input(x_0, a, controller)
    simulator = Simulator(diagram, context)
    simulator.AdvanceTo(T)
    plot()
plt.legend(list(map(lambda u_lim: f'|u| <= {u_lim:.2f}', u_lim_lst)))
plt.show()