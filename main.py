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
plt.savefig("plot.png")
plt.clf()


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
plt.savefig("plot2.png")
plt.clf()

t = np.linspace(0, T, 1000)
X_t = analytical_solution(x_0, a_lst, t) # |a_lst| x len(t) array of x_t for different parameter a
t_broadcasted = np.broadcast_to(t, X_t.shape) # broadcast t to X_t shape for plotting
plt.plot(t_broadcasted.T, X_t.T)
plt.legend(list(map(lambda a: f'a={a}', a_lst)))
plt.title('Analytical solution x(t)')
plt.savefig("plot3.png")