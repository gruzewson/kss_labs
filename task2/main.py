from functools import partial

import numpy as np

from pydrake.all import MathematicalProgram, IpoptSolver
from pydrake.math import ge, le, eq
from pydrake.symbolic import ToLatex
from IPython.display import display, Latex

def canon_dynamics(state, c, g):
    # State consist of position, q=(x, y), and velocity, q_dot=(x_dot, y_dot)
    x, y, x_dot, y_dot = state

    v_sqr = x_dot * x_dot + y_dot * y_dot
    v = v_sqr ** 0.5
    x_dot_dot = -c * (x_dot * v)
    y_dot_dot = -c * (y_dot * v) - g
    state_dot = np.array([x_dot, y_dot, x_dot_dot, y_dot_dot])
    return state_dot

def initialize_state_trajectory(N):
    # Initialize
    traj_initial = np.zeros((N, 4), dtype=np.float32)
    traj_initial[:, 0] = np.linspace(0, 3, N)
    traj_initial[:, 1] = np.linspace(0, 1, N)
    traj_initial[:-1, 2] = traj_initial[1:, 0] - traj_initial[:-1, 0]
    traj_initial[:-1, 3] = traj_initial[1:, 1] - traj_initial[:-1, 1]
    return traj_initial

c_no_drag = 0.0
c_drag = 0.4
g = 9.81
canon_dynamics_no_drag = partial(canon_dynamics, c=c_no_drag, g=g)
canon_dynamics_with_drag = partial(canon_dynamics, c=c_drag, g=g)

# Test to check the dynamics function
state_test = np.ones(4)
c_test = 0.7
state_test_dot = np.array([1., 1., -0.98994949, -10.79994949])
np.testing.assert_allclose(canon_dynamics(state_test, c_test, g), state_test_dot)

def solve_optimization(dynamics=canon_dynamics_no_drag, traj_initial=None, N=50):
    prog = MathematicalProgram()

    N = N
    t = prog.NewContinuousVariables(1, 't')
    prog.AddConstraint(ge(t, np.array([0.01])))
    prog.SetInitialGuess(t, np.float32([1.0]))

    h = t / N

    traj = prog.NewContinuousVariables(N, 4, 'traj') # x, y, x_dot, y_dot trajectory
    if traj_initial is None:
        traj_initial = initialize_state_trajectory(N)
    else:
        traj_initial=traj_initial
    prog.SetInitialGuess(traj, traj_initial)

    # TODO: Add constraint on initial position (first two elements of traj[0])
    prog.AddConstraint(eq(traj[0, :2], np.array([0., 0.])))

    # TODO: Add constraint on final position (first two elements of traj[0])
    prog.AddConstraint(eq(traj[-1, :2], np.array([3., 1.])))


    # TODO: Initialize derivative of trajectory
    # Hint: dynamics takes state (x, y, x_dot, y_dot) and returns derivative
    #       initialize traj_dot so traj_dot[i] contains derivative at i'th step in trajectory
    traj_dot = [dynamics(traj_initial[i]) for i in range(N)]

    # TODO: Add dynamics constraint
    # Hint: Check 3.2 @ https://epubs.siam.org/doi/epdf/10.1137/16M1062569
    # Hint: x_k is state at k=i, f_k is state_dot at k=i (dynamics at x_k)
    for i in range(N - 1):
        state_dot = dynamics(traj[i])  # Calculate the derivative at each step
        prog.AddConstraint(eq(traj[i + 1], traj[i] + h * state_dot))

    # TODO: Add cost on squared initial speed
    prog.AddCost(traj[0, 2]**2 + traj[0, 3]**2)

    solver = IpoptSolver()
    result = solver.Solve(prog)
    assert result.is_success()
    return result.GetSolution(traj), result.GetSolution(t)

from matplotlib import pyplot as plt

traj_sol, t_sol = solve_optimization(dynamics=canon_dynamics_no_drag)
xy = traj_sol[:, :2]
plt.plot(xy[:, 0], xy[:, 1])
plt.plot(xy[:, 0], xy[:, 1], 'o');
plt.show()

plt.plot(traj_sol[:, 2])
plt.title('Horizontal velocity $\dot{x}$');
plt.show()

traj_sol, t_sol = solve_optimization(dynamics=canon_dynamics_with_drag, traj_initial=traj_sol)
xy = traj_sol[:, :2]
plt.plot(xy[:, 0], xy[:, 1])
plt.plot(xy[:, 0], xy[:, 1], 'o');
plt.show()

plt.plot(traj_sol[:, 2])
plt.title('Horizontal velocity $\dot{x}$');
plt.show()

# Number of steps in the trajectory
N = 50

# Initialize the trajectory
traj_initial = initialize_state_trajectory(N)

# Set a reasonable initial velocity
# Estimate based on the target point (3, 1) and gravity for a parabolic arc
initial_velocity_x = 10.0  # Example value in m/s
initial_velocity_y = 15.0  # Example value in m/s

# Apply the initial velocity to the first point in traj_initial
traj_initial[0, 2] = initial_velocity_x
traj_initial[0, 3] = initial_velocity_y

# Solve the optimization with air drag and the initialized trajectory
traj_sol, t_sol = solve_optimization(dynamics=canon_dynamics_with_drag, traj_initial=traj_initial)

# Extract the x and y positions for plotting
xy = traj_sol[:, :2]

# Plot the trajectory
plt.plot(xy[:, 0], xy[:, 1], label="Trajectory with Drag")
plt.plot(xy[:, 0], xy[:, 1], 'o')
plt.xlabel("x position (m)")
plt.ylabel("y position (m)")
plt.title("Projectile Trajectory with Air Drag")
plt.legend()
plt.show()
