import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import pi
from numpy import sin, cos, pi


class simulation:
    def __init__(self, dt, time_max):
        self.dt = dt
        self.time_max = time_max
        self.time = np.arange(0.0, time_max, dt)

        # The Wheel
        self.r = .065 / 2  # in m
        self.M = 0.2  # in kg
        self.I = 0.5 * self.M * self.r ** 2  # in kg.m**2

        # The Pendulum
        self.m = .5  # in kg
        self.motor_dc_radius = .024  # in m
        self.chassis_length = .09  # in m
        self.L = self.r + (self.chassis_length / 2)
        self.i = 1 / 3 * self.m * self.L ** 2  # in kg.m**2
        self.g = 9.80665  # in m/s**2

    def simulate_numerically(self, param):
        # The Wheel
        r = self.r
        M = self.M
        I = self.I

        # The Pendulum
        m = self.m
        motor_dc_radius = self.motor_dc_radius
        chassis_length = self.chassis_length
        L = self.L
        i = self.i
        g = self.g

        # ===== Constant Definition =====
        theta = 30 * pi / 180  # in rad
        theta_dot = 0.0
        phi = 0.0
        phi_dot = 0.0
        state_arr = np.array([theta, theta_dot, phi, phi_dot])

        # ===== Controller Gain =====
        kp_th = param[0]
        kp_phi = param[1]
        kd_th = 100
        kd_phi = 10
        # [1000. 39.52953927  315.83883214   75.06586673]
        ki_th = param[2]
        ki_phi = param[3]

        def derivatives(states, t, integral_th=0, integral_phi=0, th_prev_error=0, phi_prev_error=0):
            ds = np.zeros_like(states)
            th = states[0]
            th_dot = states[1]
            phi = states[2]
            phi_dot = states[3]

            # Update integrals for I-gain
            integral_th += th
            integral_phi += phi

            # ===== Controller =====
            F = (kp_th * th + kp_phi * phi
                 + ki_th * integral_th + ki_phi * integral_phi
                 + kd_th * th_dot + kd_phi * phi_dot)
            # Compute wheel and pendulum torque
            tw = F * r  # wheel torque
            tp = L * tw * sin(th) / r  # pendulum torque

            # ===== State Derivative Calculation =====
            denominator = (I * i
                           + I * L ** 2 * m
                           + M * i * r ** 2
                           + M * L ** 2 * m * r ** 2
                           + i * m * r ** 2
                           + L ** 2 * m ** 2 * r ** 2 * sin(th) ** 2)
            ds[0] = th_dot
            ds[1] = ((- tp * I
                      - tp * M * r ** 2
                      - tw * L * m * r * cos(th)
                      - tp * m * r ** 2
                      + I * g * L * m * sin(th)
                      + M * g * L * m * r ** 2 * sin(th)
                      + g * L * m ** 2 * r ** 2 * sin(th)
                      - L ** 2 * m ** 2 * r ** 2 * sin(2 * th) * th_dot ** 2 / 2)
                     / denominator)
            ds[2] = phi_dot
            ds[3] = ((tw * i
                      + tw * L ** 2 * m
                      + tp * L * m * r * cos(th)
                      - g * L ** 2 * m ** 2 * r * sin(2 * th) / 2
                      + i * L * m * r * sin(th) * th_dot ** 2
                      + L ** 3 * m ** 2 * r * sin(th) * th_dot ** 2)
                     / denominator)
            return ds

        # ===== Run the Simulation =====
        system = integrate.odeint(derivatives, state_arr, self.time)
        # Convert F values to a NumPy array
        f_arr = system @ param
        control = f_arr

        return system, control

    def simulate_graphically(self, best_param):
        best_solution_simulated, _ = self.simulate_numerically(best_param)

        # This animation is inspired by Sergey Royz's self-balancing simulation
        ths_best = best_solution_simulated[:, 0]
        phis_best = best_solution_simulated[:, 2]

        wheel_x = phis_best * self.r
        spot_r = 0.7 * self.r
        wheel_spot_x = wheel_x + spot_r * cos(phis_best - pi / 2)
        wheel_spot_y = self.r - spot_r * sin(phis_best - pi / 2)
        mass_x = wheel_x + self.L * cos(ths_best - pi / 2)
        mass_y = self.r - self.L * sin(ths_best - pi / 2)

        limit = .15  # in m
        fig2 = plt.figure(3)
        ax = fig2.add_subplot(111, autoscale_on=False, xlim=(-limit, limit),
                              ylim=(0, limit * 1.5))
        ax.set_aspect('equal')
        ax.grid()
        line, = ax.plot([], [], 'k-', lw=2)
        wheel = plt.Circle((0.0, self.r), self.r, color='black', fill=False, lw=2)
        wheel_spot = plt.Circle((0.0, spot_r), 1 / 4 * self.r, color='red')
        mass = plt.Circle((0.0, 0.0), self.r / 3, color='black')
        time_template = 'time = %.1fs'
        time_text = ax.text(0.05, 0.1, '', transform=ax.transAxes)
        theta_template = 'θ = %.2f°'
        theta_text = ax.text(0.75, 0.1, '', transform=ax.transAxes)
        phi_template = 'distance = %.2f'
        phi_text = ax.text(0.42, 0.1, '', transform=ax.transAxes)

        def init():
            return []

        def animate(i):
            wheel.set_center((wheel_x[i], self.r))
            wheel_spot.set_center((wheel_spot_x[i], wheel_spot_y[i]))
            mass.set_center((mass_x[i], mass_y[i]))
            line.set_data([wheel_x[i], mass_x[i]], [self.r, mass_y[i]])
            time_text.set_text(time_template % (i * self.dt))
            theta_text.set_text(theta_template % (ths_best[i] * 180 / pi))
            phi_text.set_text(phi_template % (phis_best[i] * self.r))
            patches = [line, time_text, theta_text, phi_text, ax.add_patch(wheel), ax.add_patch(wheel_spot),
                       ax.add_patch(mass)]
            return patches

        ani = animation.FuncAnimation(fig2, animate, np.arange(1, len(best_solution_simulated)),
                                      interval=25, blit=True, init_func=init)
        plt.show()

    def simulate_the_solution(self, best_solutions_list, best_costs_list, costs_of_each_iter):
        # ===Convert list to array===
        best_solutions_arr = np.array(best_solutions_list)
        best_costs_arr = np.array(best_costs_list)
        # ===System definition===
        system_temp, _ = self.simulate_numerically(best_solutions_arr[0])
        system_shape = system_temp.shape
        control_shape, _ = system_shape
        system = np.zeros((len(best_solutions_arr), *system_shape))
        control = np.zeros((len(best_solutions_arr), control_shape))
        for i in range(len(best_solutions_arr)):
            system[i], control[i] = self.simulate_numerically(best_solutions_arr[i])
        # ===x and y / th and phi definition===
        ths = np.zeros((len(best_solutions_arr), system_shape[0]))
        phi = np.zeros((len(best_solutions_arr), system_shape[0]))
        ctrl_val = np.zeros(control.shape)
        for i in range(len(best_solutions_arr)):
            ths[i] = system[i, :, 0]
            phi[i] = system[i, :, 2]
            ctrl_val[i] = control[i, :]

        phis = phi
        control_val = ctrl_val
        dt = self.dt
        max_time = self.time_max
        solutions_list = best_solutions_list
        costs_list = best_costs_list

        solutions_arr = np.array(solutions_list)
        costs_arr = np.array(costs_list)
        t = np.arange(0.0, max_time, dt)

        # =====GRAPH OF COSTS OF EACH ITER=====
        cost_values = costs_of_each_iter
        populations = list(range(1, len(cost_values) + 1))
        # populations = [5,10,20]
        plt.figure(5, figsize=(10, 6))
        plt.plot(populations, cost_values, marker='o', linestyle='-')
        plt.title('Cost Function Value for Different Populations')
        plt.xlabel('Population')
        plt.ylabel('Cost Function Value')
        plt.grid(True)
        for i, value in enumerate(cost_values):
            plt.annotate(f'{value:.2f}',
                         (populations[i], value),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=8,
                         color='red')

        # =====GRAPH OF ALL PARAMS=====
        window1 = plt.figure(1)
        sub_window1 = window1.subplots(2)
        window1.subplots_adjust(hspace=0.5)
        window1.suptitle('All Obtained PI Params Compared', fontsize=16)
        sub_window1[0].set_title("Angle of Pendulum")
        sub_window1[0].set_xlabel('time (s)')
        sub_window1[0].set_ylabel('angle (°)')
        sub_window1[1].set_title("Distance of Robot")
        sub_window1[1].set_xlabel('time (s)')
        sub_window1[1].set_ylabel('distance(m)')
        for i in range(2):
            sub_window1[i].set_xlim([0, max_time])
            sub_window1[i].grid()
        n_costs = len(costs_arr)
        for i in range(n_costs):
            sub_window1[0].plot(t, ths[i] * 180 / pi)
            sub_window1[1].plot(t, (phis[i] * self.r), label=f'Cost: {costs_arr[i]:.2f}')
        sub_window1[1].legend(loc='lower right')
        plt.tight_layout()

        # =====GRAPH OF CONTROL SIGNAL (u)=====
        window_ctrl = plt.figure(6)
        sub_window_ctrl = window_ctrl.add_subplot(111)
        window_ctrl.subplots_adjust(hspace=0.5)
        window_ctrl.suptitle('Control Signal Compared', fontsize=16)
        sub_window_ctrl.set_title("Control Signal (u)")
        sub_window_ctrl.set_xlabel('time (s)')
        sub_window_ctrl.set_ylabel('Control Value (Nm)')
        sub_window_ctrl.set_xlim([0, max_time])
        sub_window_ctrl.grid()
        for i in range(n_costs):
            # sub_window_ctrl.plot(t, control_val[i], label=f'Solution {i + 1}')
            sub_window_ctrl.plot(t, control_val[i] * self.r, label=f'Cost: {costs_arr[i]:.2f}')
        sub_window_ctrl.legend(loc='lower right')
        plt.tight_layout()

        print("\n========== GRAPH INFO ==========")
        for i in range(n_costs):
            def calculate_rise_time(output, time):
                start_value = output[0] - 0
                rise_start = start_value * 0.9
                rise_end = start_value * 0.1

                try:
                    start_time = next(t for t, v in zip(time, output) if v <= rise_start)
                except StopIteration:
                    print("No start time found for rise_start:", rise_start)
                    start_time = time[0]  # Default to the initial time

                try:
                    end_time = next(t for t, v in zip(time, output) if v <= rise_end)
                except StopIteration:
                    print("No end time found for rise_end:", rise_end)
                    end_time = time[-1]  # Default to the final time

                rise_time = end_time - start_time
                # Filter the Ths and time arrays for overshoot
                rising_indices = np.where(time >= end_time)[0]
                filtered_output = output[rising_indices]
                filtered_t = time[rising_indices]
                return rise_time, filtered_output, filtered_t

            def calculate_settling_time(output, time, tolerance):
                start_value = output[0] - 0
                upper_bound = tolerance * start_value
                lower_bound = -1 * upper_bound
                for i in range(len(output) - 1, -1, -1):
                    if not (lower_bound <= output[i] <= upper_bound):
                        return time[i] + dt
                return time[0]

            def calculate_overshoot(output):
                yss = output[0]
                final_value = yss
                yp = max(abs(filtered_Ths))
                amplitude = max_value = yp
                overshoot_percentage = (max_value / final_value) * 100
                return overshoot_percentage, amplitude

            print("==== COST : ", round(costs_arr[i], 2), " ====")
            print("Parameters: ", solutions_arr[i])
            rise_times, filtered_Ths, filtered_t = calculate_rise_time(ths[i], t)
            settling_times = calculate_settling_time(ths[i], t, 0.05)
            overshoot, amplitude = calculate_overshoot(ths[i])
            print("== ANGLE SPECIFICATION ==")
            print("Rise Time:", rise_times, "s")
            print("Settling Time:", settling_times, "s")
            print("Overshoot:", abs(int(overshoot)), "%")
            print("Steady-state Error:", ths[i, -1])
            print("Amplitude:", round((amplitude * 180 / pi), 2), "°")
            print("")

        lowest_cost_id = np.argmin(best_costs_arr)
        highest_cost_id = np.argmax(best_costs_arr)
        best_solution = best_solutions_arr[lowest_cost_id]
        worst_solution = best_solutions_arr[highest_cost_id]
        self.simulate_graphically(best_solution, worst_solution)