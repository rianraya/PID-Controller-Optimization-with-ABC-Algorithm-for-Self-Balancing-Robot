import numpy as np
from Cost import objective as lqr_cost
import math
import Simulation as Simulation


class ABC:
    def __init__(self, n_employed_bees, n_scout_bees, n_onlooker_bees, lb, ub, limit, max_iter, dt, time_max):
        self.n_employed_bees = int(n_employed_bees)
        self.n_onlooker_bees = int(n_onlooker_bees)
        self.n_scout_bees = int(n_scout_bees)

        self.dim = 4  # dimension of solution (Kp_th, Kp_phi, Ki_th, Ki_phi)
        self.ub_th = ub[0]  # upper bound for the rod
        self.lb_th = lb[0]  # lower bound for the rod
        self.ub_phi = ub[1]  # upper bound for the wheel
        self.lb_phi = lb[1]  # lower bound for the wheel

        self.food_sources_arr = np.zeros((self.n_employed_bees, self.dim))  # define an array for the solution
        self.costs_arr = np.zeros(self.n_employed_bees)  # define an array for cost values
        self.trials_arr = np.zeros(self.n_employed_bees)  # define an array for trial counter (used by scout phase)
        self.limit = limit
        self.max_iter = max_iter
        self.best_costs_list = []
        self.best_solutions_list = []

        self.global_best_food_source = np.zeros(self.dim)  # define the best solution among the other locals
        self.global_best_cost = np.inf  # define the best cost among the other locals

        self.time_max = time_max  # define how long the simulation going to take
        self.dt = dt  # define the time sampling rate

    def numerically_simulate(self, param):  # This method simulate the obtained Kp and Ki parameters
        sim = Simulation.simulation(self.dt, self.time_max)
        system, control = sim.simulate_numerically(param)
        return system, control

    def initial_solution(self):  # This method randomly generate the initial solution along with its cost
        th_food_source_arr = np.random.uniform(self.lb_th, self.ub_th, (self.n_employed_bees, int(self.dim / 2)))
        phi_food_source_arr = np.random.uniform(self.lb_phi, self.ub_phi, (self.n_employed_bees, int(self.dim / 2)))
        kp_th = th_food_source_arr[:, 0]
        ki_th = th_food_source_arr[:, 1]
        kp_phi = phi_food_source_arr[:, 0]
        ki_phi = phi_food_source_arr[:, 1]

        self.food_sources_arr = np.array([kp_th, kp_phi, ki_th, ki_phi]).T
        for i in range(self.n_employed_bees):
            system, _ = self.numerically_simulate(self.food_sources_arr[i])
            _, control = self.numerically_simulate(self.food_sources_arr[i])
            cost = lqr_cost(system, control)
            self.costs_arr[i] = cost

    def employed_bees_phase(self):  # This method run the employed phase
        for i in range(self.n_employed_bees):

            # === New Solution Creation ===
            a_food_source = np.copy(self.food_sources_arr[i])
            d = np.random.randint(0, self.dim)  # d means which dimension to modify
            b = np.random.choice([idx for idx in range(self.n_employed_bees) if
                                  idx != i])  # b means other bee/solution to collaborate with
            phi = np.random.uniform(-1, 1)
            if d == 0 or d == 2:  # dimension to be changed is th
                new_source_coordinate = np.clip(self.food_sources_arr[i][d] + phi *
                                                (self.food_sources_arr[i][d] - self.food_sources_arr[b][d]), self.lb_th,
                                                self.ub_th)
            elif d == 1 or d == 3:  # dimension to be changed is phi
                new_source_coordinate = np.clip(self.food_sources_arr[i][d] + phi *
                                                (self.food_sources_arr[i][d] - self.food_sources_arr[b][d]),
                                                self.lb_phi,
                                                self.ub_phi)
            a_food_source[d] = new_source_coordinate

            # === Greedy Selection ===
            new_system, _ = self.numerically_simulate(a_food_source)
            _, new_control = self.numerically_simulate(a_food_source)
            current_system, _ = self.numerically_simulate(self.food_sources_arr[i])
            _, current_control = self.numerically_simulate(self.food_sources_arr[i])
            cost_of_new_sources = lqr_cost(new_system, new_control)
            cost_of_current_sources = lqr_cost(current_system, current_control)
            if cost_of_new_sources < cost_of_current_sources:
                self.trials_arr[i] = 0
                self.food_sources_arr[i] = np.copy(a_food_source)
                self.costs_arr[i] = np.copy(cost_of_new_sources)
            elif cost_of_new_sources >= cost_of_current_sources:
                self.trials_arr[i] += 1

    def onlooker_bees_phase(self):
        # === R Value Creation ===
        # = Establishing Fitness & Probability =
        fitness = np.zeros(self.n_employed_bees)
        probs = np.zeros(self.n_employed_bees)
        for m in range(len(self.costs_arr)):
            if self.costs_arr[m] >= 0:
                fitness[m] = 1 / (self.costs_arr[m] + 1)
            if self.costs_arr[m] < 0:
                fitness[m] = 1 + abs(self.costs_arr[m])
        total_fitness = np.sum(fitness)
        for n in range(len(self.food_sources_arr)):
            probs[n] = fitness[n] / total_fitness
        # = Choosing R value =
        r = np.median(probs)

        # === New Solution Creation (with IF condition) ===
        fs = 0
        i = 0
        while i < self.n_onlooker_bees:
            if r < probs[fs]:
                a_food_source = np.copy(self.food_sources_arr[fs])
                d = np.random.randint(0, self.dim)
                b = np.random.choice([idx for idx in range(self.n_employed_bees) if idx != fs])
                phi = np.random.uniform(-1, 1)
                # = Creating & Assigning New Food Source =
                if d == 0 or d == 2:  # dimension to be changed is th
                    new_source_coordinate = np.clip(self.food_sources_arr[fs][d] + phi *
                                                    (self.food_sources_arr[fs][d] - self.food_sources_arr[b][d]),
                                                    self.lb_th,
                                                    self.ub_th)
                elif d == 1 or d == 3:  # dimension to be changed is phi
                    new_source_coordinate = np.clip(self.food_sources_arr[fs][d] + phi *
                                                    (self.food_sources_arr[fs][d] - self.food_sources_arr[b][d]),
                                                    self.lb_phi, self.ub_phi)
                a_food_source[d] = np.copy(new_source_coordinate)

                # === Greedy Selection ===
                new_system, _ = self.numerically_simulate(a_food_source)
                _, new_control = self.numerically_simulate(a_food_source)
                current_system, _ = self.numerically_simulate(self.food_sources_arr[fs])
                _, current_control = self.numerically_simulate(self.food_sources_arr[fs])
                cost_of_new_sources = lqr_cost(new_system, new_control)
                cost_of_current_sources = lqr_cost(current_system, current_control)
                if cost_of_new_sources < cost_of_current_sources:
                    self.trials_arr[fs] = 0
                    self.food_sources_arr[fs] = np.copy(a_food_source)
                    self.costs_arr[fs] = np.copy(cost_of_new_sources)
                elif cost_of_new_sources >= cost_of_current_sources:
                    self.trials_arr[fs] += 1
                i += 1
                fs += 1
            elif r >= probs[fs]:
                fs += 1
            if fs >= len(probs):
                fs = 0

    def scout_bees_phase(self):
        for i in range(self.n_scout_bees):
            # === Trial & Limit Comparison ===
            if self.trials_arr[i] > self.limit:
                # === New Solution Creation ===
                th_food_source_arr = np.clip(np.random.uniform(self.lb_th, self.ub_th, (1, int(self.dim / 2))),
                                             self.lb_th, self.ub_th)
                phi_food_source_arr = np.clip(np.random.uniform(self.lb_phi, self.ub_phi, (1, int(self.dim / 2))),
                                              self.lb_phi, self.ub_phi)
                kp_th = th_food_source_arr[:, 0]
                ki_th = th_food_source_arr[:, 1]
                kp_phi = phi_food_source_arr[:, 0]
                ki_phi = phi_food_source_arr[:, 1]
                new_food_source = np.array([kp_th, kp_phi, ki_th, ki_phi]).T
                self.food_sources_arr[i] = np.copy(new_food_source)
                system, _ = self.numerically_simulate(self.food_sources_arr[i])
                _, control = self.numerically_simulate(self.food_sources_arr[i])
                cost = lqr_cost(system, control)
                self.costs_arr[i] = np.copy(cost)
                self.trials_arr[i] = 0

    def nan_inf_fixer(self):
        for i in range(len(self.costs_arr)):
            while math.isnan(self.costs_arr[i]) or math.isinf(self.costs_arr[i]):
                th_food_source_arr = np.clip(np.random.uniform(self.lb_th, self.ub_th, (1, int(self.dim / 2))),
                                             self.lb_th, self.ub_th)
                phi_food_source_arr = np.clip(np.random.uniform(self.lb_phi, self.ub_phi, (1, int(self.dim / 2))),
                                              self.lb_phi, self.ub_phi)
                kp_th = th_food_source_arr[:, 0]
                ki_th = th_food_source_arr[:, 1]
                kp_phi = phi_food_source_arr[:, 0]
                ki_phi = phi_food_source_arr[:, 1]
                new_food_source = np.array([kp_th, kp_phi, ki_th, ki_phi]).T
                self.food_sources_arr[i] = np.copy(new_food_source)
                system, _ = self.numerically_simulate(self.food_sources_arr[i])
                _, control = self.numerically_simulate(self.food_sources_arr[i])
                cost = lqr_cost(system, control)
                self.costs_arr[i] = np.copy(cost)
                self.trials_arr[i] = 0

    def optimize(self):
        cost_values = []

        # === Randomize Initial Solution ===
        self.initial_solution()
        # === nan / inf Fixer ===
        self.nan_inf_fixer()

        for i in range(self.max_iter):
            print("\n=== ITERATION", i + 1, "===")

            # ===  Employed Phase ===
            self.employed_bees_phase()
            print("Employed Phase Done")

            # === Onlooker Phase ===
            self.onlooker_bees_phase()
            print("Onlooker Phase Done")

            # === Scout Phase ===
            self.scout_bees_phase()
            print("Scout Phase Done")

            # Assign the current index, solution, and cost as the local index, solution, and cost
            local_best_id = np.argmin(self.costs_arr)  # The index
            local_best_food_source = self.food_sources_arr[local_best_id]  # The solution
            local_best_cost = self.costs_arr[local_best_id]  # The cost

            # Adding cost of each iter to the list for plotting purpose 
            current_cost = local_best_cost
            cost_values.append(current_cost)

            if local_best_cost < self.global_best_cost:
                self.global_best_cost = np.copy(local_best_cost)
                self.global_best_food_source = np.copy(local_best_food_source)
                self.best_solutions_list.append(self.global_best_food_source)
                self.best_costs_list.append(self.global_best_cost)

            print("Best Solution: ", self.global_best_food_source)
            print("Best Cost: ", self.global_best_cost)
        print("\n=== ABC FINAL RESULT ===")
        print("Best Solution:", self.global_best_food_source)
        print("Best Cost:", self.global_best_cost)

        return self.best_solutions_list, self.best_costs_list, cost_values