- In this program I made a class (simulation) with 3 methods and a class constructor.

- The methods are :
1. simulate_numerically()
It takes the ABC-generated parameters and simulate those parameters (Kp and Ki) to produce 2 matrices of the state and the control.

2. simualte_graphically()
It takes the best ABC-generated parameters and uses those parameters to simulate/animate the self-balancing robot motion. The animation later can be use as a prediction of the robot's movement in the physical/real world.

3. simulate_the_solution()
This method simply Simulates the system for each solution to get data like the pendulum angle, robot position, and control signals. Then it creates graphs to show the cost function changes over populations, the pendulum's angle and robot's position for all solutions over time, and the control signals for each solution. Not only that, it also calculate the important system character such as rise time, settling time, overshoot, steady-state error, and amplitude. Lastly, this method run the graphical simulation using the best solution obtained by ABC.