- In this program I made a class (ABC) with 7 methods and a class constructor. 

- The methods are :

1. numerically_simulate()
It takes the obtained parameters (Kp and Ki) and simulate them using the Simulation.py program. It returns the *state matrix of th, th_dot, ph, ph_dot (Coloumns: th, th_dot, ph, ph_dot; Row: time).
*State matrix : a matrix that represents the state of a system at any given time

2. initial_solution()
It generates an initial solution randomly.

3. employed_bees_phase()
It does the employed phase of ABC.

4. onlooker_bees_phase()
It does the onlooker phase of ABC.

5. scout_bees_phase()
It does the scout phase of ABC.

6. nan_inf_fixer()
Sometimes the initial solution could generate a nan or a inf cost value, therefore this function is re-generating the intial solution randomly until the cost value isn't nan or inf.

7. optimize()
This method works as a main, where all the other methods are called within this method. 
I break optimize() method into 3 sections (initial, main, and greedy selection) to make it easier to explain.

=====INITIAL SECTION=====
1. Generate initial solution by randomizing value of each parameter.
2. Sometimes when calculate the cost value of the initial randomize value, it could be nan or inf. Thus the function of nan_inf_fixer is run.

=====MAIN SECTION=====
*Contains the main algorithm of Artificial Bee Colony which are employed, onlooker, and scout phase.

=====GREEDY SELECTION SECTION=====
1. Assign the obtained solution from the current iteration unto the local_best variable to make it easy for the greedy selection later.
2. Additionally, add the current cost value of the obtained solution into a list to plot the cost function of every iteration.
3. With the "if" function, compare the recent cost value with the global cost function. And keep which ever is the lowest.
4. Lastly, display the solution and the cost value to update the information on the best solution and its cost value.