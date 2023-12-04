# Adjoint_Method_Tutorial

This notebook describes the step by step code for the implementation of the adjoint method outlined by Andrew M. Bradley in this [pdf](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf) using the 'simple example' from the pdf to illustrate the implementation of the adjoint method.

Because the pdf is concise and therefore dense, it can be quite hard, at a glance, to understand all the points relating to the adjoint method. I recommend looking at the youtube channel Machine Learning & Simulation and his excellent explanation on the adjoint method [playlist](https://www.youtube.com/playlist?list=PLISXH-iEM4Jk27AmSvISooRRKH4WtlWKP). The PDF by Dr. Bradley is more general and applies to Differential Algebraic Equations (DAEs) while the YouTube videos is the adjoint method focused on Explicitely represented ODEs (i.e. can be expressed in the form $\dfrac{dy}{dt} = f(y,t)$. Generally, we work with the latter type of ODEs

I wrote this notebook to help me learn how the adjoint methods works as well as bridge the gap between these two sources of information. If you wish to understand why each step is performed in the method please look at the links above.

This was written with Python 3.9+ and jax. No GPU is required to run this notebook. 

Adjoint Method:
1. Forward Solve ODE
2. Backward Solve Adjoint Solution
3. Calculate $\dfrac{dF}{dp}$
4. Update parameters via gradient descent ($\gamma is chosen step size$) : $p_{i+1} = p_i - \gamma \dfrac{dF}{dp}$

With the example used, there is an analytic solution to check the code at each step.

The Optimisation Problem is as follows:

$$
\min \int_0^T x\;\mathrm{d}t \\
 \\
\begin{aligned}
\mathrm{s.t}\quad \dot{x}= bx \\
x(0) - a =0 
\end{aligned}
$$

WE can think of this as finding the function x(t) that gives the smallest area given the ODE by changing the parameters a and b

Here the analytical solution of this ODE is : $ x(t) = ae^{bt}$ so our challenge is to find the combination of a and b such that the area under  $ x(t) = ae^{bt}$ is minimized.

Note that all stages of this method has an analytical method so you can check the results if you want to try this on your own.

We define the following:
- x : vector of state variables (can also be sometimes called y or u)
- t : independent variable (usually time) 
- p : vector representing all parameters
- f : measure of 'goodness' such as MSE
- g : is the relationship between the the state and parameters e.g. initial conditions
- h : is the ODE in implicit form
- F : Overall Objective Function across time T 
- T : final time
So from the problem above we define the following
$$
\begin{align}
F(x,t;p) &= \int_0^T x\;\mathrm{d}t \\
f(x,t;p) &= x \\
p &= [a,b]^T \\
g(x(t = 0),p) &= x(0) - a \\
h(x,\dot{x},p,t) &= \dot{x} - bx \\
t &\in [0,1]
\end{align}
$$

Our Goal is to find $\dfrac{dF}{dp}$ i.e. the gradients/sensitivies to change our parameters to improve our objective function. 
