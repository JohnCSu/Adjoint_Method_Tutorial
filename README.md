# Adjoint_Method_Tutorial

This notebook describes the step by step code for the implementation of the adjoint method outlined by Andrew M. Bradley in this [pdf](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf) using the 'simple example' from the pdf to illustrate the implementation of the adjoint method.

Because the pdf is concise and therefore dense, it can be quite hard, at a glance, to understand all the points relating to the adjoint method. I recommend looking at the youtube channel Machine Learning & Simulation and his excellent explanation on the adjoint method [playlist](https://www.youtube.com/playlist?list=PLISXH-iEM4Jk27AmSvISooRRKH4WtlWKP). The PDF by Dr. Bradley is more general and applies to Differential Algebraic Equations (DAEs) while the YouTube videos is the adjoint method focused on Explicitely represented ODEs (i.e. can be expressed in the form $\dfrac{dy}{dt} = f(y,t)$. Generally, we work with the latter type of ODEs

I wrote this notebook to help me learn how the adjoint methods works as well as bridge the gap between these two sources of information.

This was written with Python 3.9+ and jax. No GPU is required to run this notebook.

