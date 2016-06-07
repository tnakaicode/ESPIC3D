.. role:: raw-math(raw)
    :format: latex html

esPic
=======

Electrostatic particle-in-cell code.

Example: 2D Electrostatic, no particles
------------------

The 2d electrostatic example solves the 2d Laplace equation in the rectangle X0 < x < X0 + LX, Y0 < y < Y0 + LY, with (X0,Y0) = (1.0,2.0) and (LX,LY) = (1.25,2.3).

The computational grid is 26 x 31, and the Dirichlet boundary conditions are V(0,y) = V(x,0) = V(x,LY) = 0 and V(LX,y) = 2.0 * sin(pi * y / LY). 

To run the 2d example, run the command

.. code-block:: bash

    $ python3 examples/ex_2d.py

The first plot should look like this, which shows 3 absolute error tolerances: 1.0, 0.5, and 0.001:

.. image:: images/ex_2d_one.png
    :align: center

The second plot should look like this, which shows just the 0.001 absolute error tolerance.

.. image:: images/ex_2d_many.png
    :align: center

Tests
------------------

To run the tests, run the script tests/testAll.sh.
