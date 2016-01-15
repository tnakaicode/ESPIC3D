.. role:: raw-math(raw)
    :format: latex html

emSolve
=======

Electrostatic and electromagnetic field solver.

Example: 2D Electrostatic Example
------------------

Run the 2d example with the command python3 examples/ex_2d.py

This example solves the 2d Laplace equation in the rectangle 0 < x < LX, 0 < y < LY, with boundary conditions V(0,y) = V(x,0) = V(x,LY) = 0 and V(LX,y) = 2.0 * sin(pi * y / LY).

The resulting plot should look like this:

.. image:: images/ex_2d.png
    :align: center

.. code-block:: python

    >>> example code

Electromagnetic Example
------------------

.. code-block:: python

    >>> example code

Tests
------------------

To run the tests, run the script tests/testAll.sh.
