Note inspired by: https://www.stat.cmu.edu/~cshalizi/402/lectures/11-splines/lecture-11.pdf




# Meaning of Splines


We imagine, that is to say, that we have data points $(x_1, y_1),(x_2, y_2),...,(x_n, y_n)$ and we want to find a function $\hat r(x)$ which is a good approximation to the true conditional expectation or regression function $r(x)$. Note that, for simplicity, we are working with one-dimensional data, but the idea is the same in all possible dimensions.

A natural way to do this, in one dimension, is to minimize the spline objective function:
![spline_objectvie_func](https://user-images.githubusercontent.com/51177049/230335946-0279fb79-c884-4e1d-b1fc-e486f0e94a4a.png)


The first term is just the **mean squared error** of using the curve $m(x)$ to predict $y$.

The second term is the **second derivative** of $m$ with respect to $x$, that is the **curvature** of $m$ at $x$. Recall that the sign of the second derivative says wheter the curvature is concave or convex, but we don’t care about that so we square it. We then **integrate** this over all x to say how curved m is, on **average**.
Finally, we multiply by $\lambda$ and add that to the MSE. This is adding a **penalty** to the MSE criterion — given two functions with the same MSE, we prefer the one with less average curvature.

So, we have a minimization problem and its solution is expressed by:
![spline_objectvie_func_sol](https://user-images.githubusercontent.com/51177049/230336025-ed0cc62d-5218-4114-9844-ad8c0b673bdd.png)

Such solution, that is a curve, is called smoothing spline, or smoothing spline function.

It is possible to prove that we can approximate any well-behaved function arbitrarily closely, given enough pieces (data).



### Meaning of $\lambda$
We can think at the smoothing spline as the function which minimizes the mean squared error, subject to a constraint on the average curvature.

As $\lambda \rightarrow \infty$, having any curvature at all becomes infinitely penalized, and only linear functions are allowed.

On the other hand, as $\lambda \rightarrow 0$, we decide that we don’t care about curvature.
In that case, we can always come up with a function which just interpolates between the data points, an interpolation spline passing exactly through each point. More specifically, of the infinitely many functions which interpolate between those points, we pick the one with the minimum average curvature.

At intermediate values of $\lambda$, $\hat r_{\lambda}$ becomes a function which compromises between having low curvature, and bending to approach all the data points closely (on average). The larger we make λ, the more curvature is penalized. There is a bias-variance trade-off here. As $\lambda$ grows, the spline becomes less sensitive to the data, with lower variance to its predictions but more bias. As $\lambda$ shrinks, so does bias, but variance grows. For consistency, we want to let $\lambda \rightarrow \infty$ as $n \rightarrow \infty$, just as, with kernel smoothing, we let the bandwidth h → 0 while n → ∞.



### Splines in Multiple Dimensions
One approach is to generalize the spline optimization problem so that we penalize the curvature of the spline surface (no longer a curve). The appropriate penalized least-squares objective function to minimize is:
![spline_multi_dim](https://user-images.githubusercontent.com/51177049/230336096-df0f08d5-f9f1-4205-84c1-2fd146b719ef.png)

The solution is called a **thin-plate spline**.

An alternative is use the spline basis functions, that lead to the **tensor product spline**.




### Historical note
![spline](https://user-images.githubusercontent.com/51177049/230336148-dce618c2-b76b-4c89-8a5e-abb9f5bb8e53.png)

The name “spline” actually comes from a simple tool used by craftsmen to draw smooth curves, which was a thin strip of a flexible material like a soft wood. (A few years ago, when the gas company dug up my front yard, the contractors they hired to put the driveway back used a plywood board to give a smooth, outward-curve edge to the new driveway. The “knots” metal stakes which the board was placed between, and the curve of the board was a spline, and they poured concrete to one side of the board, which they left standing until the concrete dried.) Bending such a material takes energy — the stiffer the material, the more energy has to go into bending it through the same shape, and so the straighter the curve it will make between given points. For smoothing splines, using a stiffer material corresponds to increasing $\lambda$.

From [wikipedia](https://en.wikipedia.org/wiki/Thin_plate_spline).
The name _thin plate spline_ refers to a physical analogy involving the bending of a thin sheet of metal. Just as the metal has rigidity, the TPS fit resists bending also, implying a penalty involving the smoothness of the fitted surface. In the physical setting, the deflection is in the $z$ or $y$ coordinates within the plane.
