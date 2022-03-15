# Log 1

## Week 3

It is easy to find a solution for a rhs of the form `b_vec = [b ... b]`.
Since each vector has a complement, we can simply choose `b` complement 
pairs to build the solution. As a result we choose to simplify the problem. 
As a result we wish to decompose the problem. For wach potential right-hand side,
we can find a `l>=0` such that `b = l*ones + u`, where `ones` is the vector of 
ones of adequate size. 

We need to have that `u` is minimal feasible solution with regards to the l-infinity 
norm, which implies that `u - ones` should not be feasible. 

We will therefore try to enumerate the set `U` of vectors that are feasible and 
minimal with regards to the l-infinity norm. We will also try to find their minimal 
support solutions, and find if there is a case where more than one solution is possible.

To find such vectors, we simply choose vectors one by one, and automatically remove
their complement from consideration
