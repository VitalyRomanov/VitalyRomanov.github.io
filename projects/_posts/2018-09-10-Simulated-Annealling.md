---
layout: post
title: Simulated Annealling
categories: [optimization]
tags: [stochastic optimization]
description: "Some choose in life should be made by sampling randomly."
mathjax: true
show_meta: true
---

# Simulated Annealing

## Motivation

Simulated Annealing (SA) is a stochastic global optimization algorithm. As many other randomized algorithms, SA appears to be surprisingly simple, yet capable at achieving its goal (given conditions). The name and inspiration for the algorithm come from metallurgy, where controlled heating and cooling of the material allows it to form larger crystalline structure and improve properties. A popular description of the method can be found on its [Wikipedia page](https://en.wikipedia.org/wiki/Simulated_annealing).

## Algorithm

### Procedure

The procedure for SA is very similar to the Metropolis-Hastings (MH) algorithm

1. Sample initial $$\mathbf{x}_0$$, set time step $$t=0$$
2. Set initial temperature $$T$$
3. Generate $$\mathbf{x}'$$ from $$g(\mathbf{x}'\mid\mathbf{x}_t)$$
4. Calculate acceptance ratio $$\alpha=\frac{p^*(\mathbf{x}')}{p^*(\mathbf{\mathbf{x}}_t)}$$
5. Generate $$u\sim U(0,1)$$. If $$u\leq \alpha$$, accept the new state $$x_{t+1}=x'$$, otherwise propagate the old state. Pass $$\mathbf{x}_{t+1}$$ to the output of the sampler
6. Anneal temperature $$T$$
7. Increment time step $$t$$
8. Repeat from step 2 until cooled down

The description and details of every step are given further.


### Metallurgy Analogy

> When the solid is heated, its molecules start moving randomly, and its energy increases. If the subsequent process of cooling is slow, the energy of the solid decreases slowly, but there are also random increases in the energy governed by the Boltzmann distribution. If the cooling is slow enough and deep enough to unstress the solid, the system will eventually settle down to the lowest energy state where all the molecules are arranged to have minimal potential energy.

The terms in SA are borrowed from the field of metallurgy. The optimization problem (minimization or maximization) is treated as a *system*. We want it to achieve a stable state with the least of *energy* (determined by cost function). This is achieved by heating the system up (increasing the variance of sampling), and then gradually *cooling it down*. This metallurgical process is merely an abstraction, and the actual optimization relies on sampling from an artificially designed probability distribution that peaks around global optimum. The way to construct this probability distribution is described further.

### Energy Distribution

The main trick of SA is the energy distribution. This distribution peaks around the point of the global optimum, and thus, there is a good chance to sample this value. First, consider how the energy distribution is constructed. 

> SA steps are similar to the Metropolis-Hastings algorithm. Remember that the idea behind MS algorithm is to create an MCMC that samples from the distribution proportional to $$f(x)$$, so that the limiting distribution is equal to the target distribution. The idea of Simulated Annealing method is similar. 

Assume you have a function $$h(x)$$ that you want to minimize in the region $$x\in[-3.3]$$.  In simulated annealing, we say that we have a system, and the function $$h(x)$$ describes its state. 

![](https://i.imgur.com/OOyui97.png)

```python
w = np.array([0.13511111, 0.30565641, 0.85789735, 0.19849365, 0.0972464 ])
phi = np.array([0.76633178, 0.07614028, 0.57433697, 2.05374239, 1.97379992])
a = np.array([12.80479424,  1.57398667,  1.46134803,  4.93294448,  4.12038971])
h = np.vectorize(lambda x: sum(np.cos(x * a + phi) * w))
```

As you can see, the non-concavity of the function makes it hard to optimize with gradient descent. Define the cost function, also referred to as system energy

$$
E(x) = h(x)
$$

for maximization problems the sign of $$h(x)$$ would be negative. This function is used to define the energy distribution. Our goal is to create the distribution, where the global optimum has a high weight. One of the possible transformations adopted by SA is 

$$
p(x)=\frac{e^{-E(x)}}{\int e^{-E(y)}dy}
$$

This transformation is inspired by [Boltzman Distribution](https://en.wikipedia.org/wiki/Boltzmann_distribution) and is similar to softmax. We do not want to compute the denominator of this distribution since it would imply passing through the entire search space (if we do this, we can find minimum right away). 

![](https://i.imgur.com/hOfyovx.png)

Notice that the density maximum is observed exactly where the global minimum is. The problem is, even if we sample from this distribution, there is no guarantee the sample will be close to the desired minimum. This issue is solved with annealing. 

### Annealing

Annealing is the process of cooling down. When the system is heated, we want to emulate Brownian motion by transitioning the state often. In other words, when the system is heated there should be a high chance for the system to change its state. From a sampling perspective, this is achieved by making the peaks in the distribution less distinct. In SA, this is done by adding temperature parameter to the target distribution function as follows

$$
\begin{aligned}
p(x)&=\frac{e^{-E(x)/T}}{\int e^{-E(y)/T}dy} 
\end{aligned}
$$

And the proportional function is

$$
p^*(x)=e^{-E(x)/T}
$$

By gradually reducing the temperature $$T$$ we make the peaks more pronounced, and encourage the system to stabilize in the point with the minimal energy. In our example, we end up sampling from the distribution with only one peak. 

![](https://i.imgur.com/87bSSn3.gif)

### Annealing Schedule

Throughout the optimization procedure, the temperature is gradually annealed. The annealing schedule specifies the speed of the temperature decrease. The most common way to anneal is with [exponential decay](https://en.wikipedia.org/wiki/Exponential_decay) or a geometric series

$$
T_{t+1} = T_t \cdot a
$$

where $$a < 1$$ is the annealing rate. From MH algorithm we know that the sampling procedure has the burn-in time. If the temperature is decreased too fast, the process may end up stuck in a local optimum.

### Proposal Distribution

The problem at hand determines the choice of the proposal distribution. For continuous space problems, the Normal proposal distribution is a common choice. The alternative is described in the further section about combinatorial optimization.

### Putting Everything Together

The optimization procedure is similar to the MH algorithm

1. Sample initial $$\mathbf{x}_0$$, set time step $$t=0$$
2. Set initial temperature $$T$$
To avoid problems with numerical overflow when calculating the exponent, set the temperature comparable with the initial value of the system's energy
3. Generate $$\mathbf{x}'$$ from $$g(\mathbf{x}'|\mathbf{x}_t)$$
For continuous problems, the common solution is to use the normal distribution
4. Calculate acceptance ratio $$\alpha=\frac{p^*(\mathbf{x}')}{p^*(\mathbf{\mathbf{x}}_t)}$$
5. Generate $$u\sim U(0,1)$$. If $$u\leq \alpha$$, accept the new state $$x_{t+1}=x'$$, otherwise propagate the old state. Pass $$\mathbf{x}_{t+1}$$ to the output of the sampler
6. Reduce temperature $$T$$
Temperature annealing does not have to take place on every iteration. The temperature can be decreased every N iteration as well. There is no strict rule about this
7. Increment $$t$$
8. Repeat from step 2 until cooled down
The solution can be sampled before the system is cooled down, but we have no way of knowing whether this was the optimal solution

For the example above, the optimization process will look the following way

![](https://i.imgur.com/V6jG1On.gif)

## Combinatorial optimization

Due to the nature of this optimization approach, it does not require the function to be differentiable. This allows to apply this method to optimize combinatorial problems.

Consider the traveling salesman problem
> Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city once and returns to the origin city?

The energy distribution for this problem is easy to express in a format suitable for SA. 

$$
p_{sales}^*(path) = e^{-dist(path)/T}
$$

where `path` is the ordered list of cities to visit, `dist` is the function that computes path distance. The main trick to solve traveling salesman with SA is to pick a proper proposal distribution. The common proposal policy for this problem is the following

1. Pick two cities in the path
2. Exchange their positions in the path
3. Return the new proposed path

```python
import random
import time
from math import radians, acos, sin, cos, exp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from cartopy import feature
from matplotlib import animation

from config import output_frequency, init_temperature, cooling_rate


def acceptable_to_be_new_best(new_route, old_route, temperature):
    if new_route.get_whole_distance() < old_route.get_whole_distance():
        return True
    else:
        acceptance_ratio = exp((old_route.get_whole_distance() - new_route.get_whole_distance()) / temperature)
        return acceptance_ratio > random.uniform(0, 1)


class City:

    def __init__(self, lat, lon, name):
        self.lat = lat
        self.lon = lon
        self.name = name

    def get_distance(self, other_city):
        slat = radians(self.lat)
        slon = radians(self.lon)
        elat = radians(other_city.lat)
        elon = radians(other_city.lon)

        dist = 6371.01 * acos(sin(slat) * sin(elat) + cos(slat) * cos(elat) * cos(slon - elon))
        return dist

    def __str__(self):
        return self.name


class Tour:
    def __init__(self):
        self.path = []

    def add_city(self, city: City):
        self.path.append(city)

    def get_whole_distance(self):
        distance = 0
        length = len(self.path)
        for i in range(0, length):
            current_city = self.path[i]
            if i == length - 1:
                next_city = self.path[0]
            else:
                next_city = self.path[i + 1]
            distance += current_city.get_distance(next_city)
        return distance

    def __copy__(self):
        new_tour = Tour()
        for city in self.path:
            new_tour.add_city(city)
        return new_tour

    def plot(self, ax=None):

        for i in range(0, len(self.path)):
            city = self.path[i]
            if i == len(self.path) - 1:
                next_city = self.path[0]
            else:
                next_city = self.path[i + 1]

            ax.plot([city.lon, next_city.lon], [city.lat, next_city.lat],
                    color='blue', marker='o',
                    transform=ccrs.PlateCarree(),
                    )


fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

df = pd.read_csv('cities.csv')
top_cities = df.sort_values('Население', ascending=False).head(30)
init_tour = Tour()
for index, city in top_cities.iterrows():
    init_tour.add_city(City(city['Широта'], city['Долгота'], city['Город']))
    print(city['Город'], city['Население'])
print(init_tour.get_whole_distance())

T = init_temperature
current_tour = init_tour
iterations = 0
start_time = time.time()


def tour_gen():
    global T, iterations, current_tour
    while T > 1:
        first_city_index_to_swap = int(random.uniform(0, len(current_tour.path)))
        second_city_index_to_swap = int(random.uniform(0, len(current_tour.path)))

        new_tour = current_tour.__copy__()
        new_tour.path[first_city_index_to_swap], new_tour.path[second_city_index_to_swap] = new_tour.path[
                                                                                                second_city_index_to_swap], \
                                                                                            new_tour.path[
                                                                                                first_city_index_to_swap]

        if acceptable_to_be_new_best(new_tour, current_tour, T):
            current_tour = new_tour
        T = T * (1 - cooling_rate)
        iterations += 1
        if iterations % output_frequency == 0:
            print(current_tour.get_whole_distance())
            print("--- %s seconds ---" % (time.time() - start_time))
            yield current_tour

    print('End!!!')


def animate(tour):
    ax.clear()  # Clear previous data
    ax.stock_img()
    ax.set_extent([20, 140, 30, 70], crs=ccrs.PlateCarree())

    ax.add_feature(feature.LAND)
    ax.add_feature(feature.COASTLINE)
    ax.add_feature(feature.BORDERS)

    tour.plot(ax)
    ax.autoscale_view()

    return


# Animate it!
anim = animation.FuncAnimation(fig, animate, init_func=lambda: None, frames=tour_gen)
plt.show()
```

<iframe width="560" height="315" src="https://www.youtube.com/embed/pXEzDMz0jnc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>