# Keras Optimizer Visualization

Easily **see** the differences between SGD, Adam, RMSprop, etc.

The application looks like this:

[home_screen](img/home_screen.png)

to the left we can adjust the optimizer hyperparameters, including the learning surface, and to the right we have an interactive Plotly graph.

We can change the learning surface and explore how things like momentum affect learning. Here is an example of SGD with momentum overcoming a local minimum and going on to a global minimum:

[anthill](img/anthill.png)