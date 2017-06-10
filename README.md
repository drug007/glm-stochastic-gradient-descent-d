# Generalized Linear Models and Stochastic Gradient Descent in D

The [Mir GLAS](https://github.com/libmir/mir-glas) library has shown that D language is capable of high performance calculations to rival those written in C and C++. Mathematical analysis libraries native to D however are still in short supply. In this blog we roll our own gamma GLM model written in optimization style using the Newton-Raphson method and we carry out linear regression using gradient descent.

# Mathematical Preliminaries

The main aim of a regression algorithm is to find a set of coefficients $\beta$ that maximize the likelihood of a target variable ($y$) given a set of explanatory variables $x$(). The algorithm makes assumptions regarding the distribution of the target variable, and the independence of observations of the explanatory variables.

The likelihood function represents assumption of the distribution of the target variable. The likelhood function for the Gamma distribution is given by:

$$
L(x) = \frac{1}{\Gamma(x)}\theta^{k-1}_{-\frac{x}{\theta}}
$$

and the Normal distribution likelihood function is given by:

$$
L(x) = \frac{1}{2\pi \sigma^2}\exp^{-\frac{(y - \mu)^2}{2\sigma}}
$$

In practice however, we actually maximise the log-likelihood ($l$) - that is the log of the likelihood function taken over the whole data set. The gradient function for the Gamma log-likelihood is given by

$$
\frac{\partial l}{\partial \beta_j} = -k x_j + k y x_j \exp^{x\beta}
$$

and its curvature is given by

$$
-y
$$

# Regression as Optimization

Regression is thus an optimization algorithm that maximizes the log-likelihood function for a set of $\beta$ coefficients. It can thus be solved by numerical optimization.

## Newton-Raphson Algorithm


