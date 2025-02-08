# Kernels
[< Documentation](../README.md)

The adaptive kernel methodology relies on using a probability distribution in order to create a new kernel (on each epoch) using the distribution of the output of convolutional layers. In experiments run for this project, we utilize the distributions defined in this document.

## Contents:
* [Cauchy](#cauchy)
* [Gaussian](#gaussian)
* [Gumbel](#gumbel)
* [Laplace](#laplace)

## Cauchy

The Cauchy distribution, named after Augustin-Louis Cauchy, is a continuous probability distribution. It is also known, especially among physicists, as the Lorentz distribution (after Hendrik Lorentz), Cauchy–Lorentz distribution, Lorentz(ian) function, or Breit–Wigner distribution.

### Parameters

* $\chi$ (location)
* $\gamma$ (scale)

### Probability Distirbution Function

$$\frac{1}{\pi\gamma[1 + (\frac{\chi - \chi_0}{\gamma})^2]}$$

## Gaussian

In probability theory and statistics, a normal distribution or Gaussian distribution is a type of continuous probability distribution for a real-valued random variable.

### Parameters

* $\mu$ (location)
* $\sigma^2$ (scale)

### Probability Distirbution Function

$$
\frac{1}{\sqrt{}2\pi\sigma^2}e^{-\frac{(x - \mu)^2}{2x^2}}
$$

## Gumbel

In probability theory and statistics, the Gumbel distribution (also known as the type-I generalized extreme value distribution) is used to model the distribution of the maximum (or the minimum) of a number of samples of various distributions.

### Parameters

* $\mu$ (location)
* $\beta$ (scale)

### Probability Distirbution Function

$$
\frac{1}{\beta}e^{-z + e^{-z}}
$$

Where

$$
z = \frac{x - \mu}{\beta}
$$

## Laplace

In probability theory and statistics, the Laplace distribution is a continuous probability distribution named after Pierre-Simon Laplace. It is also sometimes called the double exponential distribution, because it can be thought of as two exponential distributions (with an additional location parameter) spliced together along the abscissa, although the term is also sometimes used to refer to the Gumbel distribution. The difference between two independent identically distributed exponential random variables is governed by a Laplace distribution, as is a Brownian motion evaluated at an exponentially distributed random time. Increments of Laplace motion or a variance gamma process evaluated over the time scale also have a Laplace distribution.

### Parameters

* $\mu$ (location)
* $\beta$ (scale)

### Probability Distirbution Function

$$
\frac{1}{2b}exp(-\frac{|x - \mu|}{b})
$$