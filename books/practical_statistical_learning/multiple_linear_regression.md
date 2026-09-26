---
title: "Multiple Linear Regression"
free: true
---

## Formulation

Suppose that we have the following training data with $p$ features and $n$ data points:

$$
\left\{ x_{i1}, x_{i2}, \dots, x_{ip}; y_i \right\}_{i = 1}^{n}
$$

A model can be defined as:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \dots + \beta_p x_{ip} + \varepsilon_i, \quad i = 1, \dots, n
$$

where

- $\beta_0, \dots, \beta_p$: the parameters of the model that we want to estimate
- $\varepsilon_i$: the noise (error term), which we assume to be i.i.d. with mean $0$ and variance $\sigma^2$

To make it easier to read, we can rewrite the entire expression in matrix form:

$$
y = \begin{pmatrix}
y_1 \\
y_2 \\
\vdots \\
y_n
\end{pmatrix}
\quad
\varepsilon = \begin{pmatrix}
\varepsilon_1 \\
\varepsilon_2 \\
\vdots \\
\varepsilon_n
\end{pmatrix}
\quad
\beta = \begin{pmatrix}
\beta_0 \\
\beta_1 \\
\vdots \\
\beta_p
\end{pmatrix}
$$

$$
\mathbf{X} = \begin{pmatrix}
1 & x_{11} & \cdots & x_{1p} \\
1 & x_{21} & \cdots & x_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & \cdots & x_{np}
\end{pmatrix}
$$

So we have:

$$
\underset{n \times 1}{y} = \underset{n \times (p + 1)}{\mathbf{X}} \; \underset{(p + 1) \times 1}{\beta} + \underset{n \times 1}{\varepsilon},
\quad \varepsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})
$$

where we have some terminology for each of them:

- $y$ ($n \times 1$): Response
- $\mathbf{X}$ ($n \times (p + 1)$): Design Matrix
- $\beta$ ($(p + 1) \times 1$): Coefficient Vector
- $\varepsilon$ ($n \times 1$): Error Term

:::message

We append a column of 1s to the design matrix so that the intercept $\beta_0$ is applied to every data point. And because of that, the number of columns becomes $p + 1$.

:::

### How Do We Know the $\beta$?

Given that it is impossible for us to know what those $\beta$ really are, we can only **estimate** them by finding the parameters that can get us closest to our data points. This estimated version of the parameters is a function of the data, so it is written with a hat:

$$
\hat{\beta} = \begin{pmatrix}
\hat{\beta}_0 \\
\hat{\beta}_1 \\
\vdots \\
\hat{\beta}_p
\end{pmatrix}
$$

Now, the problem is: "How do we actually perform the estimation?" The answer: **Residual Sum of Squares (RSS)**.

---

## Least Squares Estimation

RSS is a function of the parameters we plug in, so we write it as $\mathrm{RSS}(\beta)$:

$$
\mathrm{RSS}(\beta) = \lVert y - \mathbf{X}\beta \rVert^2 = (y - \mathbf{X}\beta)^{T}(y - \mathbf{X}\beta)
$$

Based on the formula, we can see that RSS is the **squared** Euclidean distance between the values our model fits and the observed responses, which is the squared length of the residual vector $y - \mathbf{X}\beta$. If we visualize it, we can get a plot like this:

:::details How to visualize this?

```python
import matplotlib.pyplot as plt
import torch

# Assume that
# - our formula is y = 3 + 0.8 * x1 +  0.5 * x2 + noise
# - we have 30 data points

# Create random data
torch.manual_seed(42)
n = 30

x1 = torch.rand(n) * 10
x2 = torch.randn(n) * 10
noise = torch.randn(n) * 3
y = 3 + 0.8 * x1 + 0.5 * x2 + noise

# Create design matrix
X = torch.stack(
    [
        torch.ones(n),
        x1,
        x2,
    ],
    dim=1,
)

# estimation
beta = torch.linalg.lstsq(X, y).solution
beta0, beta1, beta2 = beta
y_hat = X @ beta

# calculate RSS
residuals = y - y_hat
rss = residuals.pow(2).sum()

# visualization
x1_grid = torch.linspace(
    x1.min(),
    x1.max(),
    25,
)

x2_grid = torch.linspace(
    x2.min(),
    x2.max(),
    25,
)
X1_grid, X2_grid = torch.meshgrid(
    x1_grid,
    x2_grid,
    indexing="ij",
)
Y_grid = beta0 + beta1 * X1_grid + beta2 * X2_grid

# plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(
    111,
    projection="3d",
)
ax.plot_surface(
    X1_grid,
    X2_grid,
    Y_grid,
    alpha=0.25,
    color="slateblue",
    edgecolor="darkslateblue",
    linewidth=0.4,
    antialiased=True,
)
ax.scatter(
    x1,
    x2,
    y,
    s=55,
    color="orange",
)

# the dotted lines are the residuals that RSS squares and sums up
for i in range(n):
    ax.plot(
        [x1[i], x1[i]],
        [x2[i], x2[i]],
        [y_hat[i], y[i]],
        linestyle=":",
        color="gray",
    )

ax.set_title(f"RSS = {rss:.2f}")
plt.show()
```

:::

![RSS](/images/multiple-linear-regression.png)

So, by minimizing the RSS, we can estimate $\beta$. $\mathrm{RSS}(\beta)$ is a convex quadratic function of $\beta$, so it is enough to take the derivative with respect to $\beta$ and set it to 0:

$$
\begin{aligned}
\frac{\partial \mathrm{RSS}(\beta)}{\partial \beta} &= -2 \mathbf{X}^{T} (y - \mathbf{X}\beta) = 0 \\
&\Rightarrow \mathbf{X}^{T}y - \mathbf{X}^{T} \mathbf{X}\beta = 0 \\
&\Rightarrow \mathbf{X}^{T} \mathbf{X}\beta = \mathbf{X}^{T}y
\end{aligned}
$$

The last line is called the **normal equations**. Therefore, we can derive the least squares estimator in MLR to be:

$$
\hat\beta = (\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T}y
$$

:::message

$(\mathbf{X}^{T}\mathbf{X})^{-1}$ only exists when $\mathbf{X}$ has full column rank, i.e. $\mathrm{rank}(\mathbf{X}) = p + 1$. That needs $n \geq p + 1$ and no perfectly collinear features.

:::

If we now substitute the model $y = \mathbf{X}\beta + \varepsilon$ back into the estimator, we can see how far it sits from the true $\beta$:

$$
\begin{aligned}
\hat\beta &= (\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T}(\mathbf{X}\beta + \varepsilon) \\
          &= (\mathbf{X}^{T} \mathbf{X})^{-1}\mathbf{X}^{T} \mathbf{X}\beta + (\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \varepsilon \\
          &= \beta + (\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \varepsilon \\
\Rightarrow \hat\beta - \beta &= (\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \varepsilon
\end{aligned}
$$

We can also derive our prediction model as:

$$
\hat y = \mathbf{X} \hat\beta = \underbrace{\mathbf{X}(\mathbf{X}^{T}\mathbf{X})^{-1}\mathbf{X}^{T}}_{\mathbf{H}} \, y
$$

where $\mathbf{H}$ is called the **hat matrix**, and it projects $y$ onto the column space of $\mathbf{X}$.

---

## Standard Error

Based on the result that we derived for the least squares estimator $\hat\beta$, we can see that, once $\mathbf{X}$ is treated as fixed, the randomness of our estimator comes entirely from the noise $\varepsilon$. Given

$$
\varepsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})
\Rightarrow \mathrm{Cov}(\varepsilon) = \sigma^2 \mathbf{I}
$$

and knowing that the true $\beta$ is a constant that adds no randomness, we obtain

$$
\begin{aligned}
\mathrm{Cov}(\hat\beta) &= \mathrm{Cov}\left((\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \varepsilon\right) \\
               &= (\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \, \mathrm{Cov}(\varepsilon) \, \mathbf{X}(\mathbf{X}^{T} \mathbf{X})^{-1} \\
               &= (\mathbf{X}^{T} \mathbf{X})^{-1} \mathbf{X}^{T} \sigma^2 \mathbf{I} \, \mathbf{X}(\mathbf{X}^{T} \mathbf{X})^{-1} \\
               &= \sigma^2 (\mathbf{X}^{T} \mathbf{X})^{-1}
\end{aligned}
$$

and we can know that the variance of a single coefficient is the matching diagonal entry:

$$
\mathrm{Var}(\hat\beta_j) = \sigma^2 \left[(\mathbf{X}^{T} \mathbf{X})^{-1}\right]_{jj}
$$

In practice, $\sigma^2$ is unknown, so we estimate it from the residuals and plug it in. That gives us the standard error:

$$
\hat\sigma^2 = \frac{\mathrm{RSS}}{n - p - 1}, \quad
\mathrm{SE}(\hat\beta_j) = \hat\sigma \sqrt{\left[(\mathbf{X}^{T} \mathbf{X})^{-1}\right]_{jj}}
$$

:::message

$\mathrm{SE}(\hat\beta_j)$ is not $\sqrt{\mathrm{Var}(\hat\beta_j)}$ itself: it is that value with the unknown $\sigma$ replaced by the estimate $\hat\sigma$. And that substitution is exactly the reason why the statistic below follows a $t$ distribution instead of a standard normal one.

:::

so we can derive the $t$-statistic

$$
\frac{\hat\beta_j - \beta_j}{\mathrm{SE}(\hat\beta_j)} \sim t_{n - p - 1}
$$

### What's the Use of This?

For example, if you want to test the benefits of increasing the number of predictors in your prediction model, you will need a way to quantify how much error the new parameters bring down. In this case, you will usually use the F-statistic:

$$
F = \frac{(\mathrm{RSS}_{reduced} - \mathrm{RSS}_{full}) / q}{\mathrm{RSS}_{full} / (n - p_{full} - 1)}
$$

where $q$ is the number of parameters that you reduce, which is also the numerator's degrees of freedom. So, basically, you're making a null hypothesis saying that those $q$ coefficients are all $0$, i.e. the introduction of the new parameters makes no difference, and then you will try to reject it by checking the number produced by F-stats against the $F_{q, \, n - p_{full} - 1}$ distribution. Then how is this related to our $t$-statistic? That's actually straightforward. If you set the number of new parameters to one ($q = 1$), then the two tests are the same test: $t^2$ equals the F-stats exactly, and both give the same p-value. You can test this out with the code below:

```python
import numpy as np

rng = np.random.default_rng(2024)
n, p = 50, 4
X = np.c_[np.ones(n), rng.standard_normal((n, p))]
y = X @ np.array([1.0, 2.0, -1.0, 0.0, 5.0]) + rng.normal(0, 1.5, n)


def fit(X, y):
    b = np.linalg.solve(X.T @ X, X.T @ y)
    return b, np.sum((y - X @ b) ** 2)


b, rss_full = fit(X, y)
df = n - X.shape[1]
s2 = rss_full / df  # estimated noise variance, i.e. sigma_hat^2
se = np.sqrt(s2 * np.diag(np.linalg.inv(X.T @ X)))
t = b / se

# drop the 4th column, whose true coefficient is 0
_, rss_red = fit(np.delete(X, 3, axis=1), y)
# df_red - df_full = 1
F = (rss_red - rss_full) / s2
print(t[3] ** 2, F)
```
