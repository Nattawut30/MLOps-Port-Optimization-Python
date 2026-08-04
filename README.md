# <p align="center"> MLOps: Portfolio Optimization <p/>
<br>**Nattawut Boonnoon**<br/>
- LinkedIn: www.linkedin.com/in/nattawut-bn
- Email: nattawut.boonnoon@hotmail.com

***Overview***
- 

#### Link: <br>
#### Presentations Slides:

My Portfolio optimization pipeline covering Black-Litterman and risk parity allocation, Brownian motion and Heston stochastic volatility simulation, and FFT tail risk hedge pricing. Built as an MLOps project: a scheduled batch pipeline computes and versions every result on a Streamlit dashboard displays the latest output.

#### System Architecture:

The pipeline runs in two separate stages.

A scheduled batch job (GitHub Actions) extracts prices, computes returns
and covariance, estimates expected returns, optimizes portfolio weights,
simulates risk, and prices a tail risk hedge. Results are committed back
to this repository rather than stored in a hosted database, giving every
run a permanent, versioned record in git history.

A Streamlit dashboard reads only the finished output of that job. It
never recomputes anything and never touches the data source.

Data moves through three layers on disk. (Bronze -> Silver -> Gold)

Design decisions:

- The data source sits behind a single interface. The original source
  was replaced during development after it started blocking scripted
  requests. Swapping providers again touches one file, not the pipeline.
- Portfolio construction and risk simulation each have two interchangeable
  implementations behind a shared interface: max-Sharpe and risk parity
  for optimization, geometric Brownian motion and Heston for simulation.
- The dashboard never imports the compute modules directly. It reads only
  the parquet files those modules produce.

# <p align="center">Mathematical Model<p/>


**1. Returns and covariance**

$$r_t = \\ln\\left(\\frac{P_t}{P_{t-1}}\\right)$$

Log returns, used in place of price levels since they are close to stationary.

$$\\hat{\\Sigma} = (1 - \\alpha)S + \\alpha F$$

Ledoit-Wolf shrinkage covariance. S is the sample covariance, F is the
shrinkage target, alpha is estimated from the data.

**2. Expected returns**

$$\\pi = \\delta \\Sigma w_{mkt}$$

Black-Litterman equilibrium return. With no investor views, this is the
model's output directly.

$$E[R] = \\left[(\\tau\\Sigma)^{-1} + P^T\\Omega^{-1}P\\right]^{-1}\\left[(\\tau\\Sigma)^{-1}\\pi + P^T\\Omega^{-1}Q\\right]$$

Black-Litterman posterior return, blending the equilibrium with investor
views P, Q, and view uncertainty Omega.

**3. Portfolio construction**

$$\\max_{w} \\frac{w^T\\mu - r_f}{\\sqrt{w^T\\Sigma w}} \\quad \\text{subject to} \\sum_i w_i = 1,\\ w_i \\geq 0$$

Max-Sharpe, long only, solved with SLSQP.

$$w_i(\\Sigma w)_i = w_j(\\Sigma w)_j \\quad \\forall\\, i, j$$

Risk parity. Every asset contributes equally to total portfolio variance.

**4. Simulation**

$$dS_t = \\mu S_t\\,dt + \\sigma S_t\\,dW_t$$

Geometric Brownian motion, constant volatility.

$$dS_t = \\mu S_t\\,dt + \\sqrt{v_t}\\,S_t\\,dW_t^S \\qquad dv_t = \\kappa(\\theta - v_t)\\,dt + \\xi\\sqrt{v_t}\\,dW_t^v \\qquad dW_t^S dW_t^v = \\rho\\,dt$$

Heston stochastic volatility, simulated with a full truncation Euler
scheme so variance cannot go negative.

$$\\text{VaR}_\\alpha = \\inf\\{l : P(L > l) \\leq 1-\\alpha\\} \\qquad \\text{CVaR}_\\alpha = E[L \\mid L \\geq \\text{VaR}_\\alpha]$$

**5. Hedge pricing**

$$C(K) = \\frac{e^{-\\alpha k}}{\\pi}\\int_0^{\\infty} e^{-ivk}\\,\\psi(v)\\,dv$$

Heston option price via the Carr-Madan FFT method, using the model's
characteristic function. Priced against two independent methods, FFT and
Monte Carlo, cross-checked against each other on every run.

$$P = C - S_0 + Ke^{-rT}$$

Put-call parity, converting the FFT call price into a put price.

# <p align="center">Acknowledgments<p/>

***Dependencies***
- 
`Streamlit` · `Pandas` · `Numpy` · `Plotly` · `Scikit-Learn` · `PyTorch` · `MLFlow` · `PyArrow` · `SciPy`

***Academic Papers & References***
-

- Markowitz, H. (1952). *"Portfolio Selection."* The Journal of Finance, 7(1), 77-91.
- Black, F., & Scholes, M. (1973). *"The Pricing of Options and Corporate Liabilities."* Journal of Political Economy, 81(3), 637-654.
- Sharpe, W. F. (1966). *"Mutual Fund Performance."* The Journal of Business, 39(1), 119-138.
- Black, F., & Litterman, R. (1992). *"Global Portfolio Optimization."* Financial Analysts Journal, 48(5), 28-43.
- Ledoit, O., & Wolf, M. (2004). *"A Well-Conditioned Estimator for Large-Dimensional Covariance Matrices."* Journal of Multivariate Analysis, 88(2), 365-411.
- Heston, S. L. (1993). *"A Closed-Form Solution for Options with Stochastic Volatility with Applications to Bond and Currency Options."* The Review of Financial Studies, 6(2), 327-343.
- Carr, P., & Madan, D. (1999). *"Option Valuation Using the Fast Fourier Transform."* Journal of Computational Finance, 2(4), 61-73.
- Maillard, S., Roncalli, T., & Teiletche, J. (2010). *"The Properties of Equally Weighted Risk Contribution Portfolios."* The Journal of Portfolio Management, 36(4), 60-70.
- Rockafellar, R. T., & Uryasev, S. (2000). *"Optimization of Conditional Value-at-Risk."* Journal of Risk, 2(3), 21-41.
- Howell, E., (2025), *"Modern Boiler plate to build an end-to-end ML project".,* ML-Project-Starter.
