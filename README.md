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

***Academic Papers & References***
-
