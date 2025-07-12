# Incremental VGICP 的观测

根据[iG-LIO: An Incremental GICP-Based Tightly-Coupled LiDAR-Inertial Odometry](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10380742)的描述

$$
\begin{equation}
  \begin{aligned}
    \mathbf{r} = \mathbf{\mu} - {}^{W}\mathbf{R}_{I_i}({}^{I}\mathbf{R}_{L} \mathbf{p}_i + {}^{I}\mathbf{t}_{L}) - {}^{W}\mathbf{t}_{I_i}
  \end{aligned}
\end{equation}
$$

$$
\begin{equation}
  \begin{aligned}
  \Omega = \sigma_{GICP}(\Sigma + ({}^{W}\mathbf{R}_{I_i} * {}^{I}\mathbf{R}_{L}) \Sigma_{i} ({}^{W}\mathbf{R}_{I_i} * {}^{I}\mathbf{R}_{L}))^{-1}
  \end{aligned}
\end{equation}
$$

其中, $\mathbf{\mu}, \Sigma$ 是在地图中找到的对应的体素网格的均值和协方差, $\sigma_{GICP}$ 是VGICP的权重,但是我们并不是优化方法,用的是滤波,所以可以将这一项替换为鲁棒核函数,
例如,将其换为CauchyLoss:

$$
\begin{equation}
  \begin{aligned}
    \min_{\delta x} \sum_{i=1}^n \rho\left( \| r_i + H_i \delta x \|^2 \right)
  \end{aligned}
\end{equation} 
$$

其中 $\rho(s) = c^2 \cdot \log\left(1 + \frac{s}{c^2} \right)$

这个操作的Cauchy权重,其实就是对 $\rho(s)$ 求导:
$$
\begin{equation}
  \begin{aligned}
    w_i = \frac{1}{1 + \frac{s_i}{c^2}} \quad \text{(Cauchy权重)}
  \end{aligned}
\end{equation}
$$

因此, 协方差的矩阵的加权调整为:
$$
\begin{equation}
  \begin{aligned}
    \Omega^{\text{robust}} = {w_i} \Omega
  \end{aligned}
\end{equation}
$$

对于 $\mathbf{H}$ 的求解:
$$
\begin{equation}
  \begin{aligned}
    \mathbf{H} = \begin{bmatrix}
      -\mathbf{I} & {}^{W}\mathbf{R}_{I_i} [{}^{I}\mathbf{R}_{L} \mathbf{p}_i + {}^{I}\mathbf{t}_{L}]_{\times} & {}^{W}\mathbf{R}_{I_i} {}^{I}\mathbf{R}_{L} [\mathbf{p}_i]_{\times} & -{}^{W}\mathbf{R}_{I_i} & 0 & 0 & 0 & 0
    \end{bmatrix}
  \end{aligned}
\end{equation}
$$


