# DRL for collision avoidance in Carla

A combination of traditional optimal control & reinforcement learning

- MPC based 

  ![image-20200702134757530](/Users/jiqiandong/Desktop/OneDrive - purdue.edu/work2/image-20200702134757530.png)

  

- Model the system dynamic using a function approximation

  $$x' = f_\theta(x, u)$$

- Fitted value iteration 
  $$
  \forall k, \hat{J}^{*}\left(\mathbf{x}_{k}\right) \Leftarrow \min _{\mathbf{u}}\left[\ell\left(\mathbf{x}_{k}, \mathbf{u}\right)+\hat{J}^{*}\left(f\left(\mathbf{x}_{k}, \mathbf{u}\right)\right)\right]
  $$

