# SafeOptionCritic
We introduce a novel framework known as "Safe-Option-Critic" (SOC), which provides safety in Option framework. Here safety is defined as "Prevention from accidents in ML systems due to poor designing on AI systems". Options provide a method to incorporate temporal abstractions in RL setting. Here is link to [Between MDPs and semi-MDPs:
A framework for temporal abstraction
in reinforcement learning](http://www-anw.cs.umass.edu/~barto/courses/cs687/Sutton-Precup-Singh-AIJ99.pdf). The [Option-Critic Architecture](https://arxiv.org/pdf/1609.05140.pdf) provides a method for end-to-end learning of options including option policies, termination condition and policy over options.

# Getting Started
The repo contains code for running SOC framework on tabular and continous state-space environments. For experiments in ALE see [repo](https://github.com/kkhetarpal/safe_a2oc_delib)

# Prerequisites
```
* Numpy
* OpenAI Gym
* Matplotlib
* Seaborn
```

# Environments Used
* Frozen FourRoom Env [fourrooms.py]
* Puddle-World Env [puddlesimple.py]

# Training
The following command is used for training Tabular (FrozenFourRoom Environment)
```
python FourRoomSOC.py --nruns 50 --nepisodes 500 --beta 0.1 --temperature 0.001 --lr_critic 0.5 --lr_intra 0.01 --lr_term 0.1
```
Use deafult parameters in code for best setting

The following command is used for training Continuous State Space Env : Puddle-World Env
```
python PuddleSOC.py --nruns 50 --nepisodes 200 --beta 0.015 --temperature 0.1 --lr_critic 0.5 --lr_intra 0.05 --lr_term 0.05
```

# Plotting FourRoom Policies and Option Termination

Use the following PolttingPolicies.ipynb (iPython Notebook) for visualization of option policies and option termination in Frozen FourRoom Env.

# Plotting Return plots
Use the following ReturnPlots.ipynb (iPython Notebook) for visualization of return plots in Frozen FourRoom Env and Puddle-World Env.

