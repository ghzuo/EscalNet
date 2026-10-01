# EscalKNM

EscalNet is a Python package for analyzing molecular dynamics simulations based on the effective energy and plotting results based on Jupyter Notebook tools.

The effective energy, which was filtered from the total potential energy of simulation trajectories by combining fast Fourier transform (FFT) and neural network, is an efficacious order parameter to describe the slow conformational change of the complex system.

## Dependencies

- Python, Jupyter Notebook
- Numpy, Pandas, Matplotlib, Seaborn
- Sklearn, PyTorch

## Main Functions/Scripts

- toolkits/EscalNet.py: Analyze the effective energy of molecular dynamics simulations
- toolkits/MarkovTransition.py: Analyze the Markov transition matrix
- EscalNet.ipynb: Plot the results based on Jupyter Notebook tools
