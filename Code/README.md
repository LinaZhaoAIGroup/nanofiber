# Sup_Nanofiber
Supervised ML framework for nanofiber orientation prediction.
Many biological, bioinspired and synthetic materials exhibit/contain/are made of 3D networks of textured nanofibers. Their structure and properties are closely related to the fiber orientations within them.

This repository contains developed code to implement the supervised ML framework for automatically and efficiently predicting nanofiber orientation based on 1D WAXD $I{\chi}$ data, as well as related simulated and experimental dataset. The components are:

- MySim and Sim - the diffraction model program for two groups of nanofiber.
- models -  neural network models, including FCNN, PreActResNet, DenseNet and NiN.
- utils -  some useful functions for identifying gap locations for experimental data, reconstruction through diffraction models, angle-to-xy transformation etc.
- training -  demo example for training the neural networks on dynamically masked curves and added Poisson noises. Testing on simulated and experimental test data.

​	

