# Classical Molecular Dynamics

This project is used to explore concepts related with classical molecular dynamics.

This simulation make use of periodic boundary conditions and the minimum image convention.

By running `python main.py` a molecular dynamic simulation will run with default values:

- `dt = 0.004`,
- `steps = 5000`,
- `box_len = 10`,
- `lattice_structure = "BCC"`,
- `side_copies = 2`.

The parameter `side_copies` is the number of copies per side of the unitary cell. For example,
with `side_copies = 2` our system will have 8 copies of the unitary cell.

All these parameter can be changed in main.py.

After finishing the simulation, the code will save in the Results folder the positions, velocities, kinetic and potential energy in numpy files and a xyz file for visualizing in VMD.

The current libraries used are:
- numpy
- copy
- itertools
- logging
- tqdm

ToDo:

- Add units to the dimensionless simulation

The starting point of the code is the basic code found here: 
https://colab.research.google.com/drive/1ggB6LQcVtbzEC-wGg7p0KWsmYUUR3aS_?usp=sharing#scrollTo=ePFfVDovk7QU
