# CS230_quantum_dropout

## Project outline

| Week        | Task 2                                     | Task 1               |
| ----------- | ------------------------------------------ | -------------------- |
| 5/10 ~ 5/15 | Robust Outline                             | Prepare for Shahab   |
| 5/15        | Meet with Shahab + finish Milestone report | Begin deep research  |
| 5/17 ~ 5/22 | Discuss research begin paper outline       | Finish prelim Coding |
| 5/22        | Draft paper done                           | Debugging code       |
| 5/24 ~ 5/29 | Finalize paper                             | Finalize Code        |
| 6/1 ~ 6/3   | Finalize paper                             | Film Project         |

## Research

We are considering how to implement quantum dropout. Although, there have been
mentions of impelementations mentioned in 1. 2. There is no class on quantum
tensorflow or publicly available method for a quantum dropout. We will be
looking at the below papers to see if we could improve the MNIST dataset
training.

### Questions to answer:

What is dropout? How can it be translated to quantum applications?

Can we determine if this will help quantum machine learning? If so how?

What would the architecture be for the "best" quantum machine learning dropout?

Bonus: How can we stand out among the current research?

## Implementation

### Version 1 inspired by Schuld et al.

The approach that often helps is a simpledropout regu-larizationthat is both
quantum-inspired and quantumready (in the sense that it is easy in both
classicalsimulation and quantum execution). The essence of theapproach is to
randomly select and measure one of thequbits, and set it aside for a certain
numberNdropoutof parameter update epochs. After that, the qubit isre-added to
the circuit and another qubit (or, perhaps,no qubit) is randomly dropped. This
strategy works by“smoothing” the model fit and it generally inflates thetraining
error, but often deflates the generalization error.

### Version 2 inspired by Verdon et al.

As our parameters naturally have Gaussian noise inboth the gradient and
parameter value due to our opti-mization approach outlined in Section III using
Gaussianpointer state, the Gaussian multiplicative noise dropoutcomes for free
for our schemes. In a sense the Quantumuncertainty of the wavefunction serves as
natural regu-larizing noise. For Gaussian additive noise dropout, re-fer to
Section V where we describe quantum parametriccircuits for neural networks. In
this section, the com-putational registers are initialized in null-position
qu-dit or qumode eigenstates|0〉. It would be straightfor-ward to use
computational registers which have someadded Gaussian noise to their position
value, i.e., arein a simulated squeezed state rather than a perfect posi-tion
eigenstate initially. Because these types of dropoutare straightforward to
implement with our schemes, wefocus onoperation dropout: stochastically removing
cer-tain subsets of parametric operations.

35The goal of operation dropout is to probabilisticallycreate a blockage of
information flow in the feedforwardcomputational graph. Furthermore, another
importantaspect of dropout is the ability to backpropagate errorswith knowledge
of this erasure error. As our backprop-agation approach relies on the ability to
backpropagateerror signals through the quantum computational graphvia
uncomputation after the feedforward operation andphase kick, we will need to
keep in memory the registerwhich controls the erasure. We use a quantum
state’scomputational basis statistics as the source of classicalstochasticity in
this section for notational convenience,but note that could equivalently replace
these qubits withclassical random Bernoulli variables of equivalent statis-tics.

### Version 3 our method after analysis.

### Relevant Papers

<ol>
  <li> Schuld et al. Circuit-centric quantum classifiers : https://arxiv.org/pdf/1804.00633.pdf </li>
  <li>Verdon et al. A Universal Training Algorithm for Quantum Deep Learning : https://arxiv.org/abs/1806.09729</li>
  <li> Third item </li>
</ol>

## Code/Pipeline

Our pipeline will consist of working with the MNIST dataset classification that
is performed on the quantum tensorflow site (for now).

## Requirments

After you have activate your environment, before you install anything, make sure
to update your pip with: pip install --upgrade pip

## Installation/plugins with jupyter

`nbstripout` is a great option for clearing the output of the jupyter notebooks.
It can be installed using `pip install nbstripout`. For more info see below in
[before you commit section](#beforecommit)

`nbdime` is a great tool for looking at the git diff for jupyter notebooks.

For jupyterlab there is a market place extension which you need to enable first
and that will let you search and install extensions from within jupyter lab. You
can enable the marketplace extension with the following code:

`jupyter labextension install @jupyter-widgets/jupyterlab-manager`

For jupyter notebook, there is a similar extension but that just gets you all
the extension in one go and lets you enable or disable them from the jupyter
home page toolbar. You can install the extension for the jupyter notebook using:
`pip install jupyter_contrib_nbextensions`

`jupyter contrib nbextension install --user`

## <a name="beforecommit"></a> Before you commit or do a pull request:

Since jupyter is not just a text file and uses JSON format, everytime
code/markdown is changed in jupyter notebook, lot of information about the
layout changes as well. This is especially the case for python code which
outputs pictures/graphs. The pictures are stored as text which show up in the
diff. This complicates the git diff. And hence, the best way to version control
jupyter notebooks is by clearing the output before doing a commit. We have been
using nbstripout for clearing output from notebooks automatically. You can
install nbtripout using `pip install nbstripout`. Please make sure to run
`nbstripout notebook.ipynb` to clear the output in a file. To clear the output
in all the notebooks in a given folder, you can run it on a folder, e.g. the
command `nbstripout Qube/*` clears the output from all the notebooks in `Qube`
folder.
