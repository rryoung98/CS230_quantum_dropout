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

## Implementation

### Version 1 inspired by Schuld et al.

### Version 2 inspired by Verdon et al.

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
