.. _installation:

Installation
============
Clone the github repository

.. code-block:: console

    $ git clone https://github.com/prajeeshag/S2SCoupledRoadMap.git


To use *aps2s*, first create a *conda* (or *mamba*) environment and install the dependencies by running the following command.

.. code-block:: console 

    $ cd S2SCoupledRoadMap/aps2s 
    $ conda env create -f environment.yml

This will create a new conda environment named *aps2s*. Activate this environment to use the *aps2s.py* script.

.. code-block:: console

    $ conda activate aps2s

.. note:: 
    Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to get *conda*.
    Alternatively you can use *mamba* in place of conda. Refer to `Mamba documentation <https://mamba.readthedocs.io/en/latest/installation.html>`_ for installing *mamba*



Usage
============

.. click:: aps2s:app_click
   :prog: aps2s
   :nested: full