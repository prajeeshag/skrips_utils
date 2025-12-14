.. _installation:

Installation
============
Clone the github repository

.. code-block:: console

    $ git clone https://github.com/prajeeshag/skrips_utils.git
    $ cd skrips_utils
    $ mamba env create -f environment.yml
    $ pip install .



This will create a new conda environment named *skrips*. Activate this environment to use the cli utilities.

.. code-block:: console

    $ mamba activate skrips
    

.. note:: 
    Install `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ to get *conda*.
    Alternatively you can use *mamba* in place of conda. Refer to `Mamba documentation <https://mamba.readthedocs.io/en/latest/installation.html>`_ for installing *mamba*



Usage
============

.. click:: skrips_utils.mkMITgcmBC:app_click
   :prog: mkMITgcmBC
   :nested: full

.. click:: skrips_utils.mkMITgcmIC:app_click
   :prog: mkMITgcmIC
   :nested: full

.. click:: skrips_utils.diffNML:app_click
   :prog: diffNML
   :nested: full