Installation
==============


MacOS / Linux
*************

.. code-block:: bash

  git clone https://github.com/ahill187/PARC.git
  cd PARC
  python3 -m venv env
  source env/bin/activate
  pip install .


.. note::

  If the ``pip install`` doesn't work, it usually suffices to first install all the requirements
  (using pip) and subsequently install ``PARC`` (also using pip), i.e.

  .. code-block:: bash

    git clone https://github.com/ahill187/PARC.git
    cd PARC
    python3 -m venv env
    source env/bin/activate
    pip install igraph, leidenalg, hnswlib, umap-learn
    pip install .


Windows
********

Once you have Visual Studio installed it should be smooth sailing
(sometimes requires a restart after intalling VS). It might be easier to install dependences using
either ``pip install`` or ``conda -c conda-forge install``. If this doesn't work then you might need
to consider using binaries to install ``igraph`` and ``leidenalg``.

* ``python-igraph``: Download the `Python 3.6 Windows Binaries by Gohlke <http://www.lfd.uci.edu/~gohlke/pythonlibs/#python-igraph>`_.
* ``leidenalg``: Download the `Python 3.6 Windows binary <https://pypi.org/project/leidenalg/#files>`_.

.. code-block:: bash

  conda create --name parcEnv python=3.6 pip
  pip install igraph  # or install python_igraph-0.7.1.post6-cp36-cp36m-win_amd64.whl
  pip install leidenalg  # or install leidenalg-0.7.0-cp36-cp36m-win_amd64.whl
  pip install hnswlib
  pip install parc


Developers
**********

If you want to develop this package, please see the installation instructions for
developers: :doc:`Developer Installation <../dev/dev-installation>`.


References
**********

- `Leiden algorithm (V.A. Traag, 2019) <doi.org/10.1038/s41598-019-41695-z>`_

- `hsnwlib <https://arxiv.org/abs/1603.09320>`_ (Malkov Yu A., and D. A. Yashunin)
  "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small
  World graphs."  




