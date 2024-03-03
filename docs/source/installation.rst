==============
Installation
==============

MacOS and Linux
===================

**Option 1: pip install**

.. code:: bash

  conda create --name env pip
  pip install parc


If the ``pip install`` doesn't work, it usually suffices to first install all the requirements
(using pip) and subsequently install ``parc`` (also using pip):

.. code:: bash

  pip install igraph, leidenalg, hnswlib, umap-learn
  pip install parc


**Option 2: Clone repository and run setup.py** (ensure dependencies are installed)

.. code:: bash

  git clone https://github.com/ShobiStassen/PARC.git
  cd PARC
  python3 setup.py install


Windows
===================

Once you have Visual Studio installed it should be smooth sailing (sometimes requires a restart
after intalling VS). It might be easier to install dependences using either
``pip`` or ``conda -c conda-forge install``. If this doesn't work then you might need to consider
using binaries to install ``igraph`` and ``leidenalg``.

- ``python-igraph``: Download the
  `Python 3.6 Windows binary by Gohlke <http://www.lfd.uci.edu/~gohlke/pythonlibs>`_.
- ``leidenalg``: depends on ``python-igraph``. Download the
  `leidenalg Windows binary <https://pypi.org/project/leidenalg/#files>`_
  (available for ``Python 3.6`` only)

.. code:: bash

  conda create --name env python=3.6 pip
  pip install igraph #(or install python_igraph-0.7.1.post6-cp36-cp36m-win_amd64.whl )
  pip install leidenalg #(or install leidenalg-0.7.0-cp36-cp36m-win_amd64.whl)
  pip install hnswlib
  pip install parc

**References**

- `Leiden algorithm (V.A. Traag, 2019) <doi.org/10.1038/s41598-019-41695-z>`_

- `hsnwlib <https://arxiv.org/abs/1603.09320>`_ (Malkov Yu A., and D. A. Yashunin) "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs."
