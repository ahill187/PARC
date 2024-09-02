Testing
=======

To run the test suite, from the ``PARC`` root directory, type in a shell terminal:

.. code:: bash

  cd PARC
  pytest tests/

To run a specific test file:

.. code:: bash

  pytest tests/test_parc.py


To run a specific test in a file, use the ``-k`` flag:

.. code:: bash

  pytest tests/test_parc.py -k test_parc_get_leiden_partition


Doctests
********

Doctests are examples within the function and class docstrings. To run these examples:

.. code:: bash

  cd PARC
  pytest parc/ --doctest-modules