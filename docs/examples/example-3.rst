Example 3: Digits Dataset from ``sklearn``
===========================================

.. code-block:: python

	import parc
	import matplotlib.pyplot as plt
	from sklearn import datasets

	# Load the Digits dataset
	digits = datasets.load_digits()
	x_data = digits.data  # (n_samples x n_features = 1797 x 64)
	y_data = digits.target

	# Insantiate the PARC model
	parc_model = parc.PARC(
	    x_data=x_data,
	    y_data_true=y_data,
	    jac_threshold_type="median"  # "median" is default pruning level
	)

	# Run the PARC clustering
	parc_model.run_parc()
	y_data_pred = parc_model.y_data_pred