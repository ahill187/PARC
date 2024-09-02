Example 4: (mid-scale scRNA-seq): 10X PBMC (Zheng et al., 2017)
=================================================================

1. Download the input data file and save it to the ``PARC/data/`` directory:
	 `pca50_pbmc68k.txt <https://drive.google.com/file/d/1H4gOZ09haP_VPCwsYxZt4vf3hJ1GZj3b/view?usp=sharing>`_.
2. You can view the target annotations here: `PARC/data/zheng17_annotations.txt <https:://github.com/ahill187/PARC/blob/main/data/zheng17_annotations.txt>`_.

.. code-block:: python

	import pathlib
	import parc
	import numpy as np
	import pandas as pd

	# Set the directory by replacing {PATH/TO/PARC}

	PARC_DIR = "{PATH/TO/PARC}/PARC/"
	x_data_path = pathlib.Path(PARC_DIR, "data/pca50_pbmc68k.txt")
	y_data_path = pathlib.Path(PARC_DIR, "data/zheng17_annotations.txt")

	# Load data
	# 50 PCs of filtered gene matrix pre-processed as per Zheng et al. 2017)
	# (n_samples x n_features) = (68579 x 50)
	x_data = pd.read_csv(x_data_path, header=None).values.astype("float")
	y_data = list(pd.read_csv(y_data_path, header=None)[0])

	# Instantiate the PARC model
	parc_model = parc.PARC(
			x_data=x_data,
			y_data_true=y_data,
			jac_std_factor=0.15,
			jac_threshold_type="mean",
			random_seed=1,
			small_community_size=50 # setting small_community_size = 50
			# cleans up some of the smaller clusters, but can also be left at the default 10
	)

	# Run the PARC clustering
	parc_model.run_parc()
	y_data_pred = parc_model.y_data_pred

	# View the model performance
	parc_model.stats_df


.. image:: ../_static/img/10X_PBMC_PARC_andGround.png
	:width: 600
	:alt: t-SNE plot of annotations and PARC clustering