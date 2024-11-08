Example 6: Large-scale (70K subset and 1.1M cells) Lung Cancer cells (multi-ATOM imaging cytometry based features)
=====================================================================================================================

Here we have a dataset containing single cell feature data for lung cancer cells.
The data was extracted from Bright Field and QPI images taken by Multi-ATOM imaging flow cytometry.
The data is a digital mix of 7 cell lines from 7 sets of pure samples.

The full dataset is 1.1M cells; however, depending on your computer's memory capacity,
you may want to run a smaller subset, so we have provided a subset of 70K cells as well.

To run the full dataset of 1.1M cells:

1. Download the `Lung Cancer 1.1M cell features and annotations from Elsevier <https://data.mendeley.com/datasets/nnbfwjvmvw/draft?a=dae895d4-25cd-4bdf-b3e4-57dd31c11e37>`_.
	 Save the files to:

	 ``PARC/data/datamatrix_LungCancer_multiATOM_N1113369.txt``
	 ``PARC/data/true_label_LungCancer_multiATOM_N1113369.txt``

2. Download the `H1975 digital spike test cluster data (n = 100) <https://drive.google.com/open?id=1kWtx3j1ixua4nQt1HFHlwzCHnOr7gvKm>`_.
	 Save it to ``PARC/data/datamatrix_RareH1975_LC_RS209_N281604Dim24.txt``.

3. You can view the H1975 annotations (which are all 0 since it's one cluster) under ``PARC/data/true_label_RareH1975_LC_PARC_RS209_N281604.txt``.

Otherwise, you can download the 70K subset:

1. Download the input data:
	`normalized image-based feature matrix 70K cells <https://drive.google.com/open?id=1LeFjxGlaoaZN9sh0nuuMFBK0bvxPiaUz>`_.
	 Save the file to ``PARC/data/datamatrix_LC_PARC__N70000.txt``

2. Download the target data (annotations):

	`Lung Cancer cells annotation 70K cells <https://drive.google.com/open?id=1iwXQkdwEwplhZ1v0jYWnu2CHziOt_D9C>`_.
	 Save the file to ``PARC/data/true_label_LC_PARC_N70000.txt``

3. Download the `H1975 digital spike test cluster data (n = 100) <https://drive.google.com/open?id=1kWtx3j1ixua4nQt1HFHlwzCHnOr7gvKm>`_.
	 Save it to ``PARC/data/datamatrix_RareH1975_LC_RS209_N281604Dim24.txt``.

4. You can view the H1975 annotations (which are all 0 since it's one cluster) under ``PARC/data/true_label_RareH1975_LC_PARC_RS209_N281604.txt``.


.. code-block:: python

	import parc
	import pandas as pd
	import pathlib

	# Set the directory by replacing {PATH/TO/PARC}
	PARC_DIR = "{PATH/TO/PARC}/PARC/"

	# Load the full dataset of 1.1M cells
	x_data_path = pathlib.Path(PARC_DIR, "data/datamatrix_LungCancer_multiATOM_N1113369.txt")
	y_data_path = pathlib.Path(PARC_DIR, "data/true_label_LungCancer_multiATOM_N1113369.txt")

	# # Alternatively, load the subset of 70K cells
	# x_data_path = pathlib.Path(PARC_DIR, "data/datamatrix_LC_PARC__N70000.txt")
	# y_data_path = pathlib.Path(PARC_DIR, "data/true_label_LC_PARC_N70000.txt")

	# Load data
	x_data = pd.read_csv(x_data_path, header=None).values.astype("float")
	y_data = list(pd.read_csv(y_data_path, header=None)[0])  # list of cell-type annotations

	# Instantiate PARC with the lung cancer data
	parc_model = parc.PARC(
		x_data=x_data,
		y_data_true=y_data,
		jac_weighted_edges=False  # provides unweighted graph to leidenalg (faster)
	)

	# Run the PARC clustering
	parc_model.run_parc()
	y_data_pred = parc_model.y_data_pred

	# Load the H1975 cell cluster (n = 100)
	x_data_path = pathlib.Path(PARC_DIR, "data/datamatrix_RareH1975_LC_RS209_N281604Dim24.txt")
	y_data_path = pathlib.Path(PARC_DIR, "data/true_label_RareH1975_LC_PARC_RS209_N281604.txt")
	x_data = pd.read_csv(x_data_path, header=None).values.astype("float")
	y_data = list(pd.read_csv(y_data_path, header=None)[0])

	# Instantiate PARC with the H1975 spiked cells
	parc_model = parc.PARC(
		x_data=x_data,
		y_data_true=y_data,
		jac_std_factor=0.15,  # 0.15 prunes ~60% edges and can be effective for rarer populations
		jac_threshold_type="mean",
		jac_weighted_edges=False
	)

	# Run the PARC clustering
	parc_model.run_parc()
	parc_labels_rare = parc_model.y_data_pred


.. image:: ../_static/img/70K_Lung_github_overview.png
	:width: 600
	:alt: t-SNE plot of annotations and PARC clustering, heatmap of features

