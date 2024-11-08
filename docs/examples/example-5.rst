Example 5: 10X PBMC (Zheng et al., 2017) integrating ``scanpy`` pipeline
=========================================================================

The description of the data for this is example can be found in the
`GitHub repository: single-cell-3prime-paper <https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis>`_.

1. Download and unzip the `10X compressed folder "filtered genes" <http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz>`_. Save it to ``PARC/data/``.
2. You can view the target annotations here:
	 `PARC/data/zheng17_annotations.txt <https://github.com/ahill187/PARC/blob/main/data/zheng17_annotations.txt>`_.
3. Install ``scanpy``:

.. code-block:: bash

	pip install scanpy


.. code-block:: python

	import scanpy as sc
	import pandas as pd
	import pathlib

	# Set the directory by replacing {PATH/TO/PARC}
	PARC_DIR = "{PATH/TO/PARC}/PARC/"
	x_data_path = pathlib.Path(PARC_DIR, "data/filtered_matrices_mex/hg19")
	y_data_path = pathlib.Path(PARC_DIR, "data/zheng17_annotations.txt")

	# Load data
	ann_data = sc.read(f"{x_data_path}/matrix.mtx", cache=True).T  # transpose the data
	ann_data.var_names = pd.read_csv(f"{x_data_path}/genes.tsv", header=None, sep='\t')[1]
	ann_data.obs_names = pd.read_csv(f"{x_data_path}/barcodes.tsv", header=None)[0]

	# Load the annotations as per correlation with pure samples
	y_data = list(pd.read_csv(y_data_path, header=None)[0])
	ann_data.obs["annotations"] = pd.Categorical(y_data)

	# Pre-process as per Zheng et al., and take first 50 PCs for analysis
	sc.pp.recipe_zheng17(ann_data)
	sc.tl.pca(ann_data, n_comps=50)

	# Instantiate the PARC model
	parc_model = parc.PARC(
		x_data=ann_data.obsm["X_pca"],
		y_data_true=y_data,
		jac_std_factor=0.15,
		jac_threshold_type="mean",
		random_seed=1,
		small_community_size=50 # setting small_community_size to 50 cleans up some of the
		# smaller clusters, but can also be left at the default 10
	)

	# Run the PARC clustering
	parc_model.run_parc()

	# Get the predicted cell types
	y_data_pred = parc_model.y_data_pred
	ann_data.obs["PARC"] = pd.Categorical(y_data_pred)

	# Visualize UMAP
	sc.settings.n_jobs=4
	sc.pp.neighbors(ann_data, n_neighbors=10, n_pcs=40)
	sc.tl.umap(ann_data)
	sc.pl.umap(ann_data, color="annotations")
	sc.pl.umap(ann_data, color="PARC")