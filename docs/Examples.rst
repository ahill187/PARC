=========
Examples
=========

**Example Usage 1.**

(small test sets) - IRIS and Digits dataset from sklearn::


	import parc
	import matplotlib.pyplot as plt
	from sklearn import datasets

	# load sample IRIS data
	# data dimensions (n_obs x k_dim, 150x4)
	iris = datasets.load_iris()
	x_data = iris.data
	y_data_true = iris.target

	plt.scatter(x_data[:,0], x_data[:,1], c=y_data_true) // colored by 'ground truth'
	plt.show()

	# instantiate PARC
	Parc1 = parc.PARC(x_data=x_data, y_data_true=y_data_true)
	# Use 'Parc1 = parc.PARC(x_data) ' when no 'true labels' are available
	# run the clustering
	Parc1.run_parc()
	y_data_pred = Parc1.y_data_pred
	# View scatterplot colored by PARC labels
	plt.scatter(x_data[:, 0], x_data[:, 1], c=y_data_pred, cmap='rainbow')
	plt.show()

	# Run umap on the HNSW knngraph already built in PARC (more time and memory efficient for large datasets)
	# If you choose to visualize before running PARC clustering. then you need to include this line: Parc1.knn_struct = p1.make_knn_struct()
	graph = Parc1.create_knn_graph()
	x_umap = parc.run_umap_hnsw(x_data, graph)
	plt.scatter(x_umap[:, 0], x_umap[:, 1], c=Parc1.y_data_pred)
	plt.show()


**Example Usage 2**

(mid-scale scRNA-seq): 10X PBMC (Zheng et al., 2017)
`pre-processed datafile <https://drive.google.com/file/d/1H4gOZ09haP_VPCwsYxZt4vf3hJ1GZj3b/view?usp=sharing>`_

`annotations of cells by cell type <https://github.com/ShobiStassen/PARC/blob/master/Datasets/zheng17_annotations.txt>`_::


	import parc
	import csv
	import numpy as np
	import pandas as pd

	## load data (50 PCs of filtered gene matrix pre-processed as per Zheng et al. 2017)

	x_data = csv.reader(open("./pca50_pbmc68k.txt", 'rt'),delimiter = ",")
	x_data = np.array(list(x_data)) // (n_obs x k_dim, 68579 x 50)
	x_data = x_data.astype("float")
	# OR with pandas as: x_data = pd.read_csv("'./pca50_pbmc68k.txt", header=None).values.astype("float")

	y_data_true = [] # annotations
	with open('./zheng17_annotations.txt', 'rt') as f:
	    for line in f: y_data_true.append(line.strip().replace('\"', ''))
	# OR with pandas as: y_data_true =  list(pd.read_csv('./data/zheng17_annotations.txt', header=None)[0])

	# setting small_pop to 50 cleans up some of the smaller clusters, but can also be left at the default 10
	parc1 = parc.PARC(x_data=x_data, y_data_true=y_data_true, jac_std_global=0.15, random_seed =1, small_pop = 50) // instantiate PARC
	parc1.run_parc() // run the clustering
	y_data_pred = parc1.y_data_pred

**TSNE colored by PARC clusters and cell type annotations**

.. raw:: html

  <img src="https://github.com/ShobiStassen/PARC/blob/master/Images/10X_PBMC_PARC_andGround.png?raw=true" width="500px" align="center" </a>


**Example Usage 3.**
10X PBMC (Zheng et al., 2017) Using PARC with the Scanpy pipeline

`raw datafile 68K pbmc from github page <https://github.com/10XGenomics/single-cell-3prime-paper/tree/master/pbmc68k_analysis>`_

`10X compressed file "filtered genes" <http://cf.10xgenomics.com/samples/cell-exp/1.1.0/fresh_68k_pbmc_donor_a/fresh_68k_pbmc_donor_a_filtered_gene_bc_matrices.tar.gz>`_ ::

	import scanpy.api as sc
	import pandas as pd
	# load the data
	path = './data/filtered_matrices_mex/hg19/'
	adata = sc.read(path + 'matrix.mtx', cache=True).T  # transpose the data
	adata.var_names = pd.read_csv(path + 'genes.tsv', header=None, sep='\t')[1]
	adata.obs_names = pd.read_csv(path + 'barcodes.tsv', header=None)[0]

	# annotations as per correlation with pure samples
	annotations = list(pd.read_csv('./data/zheng17_annotations.txt', header=None)[0])
	adata.obs['annotations'] = pd.Categorical(annotations)

	# pre-process as per Zheng et al., and take first 50 PCs for analysis
	sc.pp.recipe_zheng17(adata)
	sc.tl.pca(adata, n_comps=50)
	# setting small_pop to 50 cleans up some of the smaller clusters, but can also be left at the default 10
	parc1 = parc.PARC(adata.obsm['X_pca'], y_data_true=annotations, jac_std_global=0.15, random_seed =1, small_pop = 50)
	#run the clustering
	parc1.run_parc()
	y_data_pred = parc1.y_data_pred
	adata.obs["PARC"] = pd.Categorical(y_data_pred)

	//visualize
	sc.settings.n_jobs=4
	sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
	sc.tl.umap(adata)
	sc.pl.umap(adata, color='annotations')
	sc.pl.umap(adata, color='PARC')


**Example Usage 4.**

Large-scale (70K subset and 1.1M cells) Lung Cancer cells (multi-ATOM imaging cytometry based features)

`normalized image-based feature matrix 70K cells <https://drive.google.com/open?id=1LeFjxGlaoaZN9sh0nuuMFBK0bvxPiaUz>`_

`Lung Cancer cells annotation 70K cells <https://drive.google.com/open?id=1iwXQkdwEwplhZ1v0jYWnu2CHziOt_D9C>`_

`Lung Cancer Digital Spike Test of n=100 H1975 cells on N281604 <https://drive.google.com/open?id=1kWtx3j1ixua4nQt1HFHlwzCHnOr7gvKm>`_

`1.1M cell features and annotations <https://data.mendeley.com/datasets/nnbfwjvmvw/draft?a=dae895d4-25cd-4bdf-b3e4-57dd31c11e37>`_ ::


	import parc
	import pandas as pd

	# load data: digital mix of 7 cell lines from 7 sets of pure samples (1.1M cells)
	x_data = pd.read_csv("'./LungData.txt", header=None).values.astype("float")
	y_data_true = list(pd.read_csv('./LungData_annotations.txt', header=None)[0]) // list of cell-type annotations

	# run PARC on 1.1M cells
	# jac_weighted_edges can be set to false which provides an unweighted graph to leiden and offers some speedup
	parc1 = parc.PARC(x_data=x_data, y_data_true=y_data_true, jac_weighted_edges = False)
	#run the clustering
	parc1.run_parc()
	y_data_pred = parc1.y_data_pred

	# run PARC on H1975 spiked cells
	parc2 = parc.PARC(x_data=x_data, y_data_true=y_data_true, jac_std_global = 0.15, jac_weighted_edges = False) // 0.15 corresponds to pruning ~60% edges and can be effective for rarer populations than the default 'median'
	# run the clustering
	parc2.run_parc()
	y_data_pred_rare = parc2.y_data_pred

**TSNE plot of annotations and PARC clustering and heatmap of features by cluster**

.. raw:: html

  <img src="https://github.com/ShobiStassen/PARC/blob/master/Images/70K_Lung_github_overview.png?raw=true" width="500px" align="center" </a>
