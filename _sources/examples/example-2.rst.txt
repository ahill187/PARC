Example 2: Iris Dataset from ``sklearn``
=========================================

.. code-block:: python

	import parc
	import pandas as pd
	import matplotlib.pyplot as plt
	from sklearn import datasets

	# Load the Iris dataset
	iris = datasets.load_iris()
	x_data = pd.DataFrame(iris.data, columns=iris.feature_names)  # (n_samples x n_features = 150 x 4)
	y_data = iris.target

	# Plot the data (coloured by ground truth)
	column_a = iris.feature_names[0]
	column_b = iris.feature_names[1]
	plt.scatter(x_data[column_a], x_data[column_b], c=y_data)
	plt.title("Iris Dataset: Ground Truth")
	plt.xlabel(column_a)
	plt.ylabel(column_b)
	plt.show()

	# Instantiate the PARC model
	parc_model = parc.PARC(x_data=x_data, y_data_true=y_data)

	# Run the PARC clustering
	parc_model.run_parc()
	y_data_pred = parc_model.y_data_pred

	# View scatterplot colored by PARC labels
	plt.scatter(x_data[column_a], x_data[column_b], c=y_data_pred, cmap="rainbow")
	plt.title("Iris Dataset: PARC Predictions")
	plt.xlabel(column_a)
	plt.ylabel(column_b)
	plt.show()

	# Run UMAP on the HNSW knngraph already built in PARC
	# (more time and memory efficient for large datasets)
	csr_array = parc_model.create_knn_graph()
	x_umap = parc_model.run_umap_hnsw(x_data=x_data, graph=csr_array)

	# Visualize UMAP results
	plt.scatter(x_umap[:, 0], x_umap[:, 1], c=parc_model.y_data_pred)
	plt.title("Iris Dataset UMAP: PARC Predictions")
	plt.xlabel("umap_x")
	plt.ylabel("umap_y")
	plt.show()