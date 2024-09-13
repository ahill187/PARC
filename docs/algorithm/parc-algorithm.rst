.. _parc_algorithm:

PARC Algorithm
==============

Algorithm
*********

Part 1: Hierarchical Navigable Small World (HNSW) Graph Construction
--------------------------------------------------------------------

The PARC algorithm starts by constructing a graph using the Hierarchical Navigable Small World
(HNSW) algorithm. The HNSW algorithm is an approximate k-nearest neighbors (kNN) algorithm that
is used to create a graph of approximately ``k`` nearest neighbors.
See the :ref:`hnsw_algorithm` section for more details.

Part 2: Graph Pruning
---------------------

In order to speed up the clustering process, we need to prune the graph constructed in Part 1.
This is divided into two steps:

1. ``Local Pruning``: For each node, remove any neighboring nodes which exceed a certain
   local (Euclidean) distance threshold from the node.

2. ``Global Pruning``: Reweight the edges using the ``Jaccard similarity coefficient``, which is
   used to determine the similarity between two nodes. Remove any edges with a weight below a
   certain global threshold.


Part 3: Leiden Algorithm
------------------------

.. figure:: ../_static/img/leiden-algorithm.png
    :width: 600
    :alt: Leiden Algorithm
    :align: left

The Leiden algorithm is a graph clustering algorithm that is used to find communities in a graph.
In Part 1, we constructed a graph using the HNSW algorithm, and then we pruned this graph in
Part 2. In Part 3, we will use the Leiden algorithm to find communities in the pruned graph.
See the :ref:`leiden_algorithm` section for more details.


Part 4: Large Communities
-------------------------

Once the Leiden partition has been created, we identify large communities and split them into
smaller subcommunities. This is done by running Parts 1 - 3 and Part 5 on the large communities,
and then merging the subcommunities back into the original partition.

Part 5: Small Communities
-------------------------

Sometimes the aggressive pruning of the PARC algorithm can generate small communities or singleton
communities that are not true outliers. To address this, we identify which communities are true
outliers and which are not. We then merge the ones which are not into neighboring communities.

**Step 1: Identify Small Communities:** We identify small communities by setting a threshold
for the minimum number of nodes in a community. If a community has fewer nodes than this threshold,
it is considered a small community. Default is ``10``.

**Step 2: Check HNSW Graph for Neighbors:** To determine whether or not a small community is a
true outlier, we check the HNSW graph for neighboring communities. If the small community has
no neighboring communities in the original HNSW graph which are greater than the threshold, it is
kept as a true outlier. Otherwise, it is merged into the neighboring community with the greatest
number of edges to the small community.
