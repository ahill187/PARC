.. _leiden_algorithm:

Leiden Algorithm
================

The Leiden algorithm was developed to improve upon the Louvain method by addressing the issue of
disconnected communities.

Algorithm
*********

.. figure:: ../_static/img/leiden-algorithm.png
    :width: 600
    :alt: Leiden Algorithm
    :align: left

    **Leiden Algorithm:** The Leiden algorithm starts from a singleton partition (a).
    The algorithm moves individual nodes from one community to another to find a partition (b),
    which is then refined (c). An aggregate network (d) is created based on the refined partition,
    using the non-refined partition to create an initial partition for the aggregate network.
    For example, the red community in (b) is refined into two subcommunities in (c), which after
    aggregation become two separate nodes in (d), both belonging to the same community.
    The algorithm then moves individual nodes in the aggregate network (e). In this case,
    refinement does not change the partition (f). These steps are repeated until no further
    improvements can be made :cite:`leiden2019`.


Step 1: Assign Nodes to Communities
------------------------------------

First, each node is assigned to its own community, consisting of a single node. For the first
iteration, this means that each individual data point is its own community.

Step 2: Calculate Quality and Reassign
---------------------------------------

Next, we decide which communities to move the nodes into and update the partition 
:math:`\mathcal{P}`.

.. code-block:: python

    def compute_delta_H(node, C, partition):
        # compute the change in quality for a node and a community
        pass

    queue = vertices # create a queue from the nodes

    while queue is not None:
        node = queue.next() # get the next node
        delta_H = 0
        for C in communities: # compute the change in quality for each community
            delta_H_community = compute_delta_H(node, C, partition)
            if delta_H_community > delta_H:
                delta_H = delta_H_community
                community = C
        if delta_H > 0:
            # move node to community
            community.add(node) 
            # find the nodes which are connected to the node but not in the community
            outside_nodes = []
            for edge in edges:
                if edge[0] == node and edge[1] not in community:
                    outside_nodes.append(edge[1])
                elif edge[1] == node and edge[0] not in community:
                    outside_nodes.append(edge[0])
            # add the outside nodes to the queue
            queue.add([node for node in outside_nodes if node not in queue])


Step 3: Assign Nodes to Refined Partition
-----------------------------------------

Similarly to ``Step 1``, we assign each node in the graph to its own community in a new partition
called :math:`\mathcal{P}_{\text{refined}}`.

Step 4: 
--------


.. code-block:: python

    def get_connected_edges(edges, community_origin, community_target):
        """Get the edges which are connected from community_origin to community_target."""

        connected_edges = [
            edge for edge in edges if edge[0] in community_origin
            and edge[1] in community_target
            and edge[1] not in community_origin
        ]
        return connected_edges


    def check_if_communities_well_connected(
        connected_edges, gamma, community_origin, community_target
    ):
        # check if the communities are well connected
        is_well_connected = len(connected_edges) >= (
            gamma *
            community_origin.recursive_size * 
            (community_target.recursive_size - community_origin.recursive_size)
        )
        return is_well_connected


    # iterate over the communities from partition from Step 2
    for community in partition:
        # find the nodes in the community which have lots of edges within the community
        well_connected_nodes = []
        for node in community:
            # create a community with only the node
            community_origin = Community([node])
            # find the edges which are connected from the singleton community to the community
            connected_edges = get_connected_edges(edges, community_origin, community)
            # check if the number of connected edges is greater than a threshold indicating
            # that the node is well connected to the community
            is_well_connected = check_if_communities_well_connected(
                connected_edges, gamma, community_origin, community
            )
            if is_well_connected:
                well_connected_nodes.append(node)

        for node in well_connected_nodes:
            # if the node is a singleton in the refined partition:
            if partition_refined.get_node(node).community.size == 1:
                well_connected_communities = []
                for community_ref in partition_refined:
                    # check if the community is a subset of the community from Step 2
                    is_a_subset = np.all([node in community for node in community_ref])
                    connected_edges = get_connected_edges(edges, community_ref, community)
                    is_well_connected = check_if_communities_well_connected(
                        connected_edges, gamma, community_ref, community
                    )
                    if is_well_connected and is_a_subset:
                        well_connected_communities.append(community_ref)

                prob_distribution = []
                for community_ref in well_connected_communities:
                    delta_H = compute_delta_H(node, community_ref, partition_refined)
                    if delta_H < 0:
                        # if the quality decreases, assign a weight of 0
                        prob_distribution.append(0)
                    else:
                        # assign a weight based on the quality increase
                        prob_distribution.append(np.exp(1 / theta * delta_H))
                    # assign the node to a community based on the probability distribution
                    community_new = np.random.choice(
                        well_connected_communities, p=prob_distribution / np.sum(prob_distribution)
                    )
                    community_new.add(node)


Step 5: Aggregate the Graph
---------------------------

Use the refined partition :math:`\mathcal{P}_{\text{refined}}` to aggregate the graph.
Each community in :math:`\mathcal{P}_{\text{refined}}` becomes a node in the new graph
:math:`G_{\text{agg}}`.

**Example:**

Suppose that we have:

.. math::

    V &= \{v_1, v_2, v_3, v_4, v_5, v_6, v_7\} \\
    C_1 &= \{v_1, v_2, v_3, v_4\} \\
    C_2 &= \{v_5, v_6, v_7\} \\
    \mathcal{P} &= \{C_1, C_2\} \\
    \mathcal{P}_{\text{refined}} &= \{C_{1a}, C_{1b}, C_2\}

Then our new set of nodes will be:

.. math::

    V_{\text{agg}} = \{C_{1a} \mapsto w_{1a}, C_{1b} \mapsto w_{1b}, C_2 \mapsto w_2\}


Step 6: Update the Partition
----------------------------

Update the partition :math:`\mathcal{P}` using the aggregated graph. We keep the communities
from partition :math:`\mathcal{P}`, but the communities can be separated into multiple nodes from
the refined partition :math:`\mathcal{P}_{\text{refined}}`:

.. math::
    
    \mathcal{P} = \{\{v ~|~ v \subseteq C, v \in V(G_{agg})\} ~|~ C \in \mathcal{P} \}

**Example:**

Suppose that :math:`C` is a poorly-connected community from the partition
:math:`\mathcal{P}`:

.. math::

    C &= \{v_1, v_2, v_3, v_4, v_5\} \\
    \mathcal{P} &= \{C\}

Then suppose during the refinement step, it was separated into two communities, 
:math:`C_1` and :math:`C_2`:

.. math::

    C_1 &= \{v_1, v_2, v_3\} \\
    C_2 &= \{v_4, v_5\} \\
    \mathcal{P}_{\text{refined}} &= \{C_1, C_2\}

When we aggregate the graph, the new nodes will be:

.. math::

    V(G_{\text{agg}}) = \{C_1, C_2\}

but we will keep the old partition:

.. math::

    \mathcal{P} = \{\{C_1, C_2\}\}

Step 7: Repeat Steps 2 - 6
--------------------------

Repeat Steps 2 - 6 until each community consists of only one node.

