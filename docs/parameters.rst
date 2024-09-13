Parameters and Attributes
==========================

For a more detailed explanation of the impact of tuning key parameters please see the Supplementary
Analysis in our paper:
`PARC Supplementary Analysis <https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/PAP/10.1093_bioinformatics_btaa042/1/btaa042_supplementary-data.pdf?Expires=1583098421&Signature=R1gJB7MebQjg7t9Mp-MoaHdubyyke4OcoevEK5817el27onwA7TlU-~u7Ug1nOUFND2C8cTnwBle7uSHikx7BJ~SOAo6xUeniePrCIzQBi96MvtoL674C8Pd47a4SAcHqrA2R1XMLnhkv6M8RV0eWS-4fnTPnp~lnrGWV5~mdrvImwtqKkOyEVeHyt1Iajeb1W8Msuh0I2y6QXlLDU9mhuwBvJyQ5bV8sD9C-NbdlLZugc4LMqngbr5BX7AYNJxvhVZMSKKl4aMnIf4uMv4aWjFBYXTGwlIKCjurM2GcHK~i~yzpi-1BMYreyMYnyuYHi05I9~aLJfHo~Qd3Ux2VVQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA>`_

Parameters
**********

.. raw:: html

  <table class="table table-responsive">
    <thead>
      <tr>
          <th>Parameter</th>
          <th>Default</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
          <td><code class="notranslate">knn</code></td>
          <td><code class="notranslate">30</code></td>
          <td>The number of nearest neighbors for the KNN algorithm.
          Larger k means more neighbors in a cluster and therefore less clusters.</td>
      </tr>
      <tr>
          <td><code class="notranslate">n_iter_leiden</code></td>
          <td><code class="notranslate">5</code></td>
          <td>The number of iterations for the Leiden algorithm.</td>
      </tr>
      <tr>
          <td><code class="notranslate">random_seed</code></td>
          <td><code class="notranslate">42</code></td>
          <td>The random seed to enable reproducible Leiden clustering.</td>
      </tr>
      <tr>
          <td><code class="notranslate">distance_metric</code></td>
          <td><code class="notranslate">"l2"</code></td>
          <td>The distance metric to be used in the KNN algorithm. One of:
          <ul>
          <li><code class="notranslate">"l2"</code>: Euclidean distance L2 norm<br>
              <code class="snippet">d = sum((x_i - y_i)^2)</code>
          </li>
          <li><code class="notranslate">"cosine"</code>: cosine similarity<br>
              <code class="snippet">
                d = 1.0 - sum(x_i*y_i) / sqrt(sum(x_i*x_i) * sum(y_i*y_i))
              </code>
          </li>
          <li><code class="notranslate">"ip"</code>: inner product distance <br>
              <code class="snippet">d = 1.0 - sum(x_i*y_i)</code>
          </li>
          </ul>
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">n_threads</code></td>
          <td><code class="notranslate">-1</code></td>
          <td>The number of threads used in the KNN algorithm.</td>
      </tr>
      <tr>
          <td><code class="notranslate">hnsw_param_ef_construction</code></td>
          <td><code class="notranslate">150</code></td>
          <td>
            A higher value increases accuracy of index construction.
            Even for O(100 000) cells, 150-200 is adequate.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">l2_std_factor</code></td>
          <td><code class="notranslate">3.0</code></td>
          <td>
            The multiplier used in calculating the Euclidean distance threshold for the
            distance between two nodes during local pruning:<br><br>
            <code class="snippet">
              max_distance = np.mean(distances) + l2_std_factor * np.std(distances)
            </code>
            <br><br>
            Avoid setting both the <code class="notranslate">jac_std_factor</code> (global) and the
            <code class="notranslate">l2_std_factor</code> (local) to < 0.5 as this is very
            aggressive pruning. Higher <code class="notranslate">l2_std_factor</code> means more
            edges are kept.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">do_prune_local</code></td>
          <td><code class="notranslate">None</code></td>
          <td>
            Whether or not to do local pruning. If <code class="notranslate">None</code> (default),
            set to <code class="notranslate">False</code> if the number of samples is > 300 000,
            and set to <code class="notranslate">True</code> otherwise.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">jac_threshold_type</code></td>
          <td><code class="notranslate">"median"</code></td>
          <td>
            One of <code class="notranslate">median</code> or <code class="notranslate">mean</code>.
            Determines how the Jaccard similarity threshold is calculated during global pruning.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">jac_std_factor</code></td>
          <td><code class="notranslate">0.15</code></td>
          <td>
            The multiplier used in calculating the Jaccard similarity threshold for the similarity
            between two nodes during global pruning for
            <code class="notranslate">jac_threshold_type="mean"</code>:
            <br><br>
            <code class="snippet">
              threshold = np.mean(similarities) - jac_std_factor * np.std(similarities)
            </code>
            <br><br>
            Setting <code class="notranslate">jac_std_factor=0.15</code> and
            <code class="notranslate">jac_threshold_type="mean"</code> performs empirically similar
            to <code class="notranslate">jac_threshold_type="median"</code>, which does
            not use the <code class="notranslate">jac_std_factor</code>. Generally values between
            0-1.5 are reasonable. Higher <code class="notranslate">jac_std_factor</code> means more
            edges are kept.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">jac_weighted_edges</code></td>
          <td><code class="notranslate">True</code></td>
          <td>Whether to partition using the weighted graph.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">resolution_parameter</code></td>
          <td><code class="notranslate">1.0</code></td>
          <td>The resolution parameter to be used in the Leiden algorithm.
          In order to change 'resolution_parameter', we switch to 'RBVP'.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">partition_type</code></td>
          <td><code class="notranslate">"ModularityVP"</code></td>
          <td>The partition type to be used in the Leiden algorithm:
          <ul>
          <li>
            <code class="notranslate">ModularityVP</code>:
            ModularityVertexPartition, resolution_parameter=1
          </li>
          <li>
            <code class="notranslate">RBVP</code>:
            RBConfigurationVP, Reichardt and Bornholdt's Potts model.
            Note that this is the same as <code class="notranslate">ModularityVP</code>
            when setting 𝛾 = 1 and normalising by 2m.
          </li>
          </ul> 
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">large_community_factor</code></td>
          <td><code class="notranslate">0.4</code></td>
          <td>
            A factor used to determine if a community is too large. If the community size is
            greater than <code class="snippet">large_community_factor * n_samples</code>,
            then the community is too large and the PARC algorithm will be run on the single
            community to split it up. The default value of <code class="notranslate">0.4</code>
            ensures that all communities will be less than the cutoff size.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">small_community_size</code></td>
          <td><code class="notranslate">100</code></td>
          <td>The smallest community size to be considered a community.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">small_community_timeout</code></td>
          <td><code class="notranslate">15</code></td>
          <td>The maximum number of seconds trying to check an outlying small community.
          </td>
      </tr>
    </tbody>
  </table>


Attributes
**********

.. raw:: html

  <table class="table table-responsive">
    <thead>
      <tr>
          <th>Attribute</th>
          <th>Default</th>
          <th>Description</th>
      </tr>
    </thead>
    <tbody>
      <tr>
          <td><code class="notranslate">y_data_pred</code></td>
          <td><code class="notranslate">None</code></td>
          <td>
            An array of the predicted output y labels, with dimensions <code>(n_samples, 1)</code>.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">f1_score</code></td>
          <td><code class="notranslate">None</code></td>
          <td>
            f1 score (not weighted by population). For details see supplementary section of the
            <a href="https://doi.org/10.1101/765628" target="_blank">PARC paper</a>.
          </td>
      </tr>
      <tr>
          <td><code class="notranslate">stats_df</code></td>
          <td><code class="notranslate">None</code></td>
          <td>A dataframe that stores parameter values and performance metrics</td>
      </tr>
    </tbody>
  </table>



   
  


