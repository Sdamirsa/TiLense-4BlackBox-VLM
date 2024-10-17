

# TiLense-4BlackBox-VLM
TiLense-4BlackBox-VLM is a tool designed to visualize and interpret the inner workings of vision-language models (VLMs). By generating heatmaps that highlight important image regions (tiles) based on model prediction behavior. 

[![GitHub stars](https://img.shields.io/github/stars/sdamirsa/TiLense-4BlackBox-VLM?style=social)](https://github.com/sdamirsa/TiLense-4BlackBox-VLM/stargazers)

----

### Approach 1: XAI-lense (base)
This method aims to identify and visualize important image tiles in a vision-language task by calculating the importance of individual tiles (masked regions) based on the most common answer across multiple prediction runs for each image. Unlike more complex approaches, this method does not consider variations in the original image's answers but relies on a single prevalent (base) answer. The method consists of two main components: determining the prevalent answer and calculating the tile importance, followed by generating a heatmap to visualize these critical regions.

This model-agnostic approach provides a computationally efficient way to highlight critical regions in vision-language model predictions by comparing tile-based predictions to a single base answer. By visualizing the regions with the greatest deviations, this method offers insights into the areas of an image that most strongly influence model predictions, aiding in model evaluation and refinement.


<details>
<summary>Methodology and Mathematical Representation</summary>

#### 1. Identifying Prevalent Answers

The first step in this approach is to establish a baseline prediction, or the most common answer (prevalent answer) for each image across multiple runs. This base answer is used as a reference point to measure how much the model's predictions for individual tiles differ from the expected outcome.

Let $N$ represent the total number of prediction runs for a given image. For each image $i$, the base answer $A_i$ is defined as the most frequently occurring prediction across all runs:

$`
A_i = \text{mode}(\{P_{i}^{(1)}, P_{i}^{(2)}, \dots, P_{i}^{(N)}\})
`$

where $P_{i}^{(r)}$ is the model's predicted answer for image $i$ in run $r$. In cases where multiple answers are equally common (i.e., ties), the image is flagged for manual review, and its tiles are not used in the subsequent calculations.

#### 2. Tile Importance Calculation

After determining the base answer $A_i$ for each image, the importance of each masked region (or tile) is calculated by comparing the model's predictions for that tile to the base answer. The more frequently the prediction for a tile differs from the base answer, the higher the importance assigned to that tile.

For each tile $j$ of image $i$, let $P_{i,j}^{(r)}$ represent the model's predicted answer for the tile in run $r$. The mask importance score $M_{i,j}$ for tile $j$ is computed as the count of runs in which the tile's predicted answer differs from the base answer:

$`
M_{i,j} = \sum_{r=1}^{N} \mathbf{1}(P_{i,j}^{(r)} \neq A_i)
`$

where $\mathbf{1}(\cdot)$ is an indicator function that returns 1 if the condition is true (i.e., the tile's predicted answer does not match the base answer). This score indicates how much the predictions for the tile deviate from the expected outcome.

The importance score is then normalized by dividing by the total number of runs $N$, yielding the relative importance of the tile:

$`
I_{i,j} = \frac{M_{i,j}}{N}
`$

The normalized importance score $I_{i,j}$ reflects the degree to which the tile influences the model's overall prediction, with higher values indicating tiles that significantly alter the model's output.

#### 3. Data Processing and Output

##### Data Collection and Preprocessing

The dataset used in this analysis includes model predictions and associated metadata collected across $N$ prediction runs, stored in an Excel file. The dataset contains columns such as `original_filename` (the image's identifier), `answer_clean` (the model’s predicted answer), and `image_idx` (indicating whether a row corresponds to an original image or a masked image). 

The key steps are:

- **Identifying Prevalent Answers**: The most frequent answer for each original image across all runs is computed and stored as the base answer.
- **Tile Importance Calculation**: Predictions for each tile are compared to the base answer, and the importance score is calculated based on the number of mismatches.

##### Heatmap Overlay Creation

To visualize the results, a heatmap overlay is generated, highlighting the important regions of each image. The key steps in generating the heatmap are as follows:

1. **Image Loading**: The original image is loaded and converted into an `RGBA` format to support transparency.
   
2. **Overlay Creation**: A transparent overlay is generated, matching the dimensions of the original image. Each tile is represented by a rectangular region on this overlay, where the opacity is proportional to the normalized importance score $I_{i,j}$. Higher scores correspond to more opaque (darker red) regions, while lower scores result in more transparent regions.
   
3. **Image Composition**: The overlay is then composited onto the original image, resulting in a heatmap that highlights important tiles with semi-transparent red overlays. Non-important tiles are either fully transparent or have minimal opacity.
   
4. **Saving and Displaying Heatmaps**: The heatmap images are saved as `.PNG` or `.PDF` files and can also be displayed using visualization libraries such as `matplotlib` for inspection and analysis.

### Mathematical Representation

Let $A_i$ denote the base answer for image $i$, and $P_{i,j}^{(r)}$ represent the model's predicted answer for tile $j$ of image $i$ during run $r$. The mask importance score $M_{i,j}$ for tile $j$ is calculated as:

$`
M_{i,j} = \sum_{r=1}^{N} \mathbf{1}(P_{i,j}^{(r)} \neq A_i)
`$

where $N$ is the total number of runs, and $\mathbf{1}(\cdot)$ is the indicator function that returns 1 when the prediction $P_{i,j}^{(r)}$ differs from the base answer $A_i$.

The normalized tile importance score is then:

$`
I_{i,j} = \frac{M_{i,j}}{N}
`$

This normalized importance score $I_{i,j}$ determines the transparency level of the red overlay on the heatmap, with higher scores corresponding to greater opacity.

#### Summary

To summarize, the process of mask importance calculation involves:

1. **Answer Consistency**: Computing the frequency distribution of model predictions for each unmasked image.
2. **Mask Importance**: For each masked region, calculate the deviation of the masked prediction from the original answer distribution.
3. **Normalization**: Adjusting mask importance scores using the answer consistency from the unmasked image.
4. **Visualization**: Showing the important tiles on the image overlay.

</details>





  
---

### Approach 2: XAI-lense with ConNorm: XXAI lense with Consistency (in the original image) Normalization

This section describes the procedure used to compute tile importance in a vision-language model prediction task, considering the consistency to the original image. The method consists of two main stages: (1) calculating answer consistency across multiple model runs and (2) determining the importance of specific masked regions in images, normalized by prediction consistency.

<details>
<summary>Methodology and Mathematical Representation</summary>

#### 1. Answer Consistency Calculation

To assess the model's prediction consistency, we calculate the frequency of different answers provided by the model when viewing unmasked versions of the images over multiple runs. This step establishes a baseline understanding of how often certain predictions are made for each image, which is later used to normalize mask importance.

Let $\mathcal{D}$ be the dataset consisting of multiple model runs, where each run produces predictions for a set of images. Each image is identified by its filename, denoted as $f \in \mathcal{F}$, where $\mathcal{F}$ is the set of all image filenames. For each filename $f$, the model produces a set of predictions $\{a_1, a_2, \dots, a_k\}$ for that image across different runs.

We define the answer consistency for each image as the frequency distribution of the model’s predictions. The number of occurrences of each unique prediction $a$ for image $f$ is given by:
$`
C_f(a) = \sum_{i=1}^{n_f} \mathbb{I}(a_i = a)
`$
where $n_f$ is the total number of predictions for image $f$, $\mathbb{I}$ is the indicator function, and $a_i$ is the prediction in the $i$-th run.

The total number of predictions for each image is:
$`
T_f = \sum_{a} C_f(a)
`$
This count is used later in mask importance normalization.

#### 2. Mask Importance Calculation with Consistency Normalization

In this step, we compute the importance of each masked region of an image. For each image, the model makes predictions with different regions masked, and the goal is to quantify how much each masked region influences the prediction. The importance is normalized by the consistency of predictions from the unmasked image.

Let $M \in \mathcal{M}$ denote a specific masked region applied to an image $f$. The model's prediction for the masked image is denoted as $a_M$. The importance of mask $M$ for image $f$, denoted as $I_f(M)$, is computed based on how much the prediction for the masked image deviates from the most common predictions for the unmasked image.

The importance score is given by:
$`
I_f(M) = 1 - \frac{C_f(a_M)}{T_f}
`$
where $C_f(a_M)$ is the count of how often the answer $a_M$ has been predicted for the unmasked image, and $T_f$ is the total number of predictions for the unmasked image, as defined earlier. If the prediction for the masked image $a_M$ is uncommon compared to the original image’s predictions, the importance score will be higher, indicating that the masked region significantly influenced the model’s output.

##### Mask Importance Aggregation
The final importance score for each image is obtained by summing the individual mask importance scores over all masked regions. Let $\mathcal{M}_f$ represent the set of all masked regions applied to image $f$:
$`
I_f = \sum_{M \in \mathcal{M}_f} I_f(M)
`$
This score indicates the overall importance of masked regions for an image and helps identify which parts of the image are most influential in driving the model’s predictions.

#### 3. Data Processing and Output

After computing the importance scores for all images, the results are stored in a structured format. The final output is a DataFrame that contains each image’s path, its associated mask importance score, and the corresponding masked regions. This information is saved as an Excel file for further analysis.

##### Heatmap overlay

The heatmap visualizes important regions in images based on mask importance scores. Each masked region (tile) is mapped to a corresponding location in the heatmap and superimposed on the original image. Tiles with high importance scores, indicating significant changes in model predictions when masked, are highlighted in red, while non-important tiles are fully transparent. The transparency of each tile is proportional to its importance score, with higher importance resulting in more opacity.

#### Summary

To summarize, the process of mask importance calculation involves:

1. **Base answer**: Finding the base answer (i.e, most prevalent, most consistent)
2. **Mask Importance**: For each masked region, calculate the deviation of the masked prediction from the original answer (1 if all N runs are different, 0 if all N runs are similar to base answer).
3. **Visualization**: Showing the important tiles on the image overlay.

</details>

### Approach 3: XAI-lense with ConNormLog: XXAI lense with Consistency Normalization and Logprob of VLM
...loading
