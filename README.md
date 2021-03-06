# Galaxy Zoo Builder: Morphological Dependence of Spiral Galaxy Pitch Angle

This repository provides supporting code and data for the paper *"Galaxy Zoo Builder: Morphological Dependence of Spiral Galaxy Pitch Angle"*.

## Motivation

Exploiting the results of the *Galaxy Builder* project ([Lingard et al. 2020]((https://ui.adsabs.harvard.edu/abs/2020arXiv200610450L/abstract))), this work uses hand-drawn annotations from citizen scientists to measure the pitch angles of arms in spiral galaxies.

We wish to use these measurements to examine the links between pitch angle and morphology, and test the simple model of spiral winding proposed by [Pringle and Dobbs (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.1470P/abstract).

The full method is documented in our paper (link to be added when published), but we provide a summary below. 

## Intro to Galaxy Builder
Volunteers were presented with an image of a spiral galaxy, selected using redshift and morphology obtained through Galaxy Zoo 2. The full sample selection is detailed in Hart et al. (2017), but only a subset of these galaxies were classified in *Galaxy Builder*, prioritising low-redshift well-resolved images. The resulting distribution of galaxy stellar mass is therefore not truly representative of the population:

![Scatter plot showing stellar mass against redshift for the full galaxy sample in Hart et al. (2017), with the galaxies in Galaxy Builder highlighted](./plots/stellar_mass_selection_plot.png).

Volunteers worked through the modelling of a galaxy in stages (disc, bulge, bar and then spiral arms), with spirals being drawn using any number of freehand poly-lines:

![Four-panel image showing the process of modelling a galaxy in Galaxy Builder, light is subtracted from a galaxy in steps: first the disc, then bulge, bar and finally spiral arms](./plots/galaxy_builder_interface.jpg).

Resulting in a dataset of 30 "models" for each galaxy, which can then be aggregated using unsupervised clustering techniques.

![](./plots/drawn_shapes.pdf)

Spiral arms were identified, outlying points and points in the centre of the galaxy removed, and the output "Arm" objects have been saved **ADD file path**. The data analysis and reduction was performed using the python package `gzbuilder_analysis`, available on GitHub at [tingard/gzbuilder_analysis](https://github.com/tingard/gzbuilder_analysis).


## Methodology
Early examination of the data suggested it was often the case that different arms in a single galaxy had dramatically different pitch angles.

![](./plots/example-spiral-angles.pdf)

However, most analysis of spiral pitch angle (and theories associated with spiral formation and evolution), do not make specific predictions for arm pitch angles, instead often quoting a single value for the "pitch angle of a galaxy". For this reason, a hierarchical view of galaxy pitch angle was adopted, where the pitch angle along an individual arm (phi_{arm}) was assumed to be constant (a logarithmic spiral), but arms in a galaxy were allowed to vary around some group mean (phi_{gal}), distributed Normally (and truncated to being between 0˚ and 90˚) with some spread common to all galaxies in the sample (sigma_{gal}). The motivation behind assuming a common sigma_{gal} was purely due to the limited sample size and low number of arms measured per galaxy.

For full details of the hierarchical model, please refer to the paper (or if you can read PyMC check out the `UniformBHSM` class in `hierarchial_model.py` - and please provide feedback to help make the code more readable!

## Results

Our model outputs predictions for all spiral parameters, including arm position, the spiral arm pitch angles, and associated galaxy pitch angles:

![](plots/example-spiral-fits.pdf)

We make use of these results to examine how spiral properties vary with galaxy morphology (determined using Galaxy Zoo 2 morphologies), and test a simple model of spiral winding.

To summarise: we do not find evidence that the presence or strength of a bar impacts spiral arm tightness, and do not find evidence to discount the model of spiral winding proposed by [Pringle & Dobbs (2019)](https://arxiv.org/pdf/1909.10291.pdf) (given limits of $15 < \phi < 50$). For further detail, please see the upcoming paper!

### Morphology

![](./plots/bulge_bar_phigal_distribution.jpg)

Testing both galaxy and individual arm pitch angle vs morphology, we cannot reject the null hypothesis at the 1% level for a significant proportion of our samples (1% or fewer):


![](./plots/bulge_bar_test_results.jpg)

### Spiral winding

We want to investigate whether spirals are static, rotating as rigid bodies,

[Video showing static spirals](./plots/qsdw_spiral.mp4)

whether their arms wind, dissapate and reform together,

[Video showing transient, linked spirals](./plots/linked_winding_spirals.mp4)

or whether each arm is formed, winds up and dissapates independently:


[Video showing independently winding spirals](./plots/reccurent_spiral.mp4)

We test the distribution of galaxy and arm pitch-angles against one uniform in cot between 15˚ and 50˚. Finding that we cannot reject the null hypothesis at the 1% level  for a significant proportion of our samples (2% or fewer):

![](./plots/combined_cot_uniform_marginalized_tests.jpg)

Notably, small shifts in the lower limit on $\phi$ will cause us to reject the hypothesis for most of the samples, suggesting further investigation into this model is required for it to be a well-understood test.


## Key Notebooks in this repository

In order to walk the reader through the analysis process, many Jupyter Notebooks have been authored to explain the stages involved. Below we detail the motivation and purpose of some key notebooks. Please submit an issue if any information is incorrect, if you have any queries or suggestions for improvements, or if you wish another notebook to be included in this list!


### A tutorial of the data structures used

`data_structure_tutorial.ipynb`

This notebook walks through the main data structures used in this analysis, from the spiral `Pipeline` and `Arm` objects output by the Galaxy Builder analysis code, through to the outputs of the `UniformBHSM` (Uniform Bayesian Hierarchical Spiral Model) class used to perform inference on the hierarchical model presented in the paper.

### Performing inference with the model

`performing_inference.ipynb`

In this notebook, we demonstrate the usage of the hierarchical model classes to fit a small number of galaxies from the dataset. It is a scaled-down version of `do_inference.py`.


### Plotting the spiral arm fits from the model

`plot_resulting_fits.ipynb`

This notebook recreates Figure 4 from the paper, showing the resulting logarithmic spiral fits overlaid on the source data points.


### Explaining the frequency of one-armed spirals in the Galaxy Builder dataset

`examining_one_armed_spirals.ipynb`

It was noted that a surprisingly large number of galaxies only contained one spiral arm. We demonstrate that this is primarily an artefact of the clustering used to group volunteer spiral arms, and is therefore a result of the trade-off between false positives and false negatives required when choosing spiral arm clustering hyperparameters.

### Investigating the link between morphology and pitch angle

`morphology_comparison.ipynb`

This notebook performs the analysis present in Section 3.2 of the paper, investigating whether we see evidence of a link between spiral pitch angle and the strength of a galaxy's bulge or bar. We find no statistically significant evidence for such a link.

### Investigating the model of spiral winding from Pringle & Dobbs (2019)

`testing_spiral_winding.ipynb`

This notebook performs the analysis present in Section 3.3 of the paper, investigating whether we see evidence against the model of spiral winding proposed by [Pringle and Dobbs (2019)](https://ui.adsabs.harvard.edu/abs/2019MNRAS.490.1470P/abstract). We find no evidence against their model for either galaxy pitch angle or arm pitch angle.
