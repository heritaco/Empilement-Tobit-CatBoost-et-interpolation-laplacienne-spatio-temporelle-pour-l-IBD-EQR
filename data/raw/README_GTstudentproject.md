## **Diatom inventories and water quality**

### Context

Diatoms are unicellular microalgae belonging to the phylum *Bacillariophyta* found in aquatic ecosystems, either as planktonic organisms in the water column or as benthic organisms in biofilm structures. Benthic diatoms are ubiquitous in river systems, forming communities of varying richness. Benthic diatom taxa have specific sensibilities to environmental conditions (Soinen et al. 2004; Carayon et al. 2019), the composition of the diatom community of a given river thus reflects whether this site is affected by environmental pressures (Fig.1). Several diatom-based diagnostic tools of river\'s ecological quality have been proposed, like the IBD and the ODD. Those bioindicators make use of integrative community features such as taxonomy or trait-based diversity metrics (Larras et al. 2017).

As part of the Water Framework Directive (WFD) implementation, biotic and abiotic data are routinely collected to monitor the environmental quality of French river systems and are openly available. We have at our disposal data from close to 50,000 distinct diatom sampling operations (OPs) collected from 2007 to 2023 across more than 8000 sites (river reaches) in metropolitan France. These data are from *Naïades* (<https://naiades.eaufrance.fr/>). This \"biological\" dataset consists of taxon inventories with their associated relative abundances at each OP. In addition to diatom abundances, we also obtained measurements for various environmental parameters (pH, nutrient concentrations, pesticide concentrations, etc.) at each site.

Data are spread across several datasets, which are described hereafter.

### *01_DiatomInventories_GTstudentproject.csv*

This file contains the taxon inventories in a long format. These are relative abundances: to compute the Biological Diatom Index (IBD) of a given OP, operators usually identify up to 400 diatom valves. This file contains the following columns:

-   *TaxonName*: full name of diatom taxa identified in the inventories.

-   *TaxonCode*: seven character code correponding to the taxon name. See *07_TaxaCode_GTstudentproject.csv* for a table matching taxa names and codes.

-   *SamplingOperations_code*: code of the diatom sampling operation; it is a combination of the sampling site code and the sampling date.

-   *CodeSite_SamplingOperations*: code of the site (i.e., river reach) where the sampling operation took place.

-   *Date_SamplingOperation*: date of the sampling operation (as *Year-month-day*).

-   *Abundance_nbcell*: number of taxon cells counted for this OP.

-   *TotalAbundance_SamplingOperation*: total number of taxon cells counted for this OP. Because of the sampling protocol, this number should typically be close to 400.

-   *Abundance_pm*: permil taxa abundance (1000 × *Abundance_nbcell/TotalAbundance_SamplingOperation*).

### *02_InfoSites_GTstudentproject.csv*

This file gives information about sampling sites (i.e., the localization of OPs). It contains the following columns:

-   *CodeSite_SamplingOperations*: code of the site (i.e., river reach) where a sampling operation took place.

-   *Longitude_Lambert93*: longitudinal coordiante of the site (geodetic system: Lambert 93).

-   *Latitude_Lambert93*: latitudinal coordinate of the site (geodetic system: Lambert 93).

-   *Watershed*: name of the administrative watershed to which the site belongs.

-   *CodeDepartement*: code of the administrative department to which the site belongs.

-   *HERlvl1Code*: code of the HER-1 (hydroecoregion lvl1) to which the site belongs.

-   *HERlvl1Name*: nameof of the HER-1 (hydroecoregion lvl1) to which the site belongs.

-   *HERlvl2Code*: code of the HER-2 (hydroecoregion lvl1) to which the site belongs.

-   *HERlvl2Name*: nameof the HER-2 (hydroecoregion lvl1) to which the site belongs.

-   *Altitude*: Altitude (in m) of the sampling site.

-   *Streamsize*: the size of the river reach according to its Strahler rank. Possible values are Very Small (*TP*), Small (*P*), Intermediate (*M*), Large (*G*) and Very Large (*TG*). There are a few NA values for this variables.

HERs are spatially homogeneous entities for physical drivers (geology, climate, relief...) that control the functioning of aquatic ecosystems. They are a useful grouping factor for analyzing those data. Shapefiles for HERs can be downloaded here:

<https://www.sandre.eaufrance.fr/atlas/srv/api/records/24284d4e-37fa-47a2-85fe-850b37abe5a7> (HER lvl 1)

<https://www.sandre.eaufrance.fr/atlas/static/api/records/40b17d2a-5d4a-48ed-acdd-0728c080598c> (HER lvl2)

### *03_IBD_GTstudentproject.csv*

This dataset associate OPs to their Biological Diatom Index (IBD; Prygiel & Coste, 2000) value, a index of river ecological quality estimated from inventories composition.

The file contains the following columns:

-   *SamplingOperations_code*: code of the diatom sampling operation.

-   *IBD*: raw IBD value. This value ranges from 0 (lowest ecological quality) to 20 (maximum ecological quality).

-   *IBD_EQR*: *Ecological quality ratio* of the IBD, computed from the raw IBD value and from thresholds specific to the hydroecoregion (HER) to which the OP site belongs. Details in French about the computation of these values can be found here (section 1.1.2.1): <https://www.legifrance.gouv.fr/jorf/id/JORFARTI000037347765>

-   *IBD_EQR_Status*: ecological quality status (*Bad*, *Poor*, *Moderate*, *Good* or *High*) estimated from the value of *IBD_EQR*.

### *04_PressureStatus_GTstudentproject.csv*

*04_PressureStatus_GTstudentproject.csv* is the dataset of for various anthropic pressure categories. Measurements for a variety of environmental parameters (pH, nutrient concentrations, pesticide concentrations, etc.) were obtained at each site. We then computed their mean values over a specified period of time (one year, 180 days or 90 days) prior to each diatom OP. These integrated environmental means were used to characterize OPs for ten categories of anthropic pressures (\"Nitrogen compounds\", \"Microorganic pollutants\", etc.) following the SEQ-Eau guidelines. Each of those categories took into account multiple parameters with associated threshold values delimiting \"good\" and \"impaired\" pressure status (Mondy et al. 2012). The pressure level allocated to a given OP for a given pressure category was the worst pressure level allocated to this OP by individual parameters from this pressure category (\"one-out-all-out\" aggregation rule). In particular, we have classified each OP as being *Bad*, *Poor*, *Moderate*, *Good* or *High* for a given pressure category.

The file contains the following columns:

-   *SamplingOperations_code*: code of the diatom sampling operation.

-   *CodeSite_SamplingOperations*: code of the site of the diatom sampling operation

-   *Date_SamplingOperation*: date of the diatom sampling operation.

-   [X]*\_Status1Y*: quality status of OP for pressure category [X] using abiotic values averaged over one year prior to the OP.

-   [X]*\_Status180D*: quality status of OP for pressure category [X] using abiotic values averaged over 180 days prior to the OP.

-   [X]*\_Status90D*: quality status of OP for pressure category [X] using abiotic values averaged over 90 days prior to the OP.

Note: in some case, the quality status for some combination of OP and pressure category could not be assessed because of a lack of environmental data. The status is then reported as *Unassessed* in the dataset.

### *05_EnvParamMeans_GTstudentproject.csv* & *06_ListEnvParamPressure_GNNprojectGT.csv*

*05_EnvParamMeans_GNNprojectGT.csv* is the dataset of environmental parameter means (post harmonization process) used to estimate diatom sampling operations' pressure status.

This file contains the following columns:

-   *CodeSite_SamplingOperations*: code of the site of the diatom sampling operation (and thus of the abiotic parameter measure).

-   *SamplingOperations_code*: code of the diatom sampling operation that the parameter measure relates to.

-   *Pressure*: pressure category that the abiotic parameter belongs to.

-   *Parameter_code*: numerical SANDRE code of the abiotic parameter in Naiades.

-   *Parameter_label*: full name of the abiotic parameter.

-   *Mean1Y*: mean of the abiotic parameter over the year (365 days) prior to the diatom sampling operation.

-   *Mean180D*: mean of the abiotic parameter over the 180 days prior to the diatom sampling operation.

-   *Mean90D*: mean of the abiotic parameter over the 90 days prior to the diatom sampling operation.

Additional information about parameter units and pressure category threshold values can be found in the file *06_ListEnvParamPressure_GNNprojectGT.csv*.

### References

Carayon, D., J. Tison-Rosebery, and F. Delmas. 2019. Defining a new autoecological trait matrix for French stream benthic diatoms. *Ecological Indicators* 103:650--658.

Larras, F., R. Coulaud, E. Gautreau, E. Billoir, J. Rosebery, and P. Usseglio-Polatera. 2017. Assessing anthropogenic pressures on streams: A random forest approach based on benthic diatom communities. *Science of The Total Environment* 586:1101--1112.

Mondy, C. P., B. Villeneuve, V. Archaimbault, and P. Usseglio-Polatera. 2012. A new macroinvertebrate-based multimetric index (I2M2) to evaluate ecological quality of French wadeable streams fulfilling the WFD demands: A taxonomical and trait approach. *Ecological Indicators* 18:452--467.

Prygiel J., and Coste M. 2000. Guide méthodologique pour la mise en œuvre de l'Indice Biologique Diatomées NF T 90-354. Agences de l'Eau - Cemagref-Groupement de Bordeaux.

Soininen, J., A. Jamoneau, J. Rosebery, and S. I. Passy. 2016. Global patterns and drivers of species and trait composition in diatoms. *Global Ecology and Biogeography* 25:940--950.
