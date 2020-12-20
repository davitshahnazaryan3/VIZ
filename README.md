# Visualization AID

Aid to postprocess analysis results, visualize and export them
The module acts as an aid for plotting and generation of vectorized or otherwise figures. Those are largely preliminary graphs, which might require some hand fixing to meet the user's needs. 

**Required packages**

numpy, pandas, matplotlib, xlrd, re

Requires inkscape: for vectorized high quality image exports

### Modules

1. EqEngineering - seismic engineering related postprocessing

    1. SPO - visualization for SPO analysis
    
    2. IDA - visualization for IDA
    
    3. Loss - visualization for los assessment

    4. SLF - visualization for storey loss functions

    5. Hazard - visualization for PSHA / Hazard

### Postprocessor

1. [x] IDA results

### Figures

1. [x] Hazard
    1. [x] Hazard
    2. [x] Hazard with 2nd-order fitting
2. [x] SLFs
    1. [x] SLFs on EDP group basis lumped at each storey
    2. [x] Individual SLFs along with fitting parameters, functions and accuracies and scatter of true data
    3. [ ] Fragility functions
    4. [ ] Consequence functions
    5. [ ] Individual SLFs, input data to be matched with SLFGenerator
3. [ ] IPBSD
    1. [x] Loss Curve
    2. [x] Design spectrum at SLS
    3. [x] Design solution space and a backbone curve of the design solution
    4. [ ] Moment-curvature relationships
    5. [ ] SLFs and design EDP estimations
    6. [x] SPO2IDA
    7. [x] SPO + fitting
4. [ ] Ground motion selection
5. [x] RCMRF
    1. [x] SPO
    2. [x] IDA results
6. [ ] Loss
    1. [ ] Vulnerability curves
    2. [ ] EAL breakdown
    3. [ ] Relative contributions to expected loss via Area plots
7. [ ] Assumption verifications
    1. [x] SPO2IDA and Model IDA comparisons (SPO, IDA, overstrength, ductilities)
    2. [ ] Loss curve comparisons
    3. [x] MAFC validation
    4. [ ] EAL validation
    5. [ ] 

#### Things to recheck, reevalute

* Check that SDOF - MDOF transformations are correct
* Check that IDA fitting is good enough
* Check input for SLF plotter to match with SLFGenerator output









