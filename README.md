# Code to fit theoretical models to ROC data

The main code can be found in `roc_models.py`. See `testing_roc_models.ipynb` to see an example run with some outputs.

Steps for using this:
1. Specify the input data which should be the counts of responses across different levels of a "decision" scale. 
2. Instantiate a particular model object or objects like `DualProcess()` or `HighThreshold()`. 
3. Add the data using the `.add_data()` method to add the signal and noise data.
4. Call the `.optifit()` method: this uses `Scipy.optimize.minimize()` on the given model's parameters (e.g. 'd', 'R') to find their best fitting values (according to the sum of G<sup>2</sup>).
5. Call `.plot()` to see the results.

By running multiple models you cacn compare their individual fits to see which model best describes the data. 


#### Note
My supervisor Bertram Opitz first wrote these models in Excel, based on the recognition memory literature (e.g. Yonelinas et al., 1996). I decided to convert this into a Python implementation that can potentially be built upon. 

### References
Byrne, J. H. (2008). Learning and memory: A comprehensive reference. Amsterdam: Elsevier.

Yonelinas AP, Dobbins I, Szymanski MD, Dhaliwal HS, King L. (1996). Signal-detection, threshold, and dual-process models of recognition memory: ROCs and conscious recollection. Conscious Cogn. 5:418–78
