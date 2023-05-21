# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 20:54:21 2022

@author: amart
"""

Wiloxin HOLM flow:
1. For each classifier, extract the counts of datasets
2. For all classifiers , calculate the friedman chi-aquare p value (must be < alpha)
3. For each classifier:
   a. Retrieve the accuracies of the classifiers over the test sets (perf1)
   b. For each of the other classifiers
	  i) Retrieve the accuracies of the other one perf2
	  ii) Compute wilkoxon p value
	  iii) store <cl1, cl2, p_value> tuple in p_values
 Sort list in ascending order of p_values
 Apply Holm's correction to the p_values
 
 <AGCTCN, AGCTCN_daily>  
    ... 
    ...
<AGCTCN, AGCTC_Satt>


 Extract accuracy array: [classifiers, num datasets]'
 Create dataset wise rank array for the classifiers
 Average the ranks
 
 Model1 -> Model2 if p value cannot be rejected
 Clique 
 