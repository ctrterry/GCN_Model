# GCN_Model
GCN Node Classification Ablation Studies

### Step 1, need install my necessary package
- pip install -r requirements.txt
### step2, Run my main code. 
- cd stage 5
- python script/stage_5_script/main.py
- This will runing all my dataset. Like 'cora', 'pubmed', 'citeseer'.
### step3, run my ablation study full version.
- cd stage 5
- python local_code/stage_5_code/ablation_full_hyperparams.py
- This will runing all of my
- 
### Data Output Result:
- the plot results will save into the result/stage_5_result

### The result explain:
- Based on the GCN research paper show that the best model accuracy can be reach to: 
- Method             Citeseer    Cora     Pubmed 
-GCN (this paper)    70.3 (7s)  81.5 (4s) 79.0 (38s)
- GCN (rand. splits) 67.9 ±0.5  80.1 ±0.5 78.9 ±0.7

- However, Our Model Accuracy can reach to the GCN Paper's Accuracy. Hence, which indicate our model can be detective the GCN very well.
- Such as, 
- Method          Citeseer    Cora     Pubmed
- GCN              0.7020     0.7910   0.7720
