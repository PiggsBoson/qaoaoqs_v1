# Error Mitigation with Bipartite Hamiltonian Switching Control
This package is for simulating the control of qubits coupled to coherent and dissapative noise using a Hamiltonian switching ansatz inspired by the quantum approximate optimization algorithm (QAOA).

To setup, 
```
git clone https://github.com/PiggsBoson/qaoaoqs_v1.git
cd qaoaoqs_v1/Environments
conda env create -f environment.yml
conda activate hsctrl
```
An example for running simulations,  

```
python run/train.py --exp_name test --path results/path --p 20 --num_iters 2000 -lr 1e-2 --testcase XmonTLS --env_dim 2 --lr_decay -b 2048 -e 5 --au_uni Had --cs_coup eq --distribution logit-normal --protocol_renormal True --impl quspin --T_tot 10 --scale 1.0
```
Please see run/train.py for details of the arguments.

To plot results, 
```
python result_analysis/analysis.py [arguments]
```
Please see esult_analysis/analysis.py for details of the arguments.

The manuscript of this work is in preparation.
