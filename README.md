# Anomaly_Detection_in_cMARL

# Installation

The experiments are conducted using the EPyMARL framework. Please follow the instructions in [SMAC](https://github.com/oxwhirl/smac) and [EPyMARL](https://github.com/uoe-agents/epymarl) to install StarCraft-II, SMAC, MPE, and LBF.

# Running Experiments

* The trained models are saved in “src/Trained Models” (the “predictors” in the paper are referred to as “trackers” in the code.)

* To run an experiment, for example, for DAA (Dynamic Adversary)  with 𝑤=10, 𝛽=−3, and for 100 episodes:

### In SMAC-MMM, run:
```
python main.py --config=qmix --env-config=sc2 with env_args.map_name=MMM test_nepisode=100 attack_type="DAA" thresholds=[-3] tracker_window=[-1,10] adv_load_adr=[Model Address]

```
### In SMAC-2s3z, run:
```
python main.py --config=qmix --env-config=sc2 with env_args.map_name=2s3z test_nepisode=100 attack_type="DAA" thresholds=[-3] tracker_window=[-1,10] adv_load_adr=[Model Address] hidden_dim=64 obs_last_action=True

```
### In MPE-Tag, run:
```
python main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleTag-v0" env_args.pretrained_wrapper="PretrainedTag" test_nepisode=100 attack_type="DAA" thresholds=[-3] tracker_window=[-1,10] adv_load_adr=[Model Address]

```
### In LBF, run:
```
python main.py --config=maa2c --env-config=gymma with env_args.time_limit=20 env_args.key="lbforaging:Foraging-3s-8x8-5p-4f-coop-v2" test_nepisode=100 attack_type="DAA" thresholds=[-3] tracker_window=[-1,10] adv_load_adr=[Model Address]

```

* For the “no window” case (window=∞), set `tracker_window=[-1]`
* To start the attack at a random time, set `attack_start_t=-1`
* [Model Address] is the address of the attack model in “src/Trained Models/DAA”

## Some Notes:
* The default config file is located in “src/config”. For the OBS attacks, set “attack_type” to “OA”, and set the DAA model to the corresponding DAA file with 𝝀=0 using `adv_load_adr`. (Note that “OA” runs slowly)
* To run an experiment without attack, set `attack_active=False`
* To train a new tracker model, set `attack_active=False` and `tracker_train=True`
* To train a new DAA (dynamic adversary) model for 𝝀=[𝑎,𝑏,𝑐,𝑑], set `adv_test_mode=False` and `lambda_DAA=[a,b,c,d]`

