# [ICML 2026] ABS: Scalable Reinforcement Learning via Adaptive Batch Scaling

---

Our code products are based on [Clearn RL](https://github.com/vwxyzjn/cleanrl). Please, refer the code in "algos" folder.

## Requirements:
```bash
# core dependencies
pip install -r requirements/requirements.txt

# ALE (PQN)
pip install -r requirements/requirements-atari.txt
pip install -r requirements/requirements-envpool.txt

# Mujoco (PPO)
pip install -r requirements/requirements-mujoco.txt

```

## To run training scripts:
### Atari (PQN)
```python
# Baseline (vanilla PQN)
python cleanrl/pqn_atari_envpool.py --env-id Amidar-v5
# or
python cleanrl/pqn_atari_envpool_ours.py --env-id Amidar-v5
```
```python
# PQN + ABS
python cleanrl/pqn_atari_envpool_ours.py --env-id Amidar-v5 --adapt_rollout --num_steps_min 16 num_steps_max 64 
```
```python
# PQN-L (PQN with Multi-Skip Network Architecture)
python cleanrl/pqn_atari_envpool_ours.py --env-id Amidar-v5 --use_multiskip

# PQN-XL
python cleanrl/pqn_atari_envpool_ours.py --env-id Amidar-v5 --use_multiskip --mlp_num_layers 10
```
```python
# PQN-L + ABS
python cleanrl/pqn_atari_envpool_ours.py --env-id Amidar-v5 --use_multiskip --adapt_rollout --num_steps_min 16 num_steps_max 128

# PQN-XL + ABS
python cleanrl/pqn_atari_envpool_ours.py --env-id Amidar-v5 --use_multiskip --adapt_rollout --num_steps_min 16 num_steps_max 128 --mlp_num_layers 10
```
```python
# PQN + GNS
python cleanrl/pqn_atari_envpool_gns.py --env-id Amidar-v5 --use_gns --num_steps_min 16 num_steps_max 64
```


### Mujoco (PPO)
```python
# Baseline (vanilla PPO)
python cleanrl/ppo_continuous_action_ours.py --env-id HalfCheetah-v4
```
```
# PPO + ABS(KL)
python cleanrl/ppo_continuous_action_ours.py --env-id HalfCheetah-v4 --adapt_rollout --num_steps_min 1024 --num_steps_max 8192
```
