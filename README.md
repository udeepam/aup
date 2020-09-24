# Auxiliary Utility Preservation
This repository contains PyTorch code for applying the theory from the [Avoiding Side Effects in Complex Environments](https://arxiv.org/abs/2006.06547) paper by Turner et al. to the [OpenAI Procgen Benchmark](https://openai.com/blog/procgen-benchmark/).

## Example usage
To obtain a learned auxiliary Q-function run
```bash
python pretrain.py --model ppo --env_name coinrun --q_aux_path q_aux_dir/coinrun/0.pt
```

To train PPO agent run
```bash
python main.py --model ppo --env_name coinrun --test True
```

To train PPO agent with AUP run
```bash
python main.py --model ppo_aup --env_name coinrun --q_aux_dir q_aux_dir/coinrun/ --test True
```

## Requirements

* Python 3
* [PyTorch](http://pytorch.org/)
* [OpenAI baselines](https://github.com/openai/baselines)

In order to install requirements, follow:

```bash

# Baselines
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .

# Other requirements
pip install -r requirements.txt
```

