# Mastering the Game of Sungka from Random Play
CS 295 Project for AY 2018-2019 by Darwin Bautista and Raimarc Dionido

## Project Structure
- `Documentation` - LaTeX source and its rendered output (paper.pdf)
- `pretrained` - pretrained model weights
- `environment.py` - Sungka environment implementation in OpenAI's Gym
- `model.py` - PyTorch model (and derived policy) and DQN-based trainer
- `policy.py` - handcrafted policies
- `train.py` - training loop. Running with default parameters would recreate the results shown in the paper.
- `test.py` - test code for evaluating or playing against (i.e. opponent == 'human') the trained DQN agent.
- `options.py` - common options and hyperparameters
- `benchmark.sh` - benchmark code for generating the data in the paper

## Training
To get the exact results shown in the paper, train the model using the default parameters:
```
$ python train.py --save_path results/
```
Every 100 training episodes, model weights will be saved in `results`. The final model weight would be in `results/p1-09999.pth`, and `results/p2-09999.pth` for the agent trained as player 2. The score and win rate plots would be saved as:
```
results/p1-test-rewards.png  (Figure 3 in the paper)
results/p1-train-rewards.png
results/p1-win-rates.png     (Figure 5 in the paper)
results/p2-test-rewards.png
results/p2-train-rewards.png
results/p2-win-rates.png
```

Training hyperparameters are as follows:
```
 --batch_size BATCH_SIZE
                        batch size; default=128
  --lr LR               learning rate; default=1e-5
  --gamma GAMMA         gamma/discount factor; default=0.9
  --mem_cap MEM_CAP     memory capacity; default=2000
  --num_episodes NUM_EPISODES
                        number of episodes; default=10000
  --num_test NUM_TEST   number of test episodes; default=100
  --opp_policy OPP_POLICY
                        opponent policy during training; default=random
  --q_net_iter Q_NET_ITER
                        number of iterations before updating target; default=100
```

## Testing
To play with the pretrained agent:
```
$ python test.py --load_path pretrained/p1-09999.pth --opp_policy human --render
```

Testing options are:
```
  --num_test NUM_TEST   number of test episodes; default=100
  --opp_policy OPP_POLICY
                        opponent policy during training; default=random
  --player PLAYER       player turn; default=1
  --render              render; default=False
  --pvp                 P1 weights vs P2 weights; default=False
```

## Benchmark
The complete benchmark code is provided for recreating the data in Tables I, II, and III:
```
$ ./benchmark.sh
```
