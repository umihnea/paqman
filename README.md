# paqman
Learning to play Pacman (and other Atari games) using DQN methods (vanilla, double DQN and prioritized experience replay).

### Training
Training can be initialized locally or remotely on a cloud environment.

If you are completely in control of the machine the training is running on, you can run:
```
python train.py -c [config] [options]
``` 

Various configuration files are stored in `config/`.

#### Options
You can specify by a flag which **algorithm** the agent will use to the training script.
```
no flag: train using vanilla DQN
--per: train using Prioritized Experience Replay
--doubledqn: train using Double DQN
```

#### Logging
Training runs will log to `data/logs/training.log` by default.

#### Remote training
I use `nohup` in order to prevent termination of a SSH session.
Output will be redirected to a file `nohup.out` by default.
The `&` will start the process in the background.
```
nohup python train.py -c [config] [options] &
```

You can track the training progress using:
```
tail -f data/logs/training.log
```

The training process knows to **shutdown softly**, while saving all progress it has made so far and archiving it to a file named `results.tar.gz`.

In order to achieve soft shutdown, kill the Python process with `SIGKILL` (signal 2):
```
pkill -2 [pid of nohup'd Python process]
```

### Utility
To clean up the `data` directory, run `cleanup.sh`. Note that this script deletes everything in `data/` and recreates the directory entirely, along with some necessary files.
```
./cleanup.sh
```

To format code and documentation, run:
```
./format.py
```