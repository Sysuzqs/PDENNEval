import sys, os
import yaml
from functools import reduce

from timeit import default_timer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(config):
    with open(file=config, mode='r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
        print(cfg)
        if cfg['model_name'] == "PINN":
            from train import run_training as run_training_PINN
            print("PINN")
            run_training_PINN(
                scenario=cfg['scenario'],
                epochs=cfg['epochs'],
                learning_rate=cfg['learning_rate'],
                model_update=cfg['model_update'],
                flnm=cfg['filename'],
                seed=cfg['seed'],
                input_ch=cfg['input_ch'],
                output_ch=cfg['output_ch'],
                root_path=cfg['root_path'],
                val_num=cfg['val_num'],
                if_periodic_bc=cfg['if_periodic_bc'],
                aux_params=cfg['aux_params'],
                val_batch_idx = cfg['val_batch_idx']
            )


if __name__ == "__main__":
    config_path="./config/args/"
    config = sys.argv[1]
    main(config_path+config)
    print("Done.")
