import json
import random
import hashlib
import itertools
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
configs = SCRIPT_DIR.parent / "configs"

available_gpus = []
load_all_to_gpu = True
num_canaries = 100
canary_type = "label_noise"

# Optimizer sweep parameters
parameter = [
    ### AdamW
    {"name": "AdamW", "betas": (0.9, 0.999), "eps": 1e-08}, # Default
    {"name": "AdamW", "betas": (0.0, 0.0), "eps": 1e-08},
    # Change beta1
    {"name": "AdamW", "betas": (0.999, 0.999), "eps": 1e-08},
    {"name": "AdamW", "betas": (0.99, 0.999), "eps": 1e-08},
    {"name": "AdamW", "betas": (0.8, 0.999), "eps": 1e-08},
    {"name": "AdamW", "betas": (0, 0.999), "eps": 1e-08},
    # Change beta2
    {"name": "AdamW", "betas": (0.9, 0.99), "eps": 1e-08},
    {"name": "AdamW", "betas": (0.9, 0.9), "eps": 1e-08},
    {"name": "AdamW", "betas": (0.9, 0.8), "eps": 1e-08},
    {"name": "AdamW", "betas": (0.9, 0), "eps": 1e-08},
    # Change eps
    {"name": "AdamW", "betas": (0.9, 0.999), "eps": 1e-07},
    {"name": "AdamW", "betas": (0.9, 0.999), "eps": 1e-06},
    {"name": "AdamW", "betas": (0.9, 0.999), "eps": 1e-05},
    ### RMSProp
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0, "eps": 1e-08}, # Default
    # Change eps
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0, "eps": 1e-07},
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0, "eps": 1e-06},
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0, "eps": 1e-05},
    # Change alpha (=beta2 of AdamW)
    {"name": "RMSprop", "alpha": 0.999, "momentum": 0, "eps": 1e-08},
    {"name": "RMSprop", "alpha": 0.9, "momentum": 0, "eps": 1e-08},
    {"name": "RMSprop", "alpha": 0.8, "momentum": 0, "eps": 1e-08},
    # Change momentum
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0.999, "eps": 1e-08},
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0.99, "eps": 1e-08},
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0.9, "eps": 1e-08},
    {"name": "RMSprop", "alpha": 0.99, "momentum": 0.8, "eps": 1e-08},
    ### SGD
    {"name": "SGD", "momentum": 0}, # Default
    # Change momentum
    {"name": "SGD", "momentum": 0.999},
    {"name": "SGD", "momentum": 0.99},
    {"name": "SGD", "momentum": 0.9},
    {"name": "SGD", "momentum": 0.8},
]
# TODO: lr and weight_decay should be taken from the base config
optimizers = ["AdamW", "SGD", "RMSprop"]
learning_rates = [1e-4, 1e-3, 1e-2]
weight_decays = [0.0, 1e-4, 1e-2]

num_repetitions = 3
num_files = 1
seed = 4242
shuffle = False

def get_deterministic_seed(*args, salt=seed):
    s = str(salt) + "_" + "_".join(str(a) for a in args)
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 1000000

random.seed(seed)

all_commands = []

def config_priority(config_path):
    name = config_path.name.lstrip("+_")
    if "MNIST" in name and "MSE" in name and "MLP" in name:
        p = 0
    elif "MNIST" in name and "CE" in name and "MLP" in name:
        p = 1
    elif "VIT" in name:
        p = 2
    elif "MADD" in name and "CE" in name:
        p = 3
    elif "MADD" in name and "MSE" in name:
        p = 4
    else:
        p = 99
    sub = 0 if "NO_GROK" in name else 1
    return (p, sub)

def get_steps(config_path):
    name = config_path.name
    if "MADD" in name:
        return 150_000
    if "MNIST" in name:
        return 150_000
    return 150_000

configs_list = list(configs.glob("*.json"))
configs_list = [c for c in configs_list]
configs_list.sort(key=config_priority)

def cmd(
    config,
    seed,
    data_seed,
    model_index,
    canary_json,
    opt_name,
    lr,
    wd,
    steps=None,
    name_prefix="",
    postfix=None,
):
    if steps is None:
        steps = get_steps(config)
    
    # We construct the JSON for the optimizer to override completely
    if opt_name == "SGD":
        opt_dict = {"name": "SGD", "lr": lr, "weight_decay": wd, "momentum": 0.9}
    elif opt_name == "AdamW":
        opt_dict = {"name": "AdamW", "lr": lr, "weight_decay": wd}
    elif opt_name == "RMSprop":
        opt_dict = {"name": "RMSprop", "lr": lr, "weight_decay": wd}
    else:
        opt_dict = {"name": opt_name, "lr": lr, "weight_decay": wd}
        
    opt_json = json.dumps(opt_dict)
    
    cmd_str = (
        f"pag train optimizer-sweep-v1 {config.name} {steps} "
        f"--run-name {name_prefix}{config.stem} "
        f"-o seed={seed} -o data.seed={data_seed} -o data.mask.seed={data_seed} "
        f"-o data.mask.model_index={model_index} -o data.canary='{canary_json}' "
        f"-o optimizer='{opt_json}'"
    )
    if load_all_to_gpu:
        cmd_str += " --load-all-to-gpu"
    if postfix is not None:
        cmd_str += postfix
    return cmd_str

file_lines = {f: [] for f in range(num_files)}
start_file = 0

for i in range(num_repetitions):
    rep_commands = []
    for config in configs_list:
        steps = get_steps(config)
        for opt, lr, wd in itertools.product(optimizers, learning_rates, weight_decays):
            data_seed = get_deterministic_seed(config.name, opt, lr, wd, "data_seed")
            c_num = 226 if "MADD" in config.name else num_canaries
            canary_dict = {"name": f"{canary_type}", "num": c_num}
            canary_json = json.dumps(canary_dict)
            run_seed = get_deterministic_seed(config.name, opt, lr, wd, i, "run_seed")
            
            name_prefix = f"{opt}_lr{lr}_wd{wd}_"
            rep_commands.append(
                cmd(
                    config,
                    run_seed,
                    data_seed,
                    i,
                    canary_json,
                    opt,
                    lr,
                    wd,
                    steps=steps,
                    name_prefix=name_prefix,
                )
            )

    if shuffle:
        random.shuffle(rep_commands)

    all_commands.extend(rep_commands)
    
    for idx, command in enumerate(rep_commands):
        file_lines[(start_file + idx) % num_files].append(command)
    start_file = (start_file + len(rep_commands)) % num_files

print(f"Total commands generated: {len(all_commands)}")

# Clean up any existing optimizer_sweep_*.txt files
for old_file in SCRIPT_DIR.glob("optimizer_sweep_*.txt"):
    old_file.unlink()

N_gpus = len(available_gpus)
for f_idx in range(num_files):
    rep_lines = file_lines[f_idx]
    if N_gpus > 0:
        for idx, line in enumerate(rep_lines):
            rep_lines[idx] = f"CUDA_VISIBLE_DEVICES={available_gpus[idx % N_gpus]} " + line

    cmd_file = SCRIPT_DIR / f"optimizer_sweep_{f_idx}.txt"
    cmd_file.write_text("\n".join(rep_lines), encoding="utf-8")
    print(f"Wrote {len(rep_lines)} commands to {cmd_file.name}")
