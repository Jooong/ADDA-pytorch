import argparse

arg_list =[]
parser = argparse.ArgumentParser()

def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_list.append(arg)
    return arg

# Data
data_arg = add_argument_group("Data")
data_arg.add_argument("--batch_size", type=int, default=128)

# Training
train_arg = add_argument_group("Training")
train_arg.add_argument("--is_train_source", type=str2bool, default=True)
train_arg.add_argument("--is_finetune", type=str2bool, default=False)
train_arg.add_argument("--model_dir", type=str, default="./pretrained/lenet-source.pth")
train_arg.add_argument("--optimizer", type=str, default="adam")
train_arg.add_argument("--max_epoch", type=int, default=10)
train_arg.add_argument("--lr",type=float, default=0.0002)
train_arg.add_argument("--beta1",type=float, default=0.5)
train_arg.add_argument("--beta2",type=float, default=0.999)
train_arg.add_argument("--weight_decay",type=float, default=0.00002)

# Misc
misc_arg = add_argument_group("Misc")
misc_arg.add_argument("--log_step", type=int, default=10)
misc_arg.add_argument("--save_step",type=int, default=10)
misc_arg.add_argument("--num_gpus",type=int, default=0)
misc_arg.add_argument("--log_dir",type=str, default="logs")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
