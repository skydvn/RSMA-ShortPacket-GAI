from agents.base_agent import *
from agents.GAI.VAE import *
from agents.GAI.LICE import *
from agents.GAI.AE import *
from agents.GAI.GAN import *

from utils.setting_setup import *
from envs.environment import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    args = get_arguments()

    alg = VAE()
    env = base_env(args)
    agent = base_agent(args,
                       env,
                       alg)

    if args.flag_d_collect == True:
        agent.data_collect()
    if args.flag_train == True:
        agent.train()
    if args.flag_eval == True:
        agent.evaluate()



