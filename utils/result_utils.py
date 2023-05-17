import os
import h5py
import torch
from typing import List


def save_results(
                 g_losses: List[float],
                 actor_losses: List[float],
                 critic_losses: List[float],
                 reward_list: List[float],
                 algo
                 ):
    result_path = "./results/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if len(g_losses):
        file_path = result_path + "{}.h5".format(algo)
        print("File path: " + file_path)

        with h5py.File(file_path, 'w') as hf:
            hf.create_dataset('reward', data=reward_list)
            hf.create_dataset('g_losses', data=g_losses)
            hf.create_dataset('actor_losses', data=actor_losses)
            hf.create_dataset('critic_losses', data=critic_losses)


def save_item(save_folder_name, item_actor, item_critic, item_name):
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    torch.save(item_actor, os.path.join(save_folder_name, "actor-" + item_name + ".pt"))
    torch.save(item_critic, os.path.join(save_folder_name, "critic-" + item_name + ".pt"))


def load_item(save_folder_name, item_name):
    return torch.load(os.path.join(save_folder_name, "actor-" + item_name + ".pt")), \
           torch.load(os.path.join(save_folder_name, "critic-" + item_name + ".pt"))

def save_data(save_folder_name, data):
    # Create an HDF5 file
    save_name = save_folder_name + f"/data.h5"
    file = h5py.File(save_name, "w")

    # Create a dataset within the file and write the data
    with h5py.File(file, 'w') as hf:
        hf.create_dataset('train', data=data)
        hf.create_dataset('test', data=data)

    # Close the file
    file.close()

