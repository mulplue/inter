import os

def delete_mimic_checkpoints():
    checkpoint_dir = "checkpoints"

    deleted_count = 0
    for exp_name in os.listdir(checkpoint_dir):
        nn_dir = os.path.join(checkpoint_dir, exp_name, "nn")
        for pth_name in os.listdir(nn_dir):
            pth_path = os.path.join(nn_dir, pth_name)
            if not pth_name == "mimic.pth":
                os.remove(pth_path)
                print("deleted", pth_name)

if __name__ == "__main__":
    delete_mimic_checkpoints()
