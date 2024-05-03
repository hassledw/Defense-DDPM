import pandas as pd
import numpy as np
from birdclassifier import BirdDataset
from torch.utils.data import DataLoader
import torch
import timm, torchmetrics
from tqdm import tqdm
import re

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, num_classes):
    '''
    Returns the model given a saved path.
    '''
    print("Loading Rexnet_150 Classification Model...")
    model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    return model

def generate_csv(root, attack_folder, defense_folder, true_folder="test", model_path="./saved_models/birds_epoch7_val_loss0.092.pth", batch_size=15):
    '''
    Given an attack, creates a CSV file with columns: model_test_label, attack_label, defense_label_t=?

    This will output a CSV file named ATTACK_FOLDER-defense-eval.csv.

    Used for evaluation.
    '''
    assert batch_size % 5 == 0
    model = load_model(model_path, 525)
    df = pd.DataFrame()
    imagenames = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"] * (batch_size // 5)

    folders = [true_folder, attack_folder, defense_folder]
    column_names = ["model_test_label", "attack_label", f"defense_label_{defense_folder.split('-')[-1]}"]

    for fi, (folder, column_name) in enumerate(zip(folders, column_names)):
        ds = None
        if "wavelet" in root or "nl-means" in root:
            if (folder == true_folder) or (folder == attack_folder):
                ds = BirdDataset("./bird-data", folder)
            else:
                ds = BirdDataset(root, folder)
        else:
            ds = BirdDataset(root, folder)

        bird_names_dict = {value: key for key, value in ds.cls_names.items()}
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
        dl_classifications = []
        imagepaths = []
        
        for i, batch in enumerate(dl):
            images, label_ids = batch
            preds = model(images.to(device))
            pred_label_ids = preds.argmax(axis=1)

            bird_names_batch = []
            for pred_label_id in pred_label_ids:
                bird_names_batch.append(bird_names_dict[pred_label_id.item()])
                dl_classifications.append(bird_names_dict[pred_label_id.item()])
            
            imagepaths.append([f"{bird_name}/{imagename}" for bird_name, imagename in zip(bird_names_batch, imagenames)])

            print(f"{folder}: Batch {i + 1}/{len(dl)} complete!")
        
        if fi == 0:
            print(np.array(imagepaths).flatten())
            df["image"] = np.array(imagepaths).flatten()

        df[column_name] = dl_classifications
        
        print(df)

    if "wavelet" in root:
        df.to_csv(f"./evaluation_csvs/{attack_folder}-wavelet-defense-eval.csv", index=False)
    elif "nl-means" in root:
        df.to_csv(f"./evaluation_csvs/{attack_folder}-nlmeans-defense-eval.csv", index=False)    
    else:
        df.to_csv(f"./evaluation_csvs/{attack_folder}-defense-eval.csv", index=False)

def append_defense_column(root, folder, attack_folder, df, model_path="./saved_models/birds_epoch7_val_loss0.092.pth", batch_size=15):
    '''
    Append a new defense column into the original df.
    '''
    model = load_model(model_path, 525)
    ds = BirdDataset(root, folder)
    bird_names_dict = {value: key for key, value in ds.cls_names.items()}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    dl_classifications = []

    for i, batch in enumerate(dl):
        images, label_ids = batch
        preds = model(images.to(device))
        pred_label_ids = preds.argmax(axis=1)

        for pred_label_id in pred_label_ids:
            dl_classifications.append(bird_names_dict[pred_label_id.item()])

        print(f"{folder}: Batch {i + 1}/{len(dl)} complete!")

    df[f"defense_label_{folder.split('-')[-1]}"] = dl_classifications

    print(df)

    if "wavelet" in root:
        df.to_csv(f"./evaluation_csvs/{attack_folder}-wavelet-defense-eval.csv", index=False)
    elif "nl-means" in root:
        df.to_csv(f"./evaluation_csvs/{attack_folder}-nlmeans-defense-eval.csv", index=False) 
    else:
        df.to_csv(f"./evaluation_csvs/{attack_folder}-defense-eval.csv", index=False)

def remove_column(df, column, csv_title):
    '''
    Wrapper code to remove a column, save to csv.
    '''
    df = df.drop(column, axis=1)
    df.to_csv(csv_title)

def order_csv(df, csv_file):
    '''
    Reorders the defense csv to be numerically ordered.
    '''
    columns = [column for column in df.columns if "defense" in column]
    result_columns = [column for column in df.columns if "defense" not in column]

    def sort_condition(s):
        match = re.search(r'\d+', s)
        return int(match.group()) if match else None
    
    sorted_order = sorted(columns, key=sort_condition)
    result_columns.extend(sorted_order)
    df = df[result_columns]
    df.to_csv(csv_file, index=False)
    print(df)


def generate_orig_defense_csv(attack_folder="FGSM05-test", csv_path="./evaluation_csvs/FGSM05-test-defense-eval.csv"):
    '''
    Creates the CSV for DDPM defense without any input transformations.
    '''
    root = "./bird-data"
    attack_name = attack_folder.split("-")[0]
    generate_csv(root=root, attack_folder=attack_folder, defense_folder=f"{attack_name}-test-def-t=1", true_folder="test")
    append_defense_column(root, f"{attack_name}-test-def-t=2", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-test-def-t=5", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-test-def-t=10", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-test-def-t=20", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-test-def-t=30", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-test-def-t=40", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-test-def-t=100", attack_folder, pd.read_csv(csv_path))

def generate_wavelet_defense_csv(attack_folder="FGSM05-test", csv_path="./evaluation_csvs/FGSM05-test-wavelet-defense-eval.csv"):
    '''
    Creates the CSV for DDPM defense with wavelet input transformations.
    '''
    root = "./bird-data/wavelet"
    attack_name = attack_folder.split("-")[0]
    generate_csv(root=root, attack_folder=attack_folder, defense_folder=f"{attack_name}-wavelet-test-def-t=1", true_folder="test")
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=2", attack_folder, 
                         pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=5", attack_folder, 
                         pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=10", attack_folder, 
                        pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=20", attack_folder, 
                        pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=30", attack_folder, 
                        pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=40", attack_folder, 
                        pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-wavelet-test-def-t=100", attack_folder, 
                        pd.read_csv(csv_path))

def generate_nlmeans_defense_csv(attack_folder="FGSM05-test", csv_path="./evaluation_csvs/FGSM05-test-nlmeans-defense-eval.csv"):
    root = "./bird-data/nl-means"
    attack_name = attack_folder.split("-")[0]
    generate_csv(root=root, attack_folder=attack_folder, defense_folder=f"{attack_name}-nlmeans-test-def-t=1", true_folder="test")
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=2", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=5", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=10", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=20", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=30", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=40", attack_folder, pd.read_csv(csv_path))
    append_defense_column(root, f"{attack_name}-nlmeans-test-def-t=100", attack_folder, pd.read_csv(csv_path))

def generate_clean_from_adv_csv(attack_folder="FGSM25-test", csv_path="./evaluation_csvs/FGSM25-test-clean-from-adv-eval.csv"):
    root = "./bird-data"
    attack_name = attack_folder.split("-")[0]
    generate_csv(root=root, attack_folder=attack_folder, defense_folder=f"{attack_name}-test-def-t=1000", true_folder="test")


def main():
    pass
    # generate_orig_defense_csv("FGSM25-test", "./evaluation_csvs/FGSM25-test-defense-eval.csv")
    # generate_nlmeans_defense_csv(attack_folder="FGSM25-test", csv_path="./evaluation_csvs/FGSM25-test-nlmeans-defense-eval.csv")
    # generate_wavelet_defense_csv(attack_folder="FGSM25-test", csv_path="./evaluation_csvs/FGSM25-test-wavelet-defense-eval.csv")
    # generate_orig_defense_csv()
    #remove_column(pd.read_csv("./evaluation_csvs/FGSM05-test-defense-eval.csv"), "defense_label_t=25", "./evaluation_csvs/FGSM05-test-defense-eval.csv")
    #append_defense_column("./bird-data", "FGSM05-test-def-t=30", "FGSM05-test", pd.read_csv("./evaluation_csvs/FGSM05-test-defense-eval.csv"))

    # order_csv(pd.read_csv("./evaluation_csvs/FGSM05-test-defense-eval.csv"), "./evaluation_csvs/FGSM05-test-defense-eval.csv")
if __name__ == "__main__":
    main()