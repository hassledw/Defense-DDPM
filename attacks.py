import torchattacks as ta
import timm
import torch
from birdclassifier import BirdDataset
from torch.utils.data import random_split, Dataset, DataLoader
from PIL import Image
import os

def load_model(model_path, num_classes):
    '''
    Returns the model given a saved path.
    '''
    model = timm.create_model("rexnet_150", pretrained=False, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def minmax_scaling_images(img_tensor):
    '''
    Normalizes a tensor to the range [0,1] using min-max scaling, used
    for tensor input into the attack framework.
    '''
    input_tensor = img_tensor
    min_value = 0.0
    max_value = 1.0

    # Normalize the tensor to the specified range
    normalized_tensor = (input_tensor - input_tensor.min()) / (input_tensor.max() - input_tensor.min())
    normalized_tensor = normalized_tensor * (max_value - min_value) + min_value
    return normalized_tensor

def tensor_to_image(tensor_imgs):
    '''
    A helper function that converts a transformed adversarial tensor 
    into an image that visualizable.

    tensor_imgs: tensors of images
    '''
    images = []
    for tensor in tensor_imgs:
        min_value = 0.0
        max_value = 1.0
        tensor = (tensor - min_value) / (max_value - min_value)
        tensor *= 255.0

        tensor = tensor.to(torch.uint8)
        numpy_image = tensor.cpu().numpy()

        pil_image = Image.fromarray(numpy_image.transpose(1, 2, 0))  # Channels last

        images.append(pil_image)

    return images

def attack_images(attack, test_ds, test_dl, save_adv_image_folder):
    '''
    Run a torch attack on the test set and saves the image data to a folder.

    attack: Any torch attack, i.e. FGSM, PGD, JSMA.
    EXAMPLE: attack = FGSM(model, eps=0.05)

    test_ds: the PyTorch dataset, this is used for retrieving the image paths.

    test_dl: the PyTorch dataloader, this is used to get the batches of input data.

    save_adv_image_folder: the folder in which to save the attacked image data.
    '''
    for idx, batch in enumerate(test_dl):
        images, labels = batch
        attacked_images_np = attack(minmax_scaling_images(images), labels)
        attacked_images = tensor_to_image(attacked_images_np)

        # since the dict is sorted, we can greedily grab all indexes by the batch_size.
        im_paths_idxes = [x for x in range(idx * test_dl.batch_size, test_dl.batch_size * (idx + 1))]

        # save each attacked image in its respective directory.
        for x, adv_image in enumerate(attacked_images):
            path_components = test_ds.im_paths[im_paths_idxes[x]].split("/")
            adv_img_path = f"{save_adv_image_folder}/{path_components[-2]}/{path_components[-1]}"

            # make the root folder "Attack" if it doesn't exist.
            if not os.path.isdir(save_adv_image_folder):
                os.mkdir(save_adv_image_folder)
            
            # make the label subfolder in "Attack" if it doesn't exist.
            if not os.path.isdir(f"{save_adv_image_folder}/{path_components[-2]}"):
                os.mkdir(f"{save_adv_image_folder}/{path_components[-2]}")

            adv_image.save(adv_img_path)

if __name__ == "__main__":
    '''
    Example usecase.
    '''
    test_ds = BirdDataset("./bird-data", "test")
    test_dl = DataLoader(test_ds, batch_size = 32, shuffle = False, num_workers = 4)
    model = load_model("./saved_models/birds_epoch7_val_loss0.092.pth", 525)

    attack_images(ta.FGSM(model, eps=0.05), test_ds, test_dl, "./bird-data/FGSM05-test")