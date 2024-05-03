from dataclasses import dataclass
from torch.utils.data import random_split, Dataset, DataLoader
from datasets import load_dataset
from birdclassifier import BirdDataset
import matplotlib.pyplot as plt
from torchvision import transforms
from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import safetensors.torch

import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import DDPMScheduler, DDPMPipeline
import math

from accelerate import Accelerator, notebook_launcher
from huggingface_hub import HfFolder, Repository, whoami
from tqdm.auto import tqdm
from pathlib import Path
import os
import glob
device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class TrainingConfig:
    image_size = 224  # the generated image resolution
    train_batch_size = 8
    eval_batch_size = 8  # how many images to sample during evaluation
    num_epochs = 200
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 30
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "bird-data-defense-ddpm"  # the model name locally and on the HF Hub

    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0


class DDPMBirds:
    def __init__(self):
        self.config = TrainingConfig()
        self.dataset = BirdDataset("./bird-data", "test")
        self.dl = DataLoader(self.dataset, batch_size = self.config.train_batch_size, shuffle = True, num_workers = 4)
        self.model = UNet2DModel(
            sample_size=self.config.image_size,  # the target image resolution
            in_channels=3,  # the number of input channels, 3 for RGB images
            out_channels=3,  # the number of output channels
            layers_per_block=2,  # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
            down_block_types=(
                "DownBlock2D",  # a regular ResNet downsampling block
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",  # a regular ResNet upsampling block
                "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )

    def plot_bird_data(self, n):
        '''
        Plots some birds from the training set.
        '''
        fig, axs = plt.subplots(1, n, figsize=(16, 4))
        images = []
        for batch in self.dl:
            ims, labels = batch[0], batch[1]
            for i, im in enumerate(ims):
                im = im - im.min()  # Translate pixel values to start from 0
                im = im / im.max()  # Scale pixel values to [0, 1]
                im = im.permute(1, 2, 0)

                axs[i].imshow(im)
                axs[i].set_axis_off()
                
                images.append(im)
                if i > n - 2:
                    break
            break

        return images

    def make_grid(self, images, rows, cols):
        '''
        Helper function to make the grid for the image output.
        '''
        w, h = images[0].size
        grid = Image.new("RGB", size=(cols * w, rows * h))
        for i, image in enumerate(images):
            grid.paste(image, box=(i % cols * w, i // cols * h))
        return grid


    def evaluate(self, epoch, pipeline):
        '''
        Sample some images from random noise (this is the backward diffusion process).
        The default pipeline output type is `List[PIL.Image]`
        '''
        images = pipeline(
            batch_size=self.config.eval_batch_size,
            generator=torch.manual_seed(self.config.seed),
        ).images

        # Make a grid out of the images
        image_grid = self.make_grid(images, rows=4, cols=4)

        # Save the images
        test_dir = os.path.join(self.config.output_dir, "samples")
        os.makedirs(test_dir, exist_ok=True)
        image_grid.save(f"{test_dir}/{epoch:04d}.png")

        basic_images = self.plot_bird_data(4)
        return basic_images
    
    def verify_data_shape(self):
        '''
        Verifies the shape of the data, returns an image.
        '''
        ### Verify the shape of the data
        for batch in self.dl:
            images, labels = batch[0], batch[1]
            sample_image = images[0].unsqueeze(0)
            print("Input shape:", sample_image.shape)

            print("Output shape:", self.model(sample_image, timestep=0).sample.shape)
            break

        return sample_image
    
    def ddpm_flow_single_example(self):
        '''
        Runs the forward and backward process on a single example.
        '''
        sample_image = self.verify_data_shape()
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        noise = torch.randn(sample_image.shape)
        timesteps = torch.LongTensor([50])
        noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)
        noise_pred = self.model(noisy_image, timesteps).sample
        loss = F.mse_loss(noise_pred, noise)
        print(loss)

        return noise_pred

    def train_ddpm(self):
        '''
        Trains the entire DDPM using UNet2D model, noise_scheduler, optimizer, lr_scheduler, etc.
        '''
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.config.lr_warmup_steps,
            num_training_steps=(len(self.dl) * self.config.num_epochs),
        )

        noise_pred = self.ddpm_flow_single_example()
        Image.fromarray(((noise_pred.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

        # Initialize accelerator and tensorboard logging
        accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=os.path.join(self.config.output_dir, "logs"),
        )
        if accelerator.is_main_process:
            if self.config.output_dir is not None:
                os.makedirs(self.config.output_dir, exist_ok=True)
            accelerator.init_trackers("train_example")

        # Prepare everything
        # There is no specific order to remember, you just need to unpack the
        # objects in the same order you gave them to the prepare method.
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )

        global_step = 0

        # Now you train the model
        for epoch in range(self.config.num_epochs):
            progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
            progress_bar.set_description(f"Epoch {epoch}")

            for step, batch in enumerate(train_dataloader):
                clean_images = batch[0]
                # Sample noise to add to the images
                noise = torch.randn(clean_images.shape).to(clean_images.device)
                bs = clean_images.shape[0]

                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
                ).long()

                # Add noise to the clean images according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

                with accelerator.accumulate(model):
                    # Predict the noise residual
                    noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                    loss = F.mse_loss(noise_pred, noise)
                    accelerator.backward(loss)

                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                progress_bar.update(1)
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)
                global_step += 1

            # After each epoch you optionally sample some demo images with evaluate() and save the model
            if accelerator.is_main_process:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

                if (epoch + 1) % self.config.save_image_epochs == 0 or epoch == self.config.num_epochs - 1:
                    self.evaluate(epoch, pipeline)

                if (epoch + 1) % self.config.save_model_epochs == 0 or epoch == self.config.num_epochs - 1:
                    pipeline.save_pretrained(self.config.output_dir)

        notebook_launcher(self.train_ddpm, num_processes=1)


class DDPMDefense:
    def __init__(self, attack_ds, attack_dl, backwards_steps, save_def_image_folder, forwards_steps=0):
        self.attack_ds = attack_ds
        self.attack_dl = attack_dl
        self.save_def_image_folder = save_def_image_folder
        self.backwards_steps = backwards_steps
        self.forwards_steps = forwards_steps

    def visualizable_image(self, reverse_input):
        '''
        Reshapes the reverse_input tensor to be visualizable.
        '''
        input_visualizable = reverse_input.permute(1, 2, 0)

        # # Step 2: Remove the batch dimension
        # input_visualizable = input_visualizable.squeeze(0)

        # If `input_visualizable` tensor is in a float format with values in [0, 1] (common for models),
        # you might need to scale it to [0, 255] and convert to an integer format for proper visualization.
        if input_visualizable.is_floating_point():
            input_visualizable = (input_visualizable * 255).byte()

        # Convert to NumPy array for visualization if necessary
        input_visualizable = input_visualizable.cpu().numpy()
        print(input_visualizable.shape)
        return input_visualizable
    
    def visualizeable_image_2(self, reverse_input):
        '''
        Better code for visualizing the images.
        '''
        im = reverse_input
        im = im - im.min()  # Translate pixel values to start from 0
        im = im / im.max()  # Scale pixel values to [0, 1]
        im = im.permute(1, 2, 0)

        return im.cpu().numpy()
    
    def save_to_folder(self, idx, reversed_inputs):
        '''
        Saves the defended images to the specified folder.
        '''
        # since the dict is sorted, we can greedily grab all indexes by the batch_size.
        im_paths_idxes = [x for x in range(idx * self.attack_dl.batch_size, self.attack_dl.batch_size * (idx + 1))]

        # save each attacked image in its respective directory.
        for x, def_image in enumerate(reversed_inputs):
            def_image = self.visualizeable_image_2(def_image)
            path_components = self.attack_ds.im_paths[im_paths_idxes[x]].split("/")
            def_img_path = f"{self.save_def_image_folder}/{path_components[-2]}/{path_components[-1]}"

            # make the root folder "Attack" if it doesn't exist.
            if not os.path.isdir(self.save_def_image_folder):
                os.mkdir(self.save_def_image_folder)
            
            # make the label subfolder in "Attack" if it doesn't exist.
            if not os.path.isdir(f"{self.save_def_image_folder}/{path_components[-2]}"):
                os.mkdir(f"{self.save_def_image_folder}/{path_components[-2]}")

            plt.imsave(def_img_path, def_image)

    def run_defense(self):
        with open('./bird-data-defense-ddpm/unet/diffusion_pytorch_model.safetensors', 'rb') as f:
            model_bytes = f.read()    

        model_weights = safetensors.torch.load(model_bytes)
        noise_scheduler = DDPMScheduler(num_train_timesteps=self.backwards_steps)
        model = UNet2DModel.from_config("./bird-data-defense-ddpm/unet/config.json")
        model.load_state_dict(model_weights)

        model.eval()  # Set the model to evaluation mode
        model = model.to(device)

        # Run Forward:
        for batch_idx, batch in enumerate(self.attack_dl):
            start_imgs, labels = batch[0], batch[1]
            noises = torch.randn(start_imgs.shape)
            timesteps = torch.LongTensor([self.forwards_steps])
            noisy_images = noise_scheduler.add_noise(start_imgs, noises, timesteps)

            # print(start_img.shape)
            # print(noise.shape)
            # print(noisy_image.shape)
            # Run Reverse:
            # reverse_inputs = noisy_images.permute(0, 1, 3, 2)
            # reverse_inputs = reverse_inputs.to(device)
            reverse_inputs = noisy_images.to(device)

            for t in noise_scheduler.timesteps:
                with torch.no_grad():
                    model_output = model(reverse_inputs, torch.tensor([t], device=reverse_inputs.device).to(device))
                    noisy_residuals = model_output.sample
                    reverse_inputs = noise_scheduler.step(noisy_residuals, t, reverse_inputs).prev_sample
                
                if t % 1 == 0:
                    print(f"Batch {batch_idx + 1}/{len(self.attack_dl)}: timestep {self.backwards_steps - t}/{self.backwards_steps}")

            self.save_to_folder(batch_idx, reverse_inputs)
    

def run_original_defense(attack_ds_name="FGSM05-test", root="./bird-data"):
    '''
    Runs DDPM defense on attacked images (no input denoising transformations).
    '''
    attack_ds = BirdDataset(root, attack_ds_name)
    attack_dl = DataLoader(attack_ds, batch_size=24)
    tests = [1, 2, 5, 10, 20, 30, 40, 100]
    for backwards_steps in tests:
        defense = DDPMDefense(attack_ds, attack_dl, backwards_steps, f"{root}/{attack_ds_name}-def-t={backwards_steps}")
        defense.run_defense()

def run_wavelet_defense(attack_ds_name="FGSM05-wavelet-test", root="./bird-data/wavelet"):
    '''
    Runs DDPM defense on wavelet input images instead of "attacked" images.
    '''
    attack_ds = BirdDataset(root, attack_ds_name)
    attack_dl = DataLoader(attack_ds, batch_size=16)
    tests = [1, 2, 5, 10, 20, 30, 40, 100]
    for backwards_steps in tests:
        defense = DDPMDefense(attack_ds, attack_dl, backwards_steps, f"{root}/{attack_ds_name}-def-t={backwards_steps}")
        defense.run_defense()

def run_nlmeans_defense(attack_ds_name="FGSM05-nlmeans-test", root="./bird-data/nl-means"):
    '''
    Runs DDPM defense on nl-means denosied images instead of "attacked" images.
    '''
    attack_ds = BirdDataset(root, attack_ds_name)
    attack_dl = DataLoader(attack_ds, batch_size=24)
    tests = [20]
    for backwards_steps in tests:
        defense = DDPMDefense(attack_ds, attack_dl, backwards_steps, f"{root}/{attack_ds_name}-def-t={backwards_steps}")
        defense.run_defense()

if __name__ == "__main__":
    run_wavelet_defense(attack_ds_name="FGSM25-wavelet-test")
