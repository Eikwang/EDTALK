import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.archs import Discriminator, Generator, VGG16
from src.data_loader import ImageData, GazeDataset

from torchvision.utils import save_image

from utils.ops import l1_loss, content_loss, style_loss
from utils.eyes_catch import eyes_catch
from utils.paste import paste

from tqdm import tqdm
from itertools import cycle

#Learning Rate
D_LR = 5e-5
G_LR = 2e-4

#D's hyperparameter
D_ADV_WEIGHT = 1
D_REG_WEIGHT = 2

#G's hyperparameter
G_ADV_WEIGHT = 1
G_REG_WEIGHT = 5
G_RECON_WEIGHT = 1
G_FEAT_WEIGHT = 1500

AUGMENT = True

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.generator = Generator(style_dim=2)
        self.discriminator = Discriminator(params)

        if params.vgg_path:
            vgg_dict = torch.load(params.vgg_path, map_location=torch.device('cpu'), weights_only=True)
            self.vgg = VGG16(vgg_dict)
            print(f"Successfully loaded pretrained weights from {params.vgg_path}")

        self.params = params
        self.global_step = torch.tensor(0, dtype=torch.int32, requires_grad=False)
        self.lr = params.lr

        def init_weights(m):
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight) # Xavier initialization
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias) # Initialize bias to 0

        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

    def data_loader(self):

        hps = self.params

        image_data_class = ImageData(load_size=hps.image_size,
                                     channels=3,
                                     data_path=hps.data_path,
                                     ids=hps.ids,
                                     correction_ratio=getattr(hps, 'correction_ratio', 0.5))
        image_data_class.preprocess()

        train_dataset_num = len(image_data_class.train_images)
        test_dataset_num = len(image_data_class.test_images)

        tqdm.write(f"train dataset number: {train_dataset_num}")
        tqdm.write(f"test dataset number: {test_dataset_num}")

        train_dataset = GazeDataset(
            image_data_class.train_images,
            image_data_class.train_angles_r,
            image_data_class.train_labels,
            image_data_class.train_images_t,
            image_data_class.train_angles_g,
            image_data_class.train_sides,
            image_data_class,
            augment=AUGMENT
        )

        test_dataset = GazeDataset(
            image_data_class.test_images,
            image_data_class.test_angles_r,
            image_data_class.test_labels,
            image_data_class.test_images_t,
            image_data_class.test_angles_g,
            image_data_class.test_sides,
            image_data_class,
            augment=False
        )

        num_workers = 2
        pin_memory = True
        train_loader = DataLoader(train_dataset, batch_size=hps.batch_size,
                                  shuffle=True, num_workers=num_workers,
                                  drop_last=True, pin_memory=pin_memory,
                                  persistent_workers=True, prefetch_factor=2)
        test_loader = DataLoader(test_dataset, batch_size=hps.batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 drop_last=False, pin_memory=pin_memory,
                                 persistent_workers=True, prefetch_factor=2)

        return train_loader, test_loader, train_dataset_num
    
    def adv_loss(self, images_r, images_g, angles_r, angles_g):

        # Discriminator output for real and generated samples
        gan_real, reg_real = self.discriminator(images_r)
        gan_fake, reg_fake = self.discriminator(images_g)

        # Generate interpolated samples (use actual batch size, not hps.batch_size,
        # because test batches may be smaller than train batches)
        eps = torch.rand((images_r.size(0), 1, 1, 1), device=images_r.device)
        interpolated = eps * images_r + (1. - eps) * images_g
        interpolated = interpolated.requires_grad_()

        gan_inter, _ = self.discriminator(interpolated)

        # Compute gradient penalty
        grad = torch.autograd.grad(
            outputs=gan_inter,
            inputs=interpolated,
            grad_outputs=torch.ones_like(gan_inter),
            create_graph=True,
            retain_graph=True
        )[0]

        slopes = torch.sqrt(torch.sum(grad**2, dim=[1, 2, 3]))
        gp = torch.mean((slopes - 1)**2)

        # Discriminator Loss
        adv_d_loss = (-torch.mean(gan_real) +
                torch.mean(gan_fake) +
                10.0 * gp)

        # Generator Loss
        adv_g_loss = -torch.mean(gan_fake)

        # Regression Loss
        reg_d_loss = F.mse_loss(angles_r, reg_real)
        reg_g_loss = F.mse_loss(angles_g, reg_fake)

        return adv_d_loss, adv_g_loss, reg_d_loss, reg_g_loss, gp

    def feat_loss(self, image_g, image_t):

        content_layers = ["conv5"]
        style_layers = [
            "conv1",
            "conv2",
            "conv3",
            "conv4"
        ]

        # Normalize from [-1, 1] to ImageNet distribution for VGG
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=image_g.device).view(1, 3, 1, 1)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=image_g.device).view(1, 3, 1, 1)
        image_g_norm = (image_g + 1.0) / 2.0  # [-1,1] -> [0,1]
        image_g_norm = (image_g_norm - imagenet_mean) / imagenet_std
        image_t_norm = (image_t + 1.0) / 2.0
        image_t_norm = (image_t_norm - imagenet_mean) / imagenet_std

        # Force fp32 for the entire perceptual loss computation.
        # VGG features can be large (e.g. conv2 up to ~140), and the Gram
        # matrix bmm overflows in fp16 AMP. VGG parameters are frozen, so
        # gradients flow back to the generator without updating VGG.
        with torch.amp.autocast('cuda', enabled=False):
            _, end_points_image_g = self.vgg(image_g_norm.float())
            _, end_points_image_t = self.vgg(image_t_norm.float())
            c_loss = content_loss(end_points_image_g, end_points_image_t, content_layers)
            s_loss = style_loss(end_points_image_g, end_points_image_t, style_layers)

        return c_loss, s_loss

    def d_loss_calculator(self, images_r, angles_r, angles_g):

        images_g = self.generator(images_r, angles_g).detach()

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp) = self.adv_loss(images_r, images_g, angles_r, angles_g)

        if (self.it + 1) % 600 == 0:
            print(f"adv_d_loss = {self.adv_d_loss:<10.5f}")
            print(f"reg_d_loss = {self.reg_d_loss:<10.5f}")
            print(" ")


        return D_ADV_WEIGHT * self.adv_d_loss + D_REG_WEIGHT * self.reg_d_loss

    def g_loss_calculator(self, images_r, angles_r, images_t, angles_g):

        images_g = self.generator(images_r, angles_g)

        images_recon = self.generator(images_g, angles_r)

        # reconstruction loss
        self.recon_loss = l1_loss(images_r, images_recon)

        # content loss and style loss
        self.c_loss, self.s_loss = self.feat_loss(images_g, images_t)

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp) = self.adv_loss(images_r, images_g, angles_r, angles_g)

        # Binocular consistency loss: penalize large differences in horizontal gaze
        # between left and right eye pairs within the same batch.
        self.binocular_loss = torch.tensor(0.0, device=images_r.device)
        if hasattr(self, 'sides') and self.sides is not None:
            # sides is a list of 'L'/'R' strings for each sample in the batch
            # Try to pair samples with the same label (same identity/head_pose)
            labels = self.labels if hasattr(self, 'labels') else None
            if labels is not None:
                # Build a mapping from label to list of (side, angle_g) tuples
                from collections import defaultdict
                label_to_samples = defaultdict(list)
                for i in range(len(labels)):
                    label_to_samples[labels[i].item()].append((i, self.sides[i]))
                
                # For each label with both L and R samples, compute consistency loss
                for label, samples in label_to_samples.items():
                    l_samples = [s for s in samples if s[1] == 'L']
                    r_samples = [s for s in samples if s[1] == 'R']
                    if l_samples and r_samples:
                        # Average target angles for left and right eyes
                        l_angles = angles_g[l_samples[0][0]]  # Use first L sample
                        r_angles = angles_g[r_samples[0][0]]  # Use first R sample
                        # Penalize difference in horizontal gaze (index 0)
                        h_diff = torch.abs(l_angles[0] - r_angles[0])
                        self.binocular_loss = self.binocular_loss + h_diff
                
                # Average over number of pairs
                if len(label_to_samples) > 0:
                    self.binocular_loss = self.binocular_loss / len(label_to_samples)

        if (self.it + 1) % 600 == 0:
            print(f"adv_g_loss = {self.adv_g_loss:<10.5f}")
            print(f"reg_g_loss = {self.reg_g_loss:<10.5f}")
            print(f"recon_loss = {self.recon_loss:<10.5f}")
            print(f"feat_loss = {self.s_loss + self.c_loss:<10.5f}")
            print(f"binocular_loss = {self.binocular_loss:<10.5f}")
            print(" ")

        return G_ADV_WEIGHT * self.adv_g_loss + G_REG_WEIGHT * self.reg_g_loss + G_RECON_WEIGHT * self.recon_loss + \
                                        G_FEAT_WEIGHT * (self.s_loss + self.c_loss) + \
                                        2.0 * self.binocular_loss

    def test_d_loss(self, images_r, angles_r, angles_g):
        with torch.no_grad():
            images_g = self.generator(images_r, angles_g)
            gan_real, reg_real = self.discriminator(images_r)
            gan_fake, reg_fake = self.discriminator(images_g)
            adv_d_loss = -torch.mean(gan_real) + torch.mean(gan_fake)
            reg_d_loss = F.mse_loss(angles_r, reg_real)
            return D_ADV_WEIGHT * adv_d_loss + D_REG_WEIGHT * reg_d_loss

    def test_g_loss(self, images_r, angles_r, images_t, angles_g):
        with torch.no_grad():
            images_g = self.generator(images_r, angles_g)
            images_recon = self.generator(images_g, angles_r)
            recon_loss = l1_loss(images_r, images_recon)
            c_loss, s_loss = self.feat_loss(images_g, images_t)
            gan_fake, reg_fake = self.discriminator(images_g)
            adv_g_loss = -torch.mean(gan_fake)
            reg_g_loss = F.mse_loss(angles_g, reg_fake)
            return (G_ADV_WEIGHT * adv_g_loss + G_REG_WEIGHT * reg_g_loss +
                    G_RECON_WEIGHT * recon_loss + G_FEAT_WEIGHT * (s_loss + c_loss))
    
    def optimizer(self, model, lr):

        hps = self.params

        if hps.optimizer == 'sgd':
            return optim.SGD(model.parameters(), lr=lr)
        if hps.optimizer == 'adam':
            return optim.Adam(model.parameters(),
                            lr=lr,
                            betas=(hps.adam_beta1, hps.adam_beta2),
                            weight_decay=1e-3)
        raise AttributeError("attribute 'optimizer' is not assigned!")

    def add_optimizer(self):

        g_op = self.optimizer(self.generator, lr=G_LR)
        d_op = self.optimizer(self.discriminator, lr=D_LR)

        return d_op, g_op

    def train(self, conti = False):

        hps = self.params

        if torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            torch.cuda.set_device(0)
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        self.device = device

        # to device
        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)
        if hasattr(self, 'vgg'):
            self.vgg = self.vgg.to(device)

        self.d_op, self.g_op = self.add_optimizer()
        self.d_op.zero_grad()
        self.g_op.zero_grad()

        self.scaler = torch.amp.GradScaler('cuda')

        current_model_path = os.path.join(hps.log_dir, "current_model.ckpt")
        if conti:
            checkpoint = torch.load(current_model_path)
            self.generator.load_state_dict(checkpoint['current_generator'])
            self.discriminator.load_state_dict(checkpoint['current_discriminator'])
            self.g_op.load_state_dict(checkpoint['optimizer_generator'])
            self.d_op.load_state_dict(checkpoint['optimizer_discriminator'])
            print(f". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.Loaded previous model. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")

        best_model_loss = float('inf')

        (train_iter, test_iter, train_size) = self.data_loader() # Load iterators for training, validation, and test datasets

        num_epoch = hps.epochs
        self.num_epoch = num_epoch
        batch_size = hps.batch_size
        frequency = 3
        num_iter = train_size // batch_size # num_iter = 1102

        d_op_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.d_op, mode='min', factor=0.95, patience=3)
        g_op_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_op, mode='min', factor=0.95, patience=3)

        last_d_test_loss = float('inf')
        last_g_test_loss = float('inf')

        try:
            for epoch in range(num_epoch):
                self.epoch = epoch

                for it, (train_batch, test_batch) in enumerate(tqdm(zip(train_iter, cycle(test_iter)), total=num_iter, desc=f"Epoch: {epoch + 1}/{num_epoch}")):
                    self.it = it

                    train_batch = [t.to(device, non_blocking=True) for t in train_batch]
                    test_batch = [t.to(device, non_blocking=True) for t in test_batch]

                    self.x_r, self.angles_r, self.labels, self.x_t, self.angles_g, self.sides = train_batch
                    self.x_test_r, self.angles_test_r, self.labels_test, self.x_test_t, self.angles_test_g, self.sides_test = test_batch

                    try:
                        self.d_op.zero_grad()
                        with torch.amp.autocast('cuda'):
                            d_loss = self.d_loss_calculator(self.x_r, self.angles_r, self.angles_g)
                        self.scaler.scale(d_loss).backward()

                        if (it + 1) % frequency == 0:
                            self.scaler.unscale_(self.d_op)
                            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                            self.scaler.step(self.d_op)

                            self.g_op.zero_grad()
                            with torch.amp.autocast('cuda'):
                                g_loss = self.g_loss_calculator(self.x_r, self.angles_r, self.x_t, self.angles_g)
                            self.scaler.scale(g_loss).backward()
                            self.scaler.unscale_(self.g_op)
                            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                            self.scaler.step(self.g_op)
                            self.scaler.update()

                    except torch.cuda.OutOfMemoryError as e:
                        tqdm.write(f"GPU OOM at epoch {epoch+1} step {it+1}: {e}")
                        torch.cuda.empty_cache()
                        continue
                    except RuntimeError as e:
                        tqdm.write(f"CUDA error at epoch {epoch+1} step {it+1}: {e}")
                        if 'illegal memory access' in str(e) or 'device-side assert' in str(e):
                            raise
                        torch.cuda.empty_cache()
                        continue

                    if (it + 1) % hps.summary_steps == 0:
                        self.global_step = epoch * num_iter + it

                        d_test_loss = self.test_d_loss(self.x_test_r, self.angles_test_r, self.angles_test_g)
                        transformed_d_test_loss = torch.exp(d_test_loss / 100)
                        g_test_loss = self.test_g_loss(self.x_test_r, self.angles_test_r, self.x_test_t, self.angles_test_g)
                        transformed_g_test_loss = torch.exp(g_test_loss / 100)

                        tqdm.write(f"generator test loss: {transformed_g_test_loss:<10.2f}, discriminator test loss: {transformed_d_test_loss:<10.2f}")

                        challenger_loss = (transformed_g_test_loss + transformed_d_test_loss * (transformed_g_test_loss + transformed_d_test_loss))
                        if challenger_loss < best_model_loss:
                            best_model_loss = challenger_loss
                            tqdm.write(f". ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.New lowest loss at step: {self.global_step}. ݁₊ ⊹ . ݁ ⟡ ݁ . ⊹ ₊ ݁.")

                        last_d_test_loss = transformed_d_test_loss.item()
                        last_g_test_loss = transformed_g_test_loss.item()

                        torch.save({
                                'current_generator': self.generator.state_dict(),
                                'current_discriminator': self.discriminator.state_dict(),
                                'optimizer_generator': self.g_op.state_dict(),
                                'optimizer_discriminator': self.d_op.state_dict()
                            }, current_model_path)

                        del d_test_loss, transformed_d_test_loss, g_test_loss, transformed_g_test_loss, challenger_loss
                        torch.cuda.empty_cache()

                # Adjust learning rate at the end of each epoch
                d_op_scheduler.step(last_d_test_loss)
                g_op_scheduler.step(last_g_test_loss)

        except KeyboardInterrupt:
            print("Training interrupted.")

    def eval(self):

        hps = self.params

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Eval using device: {device}")

        self.generator = self.generator.to(device)
        self.discriminator = self.discriminator.to(device)

        checkpoint_path = os.path.join(hps.log_dir, 'current_model.ckpt')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.generator.load_state_dict(checkpoint['current_generator'])
        self.discriminator.load_state_dict(checkpoint['current_discriminator'])

        # Set models to eval mode to disable dropout
        self.generator.eval()
        self.discriminator.eval()

        eval_dir = os.path.join(hps.log_dir, 'eval')
        os.makedirs(eval_dir, exist_ok=True)
        
        for file_name in os.listdir(hps.client_pictures_dir): # for each picture
            each_eval_dir = os.path.join(eval_dir, f'eval_{file_name}')
            os.makedirs(each_eval_dir, exist_ok=True)

            picture_eyes_patch, eyes_position, size = eyes_catch(hps, file_name) # picture_eyes_patch.shape = [eyes' number, 3, 64, 64]

            # Move input data to device
            picture_eyes_patch = picture_eyes_patch.to(device)

            '''========== Gaze Adjustment =========='''
            # Training data_loader flips right eye h_angle: h_angle = flip * raw_h / 15
            # where flip=-1 for right eye, flip=1 for left eye.
            # To match training coordinate system, eval must apply the same flip.
            gaze_angles = torch.zeros(len(picture_eyes_patch), 2, device=device)
            for i in range(len(picture_eyes_patch)):
                if i % 2 == 1:  # right eye (odd index)
                    # Apply flip=-1 to horizontal angle to match training
                    gaze_angles[i] = torch.tensor([0.0 * -1, 0.0], device=device)
                else:  # left eye
                    gaze_angles[i] = torch.tensor([0.0, 0.0], device=device)
            '''========== Gaze Adjustment =========='''

            with torch.no_grad():
                generated_image = self.generator(picture_eyes_patch, gaze_angles) # generated_image.shape = [number, 3, 64, 64]
                
                print(file_name)
                test_gan, test_reg = self.discriminator(picture_eyes_patch)
                print(f"original: gan = {test_gan}, reg = {test_reg}")
                test_gan, test_reg = self.discriminator(generated_image)
                print(f"generated: gan = {test_gan}, reg = {test_reg}")
            
            # Move generated images back to CPU for saving
            generated_image = generated_image.cpu()
            paste(hps, file_name, generated_image, eyes_position, size)
            
            for i, new_img in enumerate(generated_image):
                if i % 2 == 0: # left eye
                    file_name = os.path.join(each_eval_dir, f'generated_{i // 2}_L.png')
                    save_image(new_img, file_name)
                else: # right eye
                    file_name = os.path.join(each_eval_dir, f'generated_{i // 2}_R.png')
                    save_image(new_img, file_name)

        print("Evaluation finished.")
        