import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.archs import Discriminator, Generator, VGG16
from src.data_loader import ImageData, GazeDataset, BinocularPairSampler

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

        # num_workers=0: Windows spawn pickles the entire _cache (all images)
        # to each worker process, causing memory pressure and worker crashes
        # after several epochs. Since images are preloaded in-memory, single-
        # process data loading is fast enough and eliminates this instability.
        num_workers = 0
        pin_memory = True
        train_sampler = BinocularPairSampler(
            image_data_class.train_labels,
            image_data_class.train_sides,
            batch_size=hps.batch_size
        )
        train_loader = DataLoader(train_dataset, batch_size=hps.batch_size,
                                  sampler=train_sampler, num_workers=num_workers,
                                  drop_last=True, pin_memory=pin_memory)
        test_loader = DataLoader(test_dataset, batch_size=hps.batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 drop_last=False, pin_memory=pin_memory)

        return train_loader, test_loader, len(train_sampler)
    
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

        return adv_d_loss, adv_g_loss, reg_d_loss, reg_g_loss, gp, reg_fake

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

        _, end_points_image_g = self.vgg(image_g_norm)
        _, end_points_image_t = self.vgg(image_t_norm)
        c_loss = content_loss(end_points_image_g, end_points_image_t, content_layers)
        s_loss = style_loss(end_points_image_g, end_points_image_t, style_layers)

        return c_loss, s_loss

    def d_loss_calculator(self, images_r, angles_r, angles_g):

        images_g = self.generator(images_r, angles_g).detach()

        # regression losses and adversarial losses
        (self.adv_d_loss, self.adv_g_loss, self.reg_d_loss,
        self.reg_g_loss, self.gp, _) = self.adv_loss(images_r, images_g, angles_r, angles_g)

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
        self.reg_g_loss, self.gp, reg_fake) = self.adv_loss(images_r, images_g, angles_r, angles_g)

        # Binocular consistency loss (vectorized).
        # BinocularPairSampler guarantees [L, R, L, R, ...] alternating order,
        # so even indices are L (person's right eye, flip=-1) and odd indices
        # are R (person's left eye, no flip). For the same gaze direction,
        # L's h prediction = -h and R's h prediction = +h, so their sum -> 0.
        self.binocular_loss = torch.zeros(1, device=images_r.device)
        if reg_fake.size(0) >= 2 and reg_fake.size(0) % 2 == 0:
            l_preds = reg_fake[0::2]
            r_preds = reg_fake[1::2]
            self.binocular_loss = torch.mean(torch.abs(l_preds[:, 0] + r_preds[:, 0]))

        if (self.it + 1) % 600 == 0:
            print(f"adv_g_loss = {self.adv_g_loss:<10.5f}")
            print(f"reg_g_loss = {self.reg_g_loss:<10.5f}")
            print(f"recon_loss = {self.recon_loss:<10.5f}")
            print(f"feat_loss = {self.s_loss + self.c_loss:<10.5f}")
            print(f"binocular_loss = {self.binocular_loss.item():<10.5f}")
            print(" ")

        return G_ADV_WEIGHT * self.adv_g_loss + G_REG_WEIGHT * self.reg_g_loss + G_RECON_WEIGHT * self.recon_loss + \
                                        G_FEAT_WEIGHT * (self.s_loss + self.c_loss) + \
                                        5.0 * self.binocular_loss

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

        has_test_data = len(test_iter.dataset) > 0 if hasattr(test_iter, 'dataset') else False

        try:
            for epoch in range(num_epoch):
                self.epoch = epoch

                if has_test_data:
                    batch_iter = zip(train_iter, cycle(test_iter))
                else:
                    batch_iter = ((batch, None) for batch in train_iter)

                for it, (train_batch, test_batch) in enumerate(tqdm(batch_iter, total=num_iter, desc=f"Epoch: {epoch + 1}/{num_epoch}")):
                    self.it = it

                    # Move tensors to device; sides (tuple of str) has no .to()
                    tensors = [t for t in train_batch if isinstance(t, torch.Tensor)]
                    sides = [t for t in train_batch if not isinstance(t, torch.Tensor)]
                    train_batch_moved = [t.to(device, non_blocking=True) for t in tensors] + sides
                    self.x_r, self.angles_r, self.labels, self.x_t, self.angles_g, self.sides = train_batch_moved

                    if test_batch is not None:
                        tensors_t = [t for t in test_batch if isinstance(t, torch.Tensor)]
                        sides_t = [t for t in test_batch if not isinstance(t, torch.Tensor)]
                        test_batch_moved = [t.to(device, non_blocking=True) for t in tensors_t] + sides_t
                        self.x_test_r, self.angles_test_r, self.labels_test, self.x_test_t, self.angles_test_g, self.sides_test = test_batch_moved

                    try:
                        self.d_op.zero_grad()
                        d_loss = self.d_loss_calculator(self.x_r, self.angles_r, self.angles_g)
                        d_loss.backward()

                        if (it + 1) % frequency == 0:
                            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                            self.d_op.step()

                            self.g_op.zero_grad()
                            g_loss = self.g_loss_calculator(self.x_r, self.angles_r, self.x_t, self.angles_g)
                            g_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                            self.g_op.step()

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

                        if has_test_data and test_batch is not None:
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

                            del d_test_loss, transformed_d_test_loss, g_test_loss, transformed_g_test_loss, challenger_loss
                        else:
                            tqdm.write(f"Step {self.global_step}: no test data, skipping test evaluation.")

                        torch.save({
                                'current_generator': self.generator.state_dict(),
                                'current_discriminator': self.discriminator.state_dict(),
                                'optimizer_generator': self.g_op.state_dict(),
                                'optimizer_discriminator': self.d_op.state_dict()
                            }, current_model_path)

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
            # Dataset L/R is image-side: L = image left = person's right eye,
            # R = image right = person's left eye.
            # Training flip: L gets flip=-1, R gets flip=1 (no flip).
            # eyes_catch order: index 0 = person's left eye (=dataset R, no flip),
            #                   index 1 = person's right eye (=dataset L, flip=-1).
            gaze_angles = torch.zeros(len(picture_eyes_patch), 2, device=device)
            for i in range(len(picture_eyes_patch)):
                if i % 2 == 1:  # person's right eye (=dataset L, flip=-1)
                    gaze_angles[i] = torch.tensor([0.0 * -1, 0.0], device=device)
                else:  # person's left eye (=dataset R, no flip)
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
        