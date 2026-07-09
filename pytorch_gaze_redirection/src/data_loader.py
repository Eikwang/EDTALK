import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from PIL import Image
import torchvision.transforms as transforms


class BinocularPairSampler(Sampler):
    """确保每个 batch 中同一人的左右眼成对出现。

    每两个连续索引构成一个 (L, R) 对，DataLoader 按 batch_size 分组后，
    每个 batch 自然包含 batch_size//2 个 L-R 对，使 binocular_loss 稳定生效。
    """

    def __init__(self, labels, sides, batch_size):
        label_to_lr = {}
        for i, (label, side) in enumerate(zip(labels, sides)):
            if label not in label_to_lr:
                label_to_lr[label] = {'L': [], 'R': []}
            label_to_lr[label][side].append(i)

        self.valid_labels = [
            l for l in label_to_lr
            if label_to_lr[l]['L'] and label_to_lr[l]['R']
        ]
        self.label_to_lr = label_to_lr
        self.batch_size = batch_size
        total_pairs = sum(
            min(len(label_to_lr[l]['L']), len(label_to_lr[l]['R']))
            for l in self.valid_labels
        )
        self._length = (total_pairs * 2 // batch_size) * batch_size

    def __iter__(self):
        label_list = list(self.valid_labels)
        random.shuffle(label_list)

        indices = []
        li = 0
        while len(indices) < self._length:
            label = label_list[li % len(label_list)]
            lr = self.label_to_lr[label]
            l_idx = random.choice(lr['L'])
            r_idx = random.choice(lr['R'])
            indices.extend([l_idx, r_idx])
            li += 1

        return iter(indices[:self._length])

    def __len__(self):
        return self._length


class GazeDataset(Dataset):
    """按需加载图像的 Dataset，带解码缓存避免重复磁盘IO。"""

    def __init__(self, images, angles_r, labels, images_t, angles_g, sides,
                 image_data_class, augment):
        self.angles_r = angles_r
        self.labels = labels
        self.angles_g = angles_g
        self.sides = sides
        self.image_data_class = image_data_class
        self.augment = augment

        all_paths = set(images + images_t)
        print(f"Preloading {len(all_paths)} unique images into memory...")
        self._cache = {}
        for p in all_paths:
            img = Image.open(p).convert("RGB" if image_data_class.channels == 3 else "L")
            self._cache[p] = np.array(img)
        self._img_keys = images
        self._img_t_keys = images_t
        print(f"Preloading done. Cache size: ~{len(self._cache) * 12 // 1024}MB")

    def __len__(self):
        return len(self._img_keys)

    def __getitem__(self, idx):
        img_arr = self._cache[self._img_keys[idx]]
        img_t_arr = self._cache[self._img_t_keys[idx]]

        image = self.image_data_class.transform(Image.fromarray(img_arr))
        image_t = self.image_data_class.transform(Image.fromarray(img_t_arr))

        if self.sides[idx] == 'R':
            image = torch.flip(image, dims=[-1])
            image_t = torch.flip(image_t, dims=[-1])

        if self.augment:
            image = self.image_data_class.augmentation(image)

        return (
            image,
            torch.tensor(self.angles_r[idx], dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
            image_t,
            torch.tensor(self.angles_g[idx], dtype=torch.float32),
            self.sides[idx]
        )


class ImageData(object):

    def __init__(self, load_size, channels, data_path, ids, correction_ratio=0.5):

        self.load_size = load_size
        self.channels = channels
        self.ids = ids
        self.correction_ratio = correction_ratio

        self.data_path = data_path

        self.transform = transforms.Compose([
            transforms.Resize((load_size, load_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * channels, std=[0.5] * channels)
        ])
        self.augmentation = transforms.Compose([
            transforms.RandomErasing(p=0.3, scale=(0.05, 0.2))
        ])
        file_names = [f for f in os.listdir(data_path)
                       if f.endswith('.jpg')]

        self.file_dict = {}
        for f_name in file_names:
        # Extract attributes: identity, head_pose, side
            fields = f_name.split('.')[0].split('_')
            identity = fields[0]
            head_pose = fields[2]
            side = fields[-1]
            key = '_'.join([identity, head_pose, side])
            if key not in self.file_dict:
                self.file_dict[key] = []
            self.file_dict[key].append(f_name) # f_name is the full name of the photo

        self.train_images = []
        self.train_angles_r = []
        self.train_labels = []
        self.train_images_t = []
        self.train_angles_g = []
        self.train_sides = []  # 'L' or 'R'

        self.test_images = []
        self.test_angles_r = []
        self.test_labels = []
        self.test_images_t = []
        self.test_angles_g = []
        self.test_sides = []  # 'L' or 'R'

    def image_processing(
        self,
        filename,
        angles_r,
        labels,
        filename_t,
        angles_g,
        augment
    ):
        
        def _to_image(file_name, augment):
            img = Image.open(file_name).convert("RGB" if self.channels == 3 else "L")
            img = self.transform(img)
            if augment:
                img = self.augmentation(img)
            return img

        image = _to_image(filename, augment)
        image_t = _to_image(filename_t, augment=False)

        return image, angles_r, labels, image_t, angles_g
    
    def preprocess(self):

        for key, file_list in self.file_dict.items(): # Photos of the same person from different angles
            if len(file_list) == 1: # Skip if there is only one image
                continue

            idx = int(key.split('_')[0])
            flip = -1

            for f_r in file_list: # For each photo of a key (e.g. 0010_0P_R)
                file_path = os.path.join(self.data_path, f_r)
                h_angle_r = flip * float(f_r.split('_')[-2].split('H')[0]) / 15.0 # Horizontal angle
                v_angle_r = float(f_r.split('_')[-3].split('V')[0]) / 10.0 # Vertical angle

                for f_g in file_list:
                    file_path_t = os.path.join(self.data_path, f_g)
                    h_angle_g = flip * float(f_g.split('_')[-2].split('H')[0]) / 15.0
                    v_angle_g = float(f_g.split('_')[-3].split('V')[0]) / 10.0

                    side_info = key.split('_')[-1]  # 'L' or 'R'
                    if idx <= self.ids: # Training set
                        self.train_images.append(file_path)
                        self.train_angles_r.append([h_angle_r, v_angle_r])
                        self.train_labels.append(idx - 1)
                        self.train_images_t.append(file_path_t)
                        self.train_angles_g.append([h_angle_g, v_angle_g])
                        self.train_sides.append(side_info)
                    else: # Test set
                        self.test_images.append(file_path)
                        self.test_angles_r.append([h_angle_r, v_angle_r])
                        self.test_labels.append(idx - 1)
                        self.test_images_t.append(file_path_t)
                        self.test_angles_g.append([h_angle_g, v_angle_g])
                        self.test_sides.append(side_info)

        # Correction mode: force target gaze to (0, 0) for a subset of training pairs.
        # This teaches the model to redirect arbitrary gaze back to center.
        num_train = len(self.train_images)
        num_correction = int(num_train * self.correction_ratio)
        # Select evenly distributed samples to avoid bias toward early identities.
        indices = list(range(num_train))
        step = max(num_train // num_correction, 1) if num_correction > 0 else 1
        correction_indices = indices[::step][:num_correction]

        for idx_corr in correction_indices:
            src_path = self.train_images[idx_corr]
            src_name = os.path.basename(src_path)
            fields = src_name.split('.')[0].split('_')
            # fields: [identity, 2m, pose, V, H, side]
            identity, _, head_pose, _, _, side = fields
            target_file = '_'.join([identity, '2m', head_pose, '0V', '0H', side]) + '.jpg'
            target_path = os.path.join(self.data_path, target_file)
            if not os.path.exists(target_path):
                continue

            self.train_images_t[idx_corr] = target_path
            self.train_angles_g[idx_corr] = [0.0, 0.0]

        correction_count = sum([1 for ag in self.train_angles_g if ag == [0.0, 0.0]])

        if len(self.test_images) == 0 and len(self.train_images) > 0:
            test_ratio = 0.1
            num_test = max(1, int(len(self.train_images) * test_ratio))
            indices = list(range(len(self.train_images)))
            random.shuffle(indices)
            test_indices = set(indices[:num_test])
            for i in sorted(test_indices, reverse=True):
                self.test_images.append(self.train_images.pop(i))
                self.test_angles_r.append(self.train_angles_r.pop(i))
                self.test_labels.append(self.train_labels.pop(i))
                self.test_images_t.append(self.train_images_t.pop(i))
                self.test_angles_g.append(self.train_angles_g.pop(i))
                self.test_sides.append(self.train_sides.pop(i))
            print(f'No test data (ids >= max identity). Moved {num_test} samples from train to test.')

        num_train = len(self.train_images)
        print(f'\nFinished preprocessing the dataset... correction pairs: {correction_count}/{num_train}')
        print(f'train pairs: {num_train}, test pairs: {len(self.test_images)}')