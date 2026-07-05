import os
import glob
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PerImageNormalize:
    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = x.std(dim=(1, 2), keepdim=True)
        return (x - mean) / (std + self.eps)


def build_transforms(use_grayscale=True, mean=None, std=None, training=True):
    ops = []
    ops.append(T.Resize((224, 224)))
    
    if training:
        if use_grayscale:
            ops.append(T.Grayscale(num_output_channels=3))
        ops.append(T.RandomHorizontalFlip(p=0.5))
        ops.append(T.RandomRotation(degrees=5))
        ops.append(T.RandomAffine(degrees=0, translate=(0.02, 0.02)))
        ops.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0, hue=0))
    else:
        if use_grayscale:
            ops.append(T.Grayscale(num_output_channels=3))

    ops.append(T.ToTensor())
    if training:
        ops.append(T.RandomErasing(p=0.1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'))
    ops.append(PerImageNormalize(eps=1e-6))
    return T.Compose(ops)


class ThyroidDataset(Dataset):
    """Thyroid ultrasound dataset with optional protein features."""
    
    def __init__(self, data_dir='../data',
                 clinical_file='stageA_clinical.xlsx', protein_file='stageA_protein.xlsx',
                 transform=None, training=True, fold=0, n_folds=5, image_mode='RGB',
                 load_protein=True, proto_assign_file=None,
                 image_subdir='stageA_ultrasound'):
        self.data_dir = data_dir
        self.image_subdir = image_subdir
        self.transform = transform if transform is not None else build_transforms(
            use_grayscale=(image_mode == 'L'),
            training=training
        )
        self.training = training
        self.image_mode = image_mode
        self.load_protein = load_protein
        
        self.proto_id_map = {}
        if proto_assign_file is not None and os.path.isfile(proto_assign_file):
            assign_df = pd.read_csv(proto_assign_file)
            assign_df['ID'] = assign_df['ID'].astype(str).str.strip()
            for _, row in assign_df.iterrows():
                pid = row['ID']
                cluster = int(row['cluster'])
                proto_id = 0 if cluster == -1 else (cluster + 1)
                self.proto_id_map[pid] = proto_id
            print(f"[Dataset] loaded prototype labels for {len(self.proto_id_map)} patients")
        
        clinical_path = os.path.join(data_dir, clinical_file)
        if clinical_file.lower().endswith('.csv'):
            self.clinical_df = pd.read_csv(clinical_path)
        else:
            self.clinical_df = pd.read_excel(clinical_path)
        if 'ID' in self.clinical_df.columns:
            self.clinical_df['ID'] = self.clinical_df['ID'].astype(str).str.strip()
        
        protein_path = os.path.join(data_dir, protein_file)
        if self.load_protein:
            if protein_file.lower().endswith('.csv'):
                self.protein_df = pd.read_csv(protein_path)
            else:
                self.protein_df = pd.read_excel(protein_path)
            if 'ID' in self.protein_df.columns:
                self.protein_df['ID'] = self.protein_df['ID'].astype(str).str.strip()
                self.protein_df.set_index('ID', inplace=True)
            else:
                first_col = self.protein_df.columns[0]
                self.protein_df[first_col] = self.protein_df[first_col].astype(str).str.strip()
                self.protein_df.set_index(first_col, inplace=True)
            self.protein_dim = self.protein_df.shape[1]
        else:
            self.protein_df = None
            self.protein_dim = 680  # Placeholder matching the released ProPRINT checkpoint.
        
        self._create_stratified_folds(n_folds)
        
        if training:
            initial_df = self.clinical_df[self.clinical_df['fold'] != fold].reset_index(drop=True)
        else:
            initial_df = self.clinical_df[self.clinical_df['fold'] == fold].reset_index(drop=True)
        
        self.set_dataframe(initial_df)
        
    def _create_stratified_folds(self, n_folds=5):
        from sklearn.model_selection import StratifiedKFold
        
        if 'fold' not in self.clinical_df.columns:
            labels = self.clinical_df['Label'].values if 'Label' in self.clinical_df.columns else self.clinical_df.iloc[:, 1].values
            
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            folds = np.zeros(len(self.clinical_df))
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                folds[val_idx] = fold_idx
            
            self.clinical_df['fold'] = folds.astype(int)

    def set_dataframe(self, df):
        patient_df = df.copy().reset_index(drop=True)
        if 'ID' in patient_df.columns:
            patient_df['ID'] = patient_df['ID'].astype(str).str.strip()
        self.patient_df = patient_df

        if self.training:
            self.df = patient_df.reset_index(drop=True)
            return

        rows = []
        for _, row in patient_df.iterrows():
            pid = str(row['ID']).strip()
            paths = self._get_image_paths(pid)
            if not paths:
                row_dict = row.to_dict()
                row_dict['_image_path'] = None
                rows.append(row_dict)
                continue
            for path in paths:
                row_dict = row.to_dict()
                row_dict['_image_path'] = path
                rows.append(row_dict)
        self.df = pd.DataFrame(rows).reset_index(drop=True)
    
    def _get_image_paths(self, patient_id):
        image_dir = os.path.join(self.data_dir, self.image_subdir)
        pattern = os.path.join(image_dir, f"{patient_id}_*.jpg")
        paths = sorted(glob.glob(pattern))
        
        if not paths:
            pattern = os.path.join(image_dir, f"{patient_id}*.jpg")
            paths = sorted(glob.glob(pattern))
        
        return paths
    
    def _load_image(self, patient_id, image_path=None):
        if image_path is None:
            paths = self._get_image_paths(patient_id)
        else:
            paths = [image_path]

        if not paths or paths[0] is None:
            print(f"[WARNING] No image found for patient {patient_id}; using a zero image")
            return torch.zeros(3, 224, 224)

        if self.training and image_path is None:
            path = random.choice(paths)
        else:
            path = paths[0]
        
        try:
            image = Image.open(path).convert(self.image_mode)
            if self.transform:
                image = self.transform(image)
            return image
        except Exception as e:
            print(f"[WARNING] Failed to load image {path}: {e}")
            return torch.zeros(3, 224, 224)
    
    def _load_protein(self, patient_id):
        if self.protein_df is None:
            return torch.zeros(self.protein_dim)
        try:
            protein = self.protein_df.loc[str(patient_id)].values.astype(np.float32)
            protein = np.nan_to_num(protein, nan=0.0)
            # protein.xlsx is expected to contain the preprocessed protein matrix
            # generated according to Supplementary Method 2.
            return torch.tensor(protein)
        except KeyError:
            print(f"[WARNING] No protein row found for patient {patient_id}; using a zero vector")
            return torch.zeros(self.protein_dim)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['ID']
        image_path = row.get('_image_path', None)
        
        image = self._load_image(patient_id, image_path=image_path)
        protein = self._load_protein(patient_id)
        label = int(row['Label']) if 'Label' in row else int(row.iloc[1])
        proto_id = self.proto_id_map.get(str(patient_id), -1)
        return image, protein, label, patient_id, proto_id


class ThyroidDataModule:
    """Stage A dataloader wrapper."""
    
    def __init__(self, data_dir='../data',
                 clinical_file='stageA_clinical.xlsx', protein_file='stageA_protein.xlsx',
                 batch_size=16, num_workers=4, fold=0, n_folds=5,
                 use_grayscale=True, use_dataset_stats=True,
                 split_by_dataset=False, dataset_train_tag='finetune', dataset_val_tag='external',
                 load_protein=True, internal_finetune_split=False, finetune_train_ratio=0.8,
                 proto_assign_file=None, image_subdir='stageA_ultrasound'):
        self.data_dir = data_dir
        self.image_subdir = image_subdir
        self.clinical_file = clinical_file
        self.protein_file = protein_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fold = fold
        self.n_folds = n_folds
        self.use_grayscale = use_grayscale
        self.use_dataset_stats = use_dataset_stats
        self.image_mode = 'L' if use_grayscale else 'RGB'
        self.internal_finetune_split = internal_finetune_split
        self.finetune_train_ratio = finetune_train_ratio
        self.proto_assign_file = proto_assign_file

        clinical_path = os.path.join(data_dir, clinical_file)
        if clinical_file.lower().endswith('.csv'):
            cdf = pd.read_csv(clinical_path)
        else:
            cdf = pd.read_excel(clinical_path)
        if 'ID' in cdf.columns:
            cdf['ID'] = cdf['ID'].astype(str).str.strip()
        else:
            first_col = cdf.columns[0]
            cdf[first_col] = cdf[first_col].astype(str).str.strip()
            cdf = cdf.rename(columns={first_col: 'ID'})

        dataset_col = None
        for col in cdf.columns:
            if str(col).lower() == 'dataset':
                dataset_col = col
                break

        if split_by_dataset and dataset_col is not None:
            cdf[dataset_col] = cdf[dataset_col].astype(str).str.strip().str.lower()
            if internal_finetune_split and (dataset_train_tag == dataset_val_tag):
                subset = cdf[cdf[dataset_col] == dataset_train_tag].reset_index(drop=True)
                labels = subset['Label'].values if 'Label' in subset.columns else subset.iloc[:, 1].values
                try:
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=max(0.0, min(1.0, 1.0 - float(self.finetune_train_ratio))), random_state=42)
                    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
                    train_ids = subset.iloc[train_idx]['ID'].tolist()
                    val_ids = subset.iloc[val_idx]['ID'].tolist()
                except Exception:
                    rng = np.random.RandomState(42)
                    idx = np.arange(len(labels))
                    rng.shuffle(idx)
                    split = int(len(idx) * float(self.finetune_train_ratio))
                    train_ids = subset.iloc[idx[:split]]['ID'].tolist()
                    val_ids = subset.iloc[idx[split:]]['ID'].tolist()
            else:
                train_ids = cdf[cdf[dataset_col] == dataset_train_tag]['ID'].tolist()
                val_ids = cdf[cdf[dataset_col] == dataset_val_tag]['ID'].tolist()
        else:
            if 'fold' not in cdf.columns:
                from sklearn.model_selection import StratifiedKFold
                labels = cdf['Label'].values if 'Label' in cdf.columns else cdf.iloc[:, 1].values
                skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                folds = np.zeros(len(cdf))
                for fold_idx, (_, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
                    folds[val_idx] = fold_idx
                cdf['fold'] = folds.astype(int)
            train_ids = cdf[cdf['fold'] != fold]['ID'].tolist()
            val_ids = cdf[cdf['fold'] == fold]['ID'].tolist()

        mean = None
        std = None

        train_transform = build_transforms(
            use_grayscale=self.use_grayscale,
            mean=mean,
            std=std,
            training=True
        )
        val_transform = build_transforms(
            use_grayscale=self.use_grayscale,
            mean=mean,
            std=std,
            training=False
        )

        self.train_dataset = ThyroidDataset(
            data_dir=data_dir,
            clinical_file=clinical_file,
            protein_file=protein_file,
            transform=train_transform,
            training=True,
            fold=fold,
            n_folds=n_folds,
            image_mode=self.image_mode,
            load_protein=load_protein,
            proto_assign_file=proto_assign_file,
            image_subdir=image_subdir,
        )

        self.val_dataset = ThyroidDataset(
            data_dir=data_dir,
            clinical_file=clinical_file,
            protein_file=protein_file,
            transform=val_transform,
            training=False,
            fold=fold,
            n_folds=n_folds,
            image_mode=self.image_mode,
            load_protein=load_protein,
            proto_assign_file=proto_assign_file,
            image_subdir=image_subdir,
        )

        if split_by_dataset and dataset_col is not None:
            if internal_finetune_split and (dataset_train_tag == dataset_val_tag):
                self.train_dataset.set_dataframe(cdf[cdf['ID'].isin(train_ids)].reset_index(drop=True))
                self.val_dataset.set_dataframe(cdf[cdf['ID'].isin(val_ids)].reset_index(drop=True))
            else:
                self.train_dataset.set_dataframe(cdf[cdf[dataset_col] == dataset_train_tag].reset_index(drop=True))
                self.val_dataset.set_dataframe(cdf[cdf[dataset_col] == dataset_val_tag].reset_index(drop=True))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
