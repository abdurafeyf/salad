import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T
from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

# Split cities into training and validation sets
TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL',
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT',
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG',
    'WashingtonDC',
    'Brussels'
]

# Use the last few cities for validation
VAL_CITIES = [
    'London',
    'Melbourne',
    'Osaka',
    'PRS'
]

class GSVCitiesDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=4,
                 show_data_stats=True,
                 train_cities=TRAIN_CITIES,
                 val_cities=VAL_CITIES,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_place=True
                 ):
        super().__init__()
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.train_cities = train_cities
        self.val_cities = val_cities
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        
        # Save hyperparameters with PyTorch Lightning
        self.save_hyperparameters()
        
        # Define transforms
        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        
        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])
        
        # DataLoader configurations
        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all
        }
        
        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False
        }

    def setup(self, stage):
        if stage == 'fit':
            # Load train dataset with reload routine
            self.reload()
            
            # Create validation datasets using GSVCities validation cities
            self.val_datasets = []
            
            # Create validation dataset from GSVCities validation cities
            val_dataset = GSVCitiesDataset(
                cities=self.val_cities,
                img_per_place=self.img_per_place,
                min_img_per_place=self.min_img_per_place,
                random_sample_from_each_place=self.random_sample_from_each_place,
                transform=self.valid_transform
            )
            self.val_datasets.append(val_dataset)
            
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        """Reload training dataset"""
        self.train_dataset = GSVCitiesDataset(
            cities=self.train_cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()  # print a new line
        
        # Training dataset stats
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of training cities", f"{len(self.train_cities)}"])
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        print(table.get_string(title="Training Dataset"))
        print()
        
        # Validation dataset stats
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of validation cities", f"{len(self.val_cities)}"])
        table.add_row(["Validation cities", f"{', '.join(self.val_cities)}"])
        table.add_row(["# of validation places", f'{self.val_datasets[0].__len__()}'])
        table.add_row(["# of validation images", f'{self.val_datasets[0].total_nb_images}'])
        print(table.get_string(title="Validation Dataset"))
        print()
        
        # Training configuration stats
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training Config"))
        print()
