#import
from glob import glob
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyVOCSegmentation, MyImageFolder, ImageLightningDataModule
from typing import Optional, Callable
import numpy as np
from typing import Tuple, Any
from PIL import Image
from os.path import join
import random
import torch
from torchvision.datasets.folder import IMG_EXTENSIONS


#def
def create_datamodule(project_parameters):
    if project_parameters.predefined_dataset:
        dataset_class = eval('My{}'.format(
            project_parameters.predefined_dataset))
    else:
        dataset_class = MyImageFolder
    return ImageLightningDataModule(
        root=project_parameters.root,
        predefined_dataset=project_parameters.predefined_dataset,
        classes=project_parameters.classes,
        max_samples=project_parameters.max_samples,
        batch_size=project_parameters.batch_size,
        num_workers=project_parameters.num_workers,
        device=project_parameters.device,
        transforms_config=project_parameters.transforms_config,
        target_transforms_config=project_parameters.target_transforms_config,
        dataset_class=dataset_class)


#class
class MyImageFolder(MyImageFolder):
    def __init__(self,
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        super().__init__(root,
                         transform=transform,
                         target_transform=target_transform)
        self.images = []
        self.masks = []
        for ext in IMG_EXTENSIONS:
            self.images += glob(join(root, f'image/*{ext}'))
            self.masks += glob(join(root, f'mask/*{ext}'))
        self.images = sorted(self.images)
        self.masks = sorted(self.masks)
        self.classes = np.loadtxt(fname=join(
            root.rsplit('/', 1)[0], 'classes.txt'),
                                  dtype=str).tolist()
        self.class_to_idx = {k: v for v, k in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])
        if self.transform is not None:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(new_state=state)
            target = self.transform(target)
        if target.dtype == torch.float32:
            target = (target * 255).round().long()
            target[target == 255] = 0
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def decrease_samples(self, max_samples):
        if max_samples is not None:
            index = random.sample(population=range(len(self.images)),
                                  k=max_samples)
            self.images = np.array(self.images)[index]
            self.masks = np.array(self.masks)[index]


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create datamodule
    datamodule = create_datamodule(project_parameters=project_parameters)

    # prepare data
    datamodule.prepare_data()

    # set up data
    datamodule.setup()

    # get train, validation, test dataset
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    test_dataset = datamodule.test_dataset

    # get the first sample and target in the train dataset
    x, y = train_dataset[0]

    # display the dimension of sample and target
    print('the dimension of sample: {}'.format(x.shape))
    print('the dimension of target: {}'.format(y.shape))
