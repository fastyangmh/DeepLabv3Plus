#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.data_preparation import MyVOCSegmentation, MyImageFolder, ImageLightningDataModule
from typing import Optional, Callable
import numpy as np
from typing import Tuple, Any
from PIL import Image


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
        samples = np.array(self.samples)
        targets = np.array(self.targets)
        self.images = samples[targets == self.class_to_idx['image'], 0]
        self.masks = samples[targets == self.class_to_idx['mask'], 0]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index])  #assume the image is RGB channels
        target = Image.open(self.masks[index])
        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)
        return img, target


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
