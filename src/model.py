#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
from os.path import isfile
from torchsummary import summary
import torch


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        input_height=project_parameters.input_height,
        in_chans=project_parameters.in_chans,
        hidden_chans=project_parameters.hidden_chans,
        chans_scale=project_parameters.chans_scale,
        depth=project_parameters.depth,
        loss_function_name=project_parameters.loss_function_name)
    if project_parameters.checkpoint_path is not None:
        if isfile(project_parameters.checkpoint_path):
            model = load_from_checkpoint(
                device=project_parameters.device,
                checkpoint_path=project_parameters.checkpoint_path,
                model=model)
        else:
            assert False, 'please check the checkpoint_path argument.\nthe checkpoint_path value is {}.'.format(
                project_parameters.checkpoint_path)
    return model


#class
class UNet(nn.Module):
    def __init__(self, input_height, in_chans, hidden_chans, chans_scale,
                 depth) -> None:
        super().__init__()
        assert input_height >= 2**depth, 'input_height must be greater or equal to 2**depth'
        self.kernel_size = 4
        self.stride = 2
        self.padding = 1
        self.depth = depth
        self.last_layer = nn.Conv2d(in_channels=in_chans,
                                    out_channels=1,
                                    kernel_size=1) if in_chans > 1 else None
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for index in range(self.depth):
            encode = []
            encode += [
                nn.Conv2d(in_channels=in_chans,
                          out_channels=hidden_chans,
                          kernel_size=self.kernel_size,
                          stride=self.stride,
                          padding=self.padding),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_chans,
                          out_channels=hidden_chans * chans_scale,
                          kernel_size=1),
                nn.GLU(1)
            ]
            self.encoder.append(nn.Sequential(*encode))
            decode = []
            decode += [
                nn.Conv2d(in_channels=hidden_chans,
                          out_channels=chans_scale * hidden_chans,
                          kernel_size=1),
                nn.GLU(1),
                nn.ConvTranspose2d(in_channels=hidden_chans,
                                   out_channels=in_chans,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   padding=self.padding)
            ]
            if index > 0:
                decode.append(nn.ReLU())
            self.decoder.insert(0, nn.Sequential(*decode))
            in_chans = hidden_chans
            hidden_chans = int(chans_scale * hidden_chans)

    def forward(self, x):
        skips = []
        for encode in self.encoder:
            x = encode(x)
            skips.append(x)
        for decode in self.decoder:
            skip = skips.pop(-1)
            x = x + skip
            x = decode(x)
        if self.last_layer is not None:
            x = self.last_layer(x)
        return x


class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config,
                 input_height, in_chans, hidden_chans, chans_scale, depth,
                 loss_function_name) -> None:
        super().__init__(optimizers_config, lr, lr_schedulers_config)
        self.backbone_model = UNet(input_height=input_height,
                                   in_chans=in_chans,
                                   hidden_chans=hidden_chans,
                                   chans_scale=chans_scale,
                                   depth=depth)
        self.loss_function = self.create_loss_function(
            loss_function_name=loss_function_name)
        self.activation_function = nn.Sigmoid()
        self.stage_index = 0

    def forward(self, x):
        y = self.backbone_model(x)
        return self.activation_function(y)

    def shared_step(self, batch):
        x, y = batch
        y_hat = self.forward(x=x)
        loss = self.loss_function(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch=batch)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch=batch)
        self.log('val_loss', loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch=batch)
        self.log('test_loss', loss, prog_bar=True)


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans,
                        project_parameters.input_height,
                        project_parameters.input_height),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   project_parameters.input_height,
                   project_parameters.input_height)

    # get model output
    y_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(y_hat.shape))
