#import
from src.project_parameters import ProjectParameters
from DeepLearningTemplate.model import BaseModel, load_from_checkpoint
import torch.nn as nn
from os.path import isfile
from torchsummary import summary
import torch
import math
import torch.nn.functional as F
import timm
from torchmetrics.functional import dice_score
from torchmetrics import ConfusionMatrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#def
def create_model(project_parameters):
    model = UnsupervisedModel(
        optimizers_config=project_parameters.optimizers_config,
        lr=project_parameters.lr,
        lr_schedulers_config=project_parameters.lr_schedulers_config,
        model_name=project_parameters.model_name,
        in_chans=project_parameters.in_chans,
        loss_function_name=project_parameters.loss_function_name,
        classes=project_parameters.classes)
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
class ASPP_module(nn.Module):
    def __init__(self, x_inplanes, x_planes, rate):
        super(ASPP_module, self).__init__()
        if rate == 1:
            kernel_size = 1
            padding = 0
        else:
            kernel_size = 3
            padding = rate
        self.atrous_convolution = nn.Conv2d(x_inplanes,
                                            x_planes,
                                            kernel_size=kernel_size,
                                            stride=1,
                                            padding=padding,
                                            dilation=rate,
                                            bias=False)
        self.bn = nn.BatchNorm2d(x_planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_convolution(x)
        x = self.bn(x)

        return self.relu(x)


class DeepLabV3Plus(nn.Module):
    def __init__(self, model_name, in_chans, num_classes) -> None:
        super().__init__()
        self.backbone_model = timm.create_model(model_name=model_name,
                                                in_chans=in_chans,
                                                features_only=True,
                                                pretrained=True)
        output_channels = self.backbone_model.feature_info.channels()
        x_inplanes = output_channels[-1]
        x_planes = x_inplanes // 8
        low_level_features_inplanes = output_channels[1]
        low_level_features_planes = low_level_features_inplanes // 2
        rates = [1, 6, 12, 18]
        self.aspp_blocks = nn.ModuleList([
            ASPP_module(x_inplanes=x_inplanes, x_planes=x_planes, rate=rate)
            for rate in rates
        ])
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(x_inplanes, x_planes, 1, stride=1, bias=False),
            nn.BatchNorm2d(x_planes), nn.ReLU())
        self.decoder_block1 = nn.Sequential(
            nn.Conv2d(in_channels=x_planes * 5,
                      out_channels=x_planes,
                      kernel_size=1,
                      bias=False), nn.BatchNorm2d(num_features=x_planes),
            nn.ReLU())
        self.decoder_block2 = nn.Sequential(
            nn.Conv2d(in_channels=low_level_features_inplanes,
                      out_channels=low_level_features_planes,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(num_features=low_level_features_planes), nn.ReLU())

        self.last_conv = nn.Sequential(
            nn.Conv2d(in_channels=(x_planes + low_level_features_planes),
                      out_channels=x_planes,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(num_features=x_planes),
            nn.ReLU(),
            nn.Conv2d(in_channels=x_planes,
                      out_channels=x_planes,
                      kernel_size=3,
                      padding=1,
                      bias=False), nn.BatchNorm2d(num_features=x_planes),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=x_planes,
                out_channels=num_classes,
                kernel_size=1,
            ))

    def forward(self, x):
        b, c, w, h = x.shape

        #backbone
        outputs = self.backbone_model(x)
        low_level_features, x = outputs[1], outputs[-1]

        #aspp
        xs = []
        for aspp in self.aspp_blocks:
            xs.append(aspp(x))
        xs.append(
            F.upsample(input=self.global_avg_pool(x),
                       size=xs[-1].size()[2:],
                       mode='bilinear',
                       align_corners=True))
        x = torch.cat(tensors=xs, dim=1)

        #decoder block 1
        x = self.decoder_block1(x)
        x = F.upsample(x,
                       size=(int(math.ceil(w / 4)), int(math.ceil(h / 4))),
                       mode='bilinear',
                       align_corners=True)

        #decoder block 2
        low_level_features = self.decoder_block2(low_level_features)

        x = torch.cat((x, low_level_features), dim=1)
        x = self.last_conv(x)
        x = F.upsample(input=x,
                       size=(w, h),
                       mode='bilinear',
                       align_corners=True)

        return x


class UnsupervisedModel(BaseModel):
    def __init__(self, optimizers_config, lr, lr_schedulers_config, model_name,
                 in_chans, loss_function_name, classes) -> None:
        super().__init__(optimizers_config, lr, lr_schedulers_config)
        self.backbone_model = DeepLabV3Plus(model_name=model_name,
                                            in_chans=in_chans,
                                            num_classes=len(classes))
        self.loss_function = self.create_loss_function(
            loss_function_name=loss_function_name)
        self.activation_function = nn.Sigmoid()
        self.dice_score = dice_score
        self.confusion_matrix_function = ConfusionMatrix(
            num_classes=len(classes))
        self.classes = classes
        self.stage_index = 0

    def forward(self, x):
        y = self.activation_function(self.backbone_model(x))
        return y.argmax(1)[:, None]

    def shared_step(self, batch, test=False):
        x, y = batch
        y_hat = self.backbone_model(x)
        loss = self.loss_function(y_hat, y)
        if y.dtype == torch.float32 and len(
                y.shape) > 3:  #the y dimension is (b, num_classes, w, h)
            y = y.argmax(1)  #the y dimension is (b, w, h)
        elif y.dtype == torch.int64 and len(y.shape) > 3:
            y = y[:, 0]  #the y dimension is (b, w, h)
        score = self.dice_score(preds=self.activation_function(y_hat),
                                target=y)
        if test:
            return loss, score, self.activation_function(y_hat), y
        else:
            return loss, score

    def training_step(self, batch, batch_idx):
        loss, score = self.shared_step(batch=batch, test=False)
        self.log('train_loss',
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        self.log('train_dice_score',
                 score,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True,
                 logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, score = self.shared_step(batch=batch, test=False)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_dice_score', score, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, score, y_hat, y = self.shared_step(batch=batch, test=True)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_dice_score', score, prog_bar=True)
        confusion_matrix_step = self.confusion_matrix_function(
            y_hat.argmax(1), y).cpu().data.numpy()
        loss_step = loss.item()
        score_step = score.item()
        return {
            'confusion_matrix': confusion_matrix_step,
            'loss': loss_step,
            'score': score_step
        }

    def test_epoch_end(self, test_outs):
        stages = ['train', 'val', 'test']
        print('\ntest the {} dataset'.format(stages[self.stage_index]))
        print('the {} dataset confusion matrix:'.format(
            stages[self.stage_index]))
        confusion_matrix = np.sum([v['confusion_matrix'] for v in test_outs],
                                  0)
        loss = np.mean([v['loss'] for v in test_outs])
        score = np.mean([v['score'] for v in test_outs])
        # use pd.DataFrame to wrap the confusion matrix to display it to the CLI
        confusion_matrix = pd.DataFrame(data=confusion_matrix,
                                        columns=self.classes,
                                        index=self.classes).astype(int)
        print(confusion_matrix)
        plt.figure(figsize=[11.2, 6.3])
        plt.title('{}\nloss: {}\ndice score: {}'.format(
            stages[self.stage_index], loss, score))
        figure = sns.heatmap(data=confusion_matrix,
                             cmap='Spectral',
                             annot=True,
                             fmt='g').get_figure()
        plt.yticks(rotation=0)
        plt.ylabel(ylabel='Actual class')
        plt.xlabel(xlabel='Predicted class')
        plt.close()
        self.logger.experiment.add_figure(
            '{} confusion matrix'.format(stages[self.stage_index]), figure,
            self.current_epoch)
        self.stage_index += 1


if __name__ == '__main__':
    # project parameters
    project_parameters = ProjectParameters().parse()

    # create model
    model = create_model(project_parameters=project_parameters)

    # display model information
    summary(model=model,
            input_size=(project_parameters.in_chans, 224, 224),
            device='cpu')

    # create input data
    x = torch.ones(project_parameters.batch_size, project_parameters.in_chans,
                   224, 224)

    # get model output
    y_hat = model(x)

    # display the dimension of input and output
    print('the dimension of input: {}'.format(x.shape))
    print('the dimension of output: {}'.format(y_hat.shape))
