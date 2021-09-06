import torch
import torch.nn.functional as F
import torch.nn as nn
# from torch.nn.modules.loss import _WeightedLoss
__all__ = ['SegmentationLosses']

#sliver07 loss
class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"
        self.entropyloss = nn.BCEWithLogitsLoss()

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                # print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            # inputs = torch.where(inputs>0.5 ,torch.ones_like(inputs),torch.zeros_like(inputs))
            # inputs = inputs.requires_grad_()
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        # inputs = torch.where(inputs>0.5,torch.ones_like(inputs),torch.zeros_like(inputs))
        # inputs = inputs.requires_grad_()
        intersection = torch.sum(inputs * targets)
        return intersection
    # 1.13
    # def forward(self, inputs, target):
    #     dice = 0
    #     for i in range(target.size(0)):
    #         dice = dice + self.binary_dice(inputs[i, ...], target[i, ...], i)
    #
    #     final_dice = dice / target.size(0)
    #     # target_dim3 = torch.squeeze(target, dim=1)
    #     ce = self.entropyloss(inputs, target.float())
    #     return 0.5 * final_dice + 0.5 * ce
    #     # return final_dice
    def forward(self, inputs, input_vice, target):
        dice = 0
        # 1.20
        # for i in range(target.size(0)):
        #     dice = dice + self.binary_dice(inputs[i, ...], target[i, ...], i)
        #
        # final_dice = dice / target.size(0)
        # target_dim3 = torch.squeeze(target, dim=1)
        # ce = F.cross_entropy(input_vice, target_dim3.long())
        # 1.20
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / target.size(0)
        # return 0.5 * final_dice + 0.5 * ce
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices
#sliver07 loss

class SegmentationLosses(nn.CrossEntropyLoss):
    def __init__(self, name='dice_loss', se_loss=False,
                 aux_weight=None, weight=None, ignore_index=0):
        '''2D Cross Entropy Loss with Auxiliary Loss or Dice Loss

        :param name: (string) type of loss : ['dice_loss', 'cross_entropy', 'cross_entropy_with_dice']
        :param aux_weight: (float) weights of an auxiliary layer or the weight of dice loss
        :param weight: (torch.tensor) the weights of each class
        :param ignore_index: (torch.tensor) ignore i class.
        '''
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.name = name
        self.aux_weight = aux_weight
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = True
        self.reduce = True
        print('Using loss: {}'.format(name))

    def forward(self, input, target):
        if target.dtype == torch.float32:
            target = target.long()
        if self.name == 'dice_loss':
            return self._dice_loss3(input, target) #self._dice_loss(*inputs)
        elif self.name == 'cross_entropy':
            if self.aux_weight == 0 or self.aux_weight == None:
                return super(SegmentationLosses, self).forward(input, target)
            else:
                pred1, pred2= input
                target = target
                loss1 = super(SegmentationLosses, self).forward(pred1, target)
                loss2 = super(SegmentationLosses, self).forward(pred2, target)
                return loss1 + self.aux_weight * loss2
        elif self.name == 'cross_entropy_with_dice':
            # ce = F.binary_cross_entropy_with_logits(input, target)
            ce = F.cross_entropy(input, target)
            dice1 = self._dice_loss1(input, target)
            # return super(SegmentationLosses, self).forward(*inputs)\
            #        + self.aux_weight * self._dice_loss3(*inputs)
            return 0.5 * ce + 0.5 * dice1
        else:
            raise NotImplementedError

    def _dice_loss1(self, input, target):
        """
        input : (NxCxHxW Tensor) which is feature output as output = model(x)
        target :  (NxHxW LongTensor)
        :return: the average dice loss for each channel
        """
        smooth = 1.0

        probs = F.softmax(input, dim=1)
        encoded_target = probs.detach() * 0

        # one-hot encoding
        if self.ignore_index != -1:
            mask = target == self.ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if self.weight is None:
            weight = 1

        intersection = probs * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = probs + encoded_target

        if self.ignore_index != -1:
            denominator[mask] = 0

        denominator = denominator.sum(0).sum(1).sum(1)
        loss_per_channel = weight * (1 - ((numerator + smooth) / (denominator + smooth)))
        average = encoded_target.size()[0] if self.reduction == 'mean' else 1

        return loss_per_channel.mean().mean()

    def _dice_loss2(self, input, target, optimize_bg=False, smooth=1.0):
        """input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
        target is a 1-hot representation of the groundtruth, shoud have same size as the input
        weight (Variable, optional): a manual rescaling weight given to each
                class. If given, has to be a Variable of size "nclasses"""

        def dice_coefficient(input, target, smooth=1.0):

            assert smooth > 0, 'Smooth must be greater than 0.'
            probs = F.softmax(input, dim=1)

            encoded_target = probs.detach() * 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            encoded_target = encoded_target.float()

            num = probs * encoded_target  # b, c, h, w -- p*g
            num = torch.sum(num, dim=3)  # b, c, h
            num = torch.sum(num, dim=2)  # b, c

            den1 = probs * probs  # b, c, h, w -- p^2
            den1 = torch.sum(den1, dim=3)  # b, c, h
            den1 = torch.sum(den1, dim=2)  # b, c

            den2 = encoded_target * encoded_target  # b, c, h, w -- g^2
            den2 = torch.sum(den2, dim=3)  # b, c, h
            den2 = torch.sum(den2, dim=2)  # b, c

            dice = (2 * num + smooth) / (den1 + den2 + smooth)  # b, c

            return dice

        dice = dice_coefficient(input, target, smooth=smooth)

        if not optimize_bg:
            dice = dice[:, 1:]                 # we ignore bg dice val, and take the fg

        if not type(self.weight) is type(None):
            if not optimize_bg:
                weight = self.weight[1:]             # ignore bg weight
            weight = weight.size(0) * weight / weight.sum()  # normalize fg weights
            dice = dice * weight                # weighting

        dice_loss = 1 - dice.mean(1)     # loss is calculated using mean over dice vals (n,c) -> (n)

        if not self.reduce:
            return dice_loss

        if self.size_average:
            return dice_loss.mean()

        return dice_loss.sum()

    def _dice_loss3(self, input, target):
        """Calculating the dice loss
            Args:
                prediction = predicted image
                target = Targeted image
            Output:
                dice_loss"""
        input = torch.argmax(input, dim=1)
        smooth = 1.0

        i_flat = input.view(-1)
        t_flat = target.view(-1)

        intersection = (i_flat * t_flat).sum()

        return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))





