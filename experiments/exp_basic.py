import os
import torch
from model import Transformer, Informer, Reformer, Flowformer, Flashformer, \
    iTransformer, iInformer, iReformer, iFlowformer, iFlashformer, EMAformer
from model.EMAformerDynamic_fixed import Model as EMAformerDynamic
from model.EMAformerDynamic_fixed import EMAformerDynamicZeroShot
from model.EMAformerDynamic_fixed import EMAformerDynamicTransfer


class Exp_Basic_Fixed(object):
    """
    实验基类（修复版）

    与原版 exp_basic.py 的区别：
    - EMAformerDynamic / EMAformerDynamicZeroShot / EMAformerDynamicTransfer
      指向修复版 model.EMAformerDynamic_fixed 模块
    - 其余模型保持不变
    """
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Transformer': Transformer,
            'Informer': Informer,
            'Reformer': Reformer,
            'Flowformer': Flowformer,
            'Flashformer': Flashformer,
            'iTransformer': iTransformer,
            'iInformer': iInformer,
            'iReformer': iReformer,
            'iFlowformer': iFlowformer,
            'iFlashformer': iFlashformer,
            'EMAformer': EMAformer,
            # 动态嵌入护甲系列（指向修复版）
            'EMAformerDynamic': EMAformerDynamic,
            'EMAformerDynamicZeroShot': EMAformerDynamicZeroShot,
            'EMAformerDynamicTransfer': EMAformerDynamicTransfer,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass