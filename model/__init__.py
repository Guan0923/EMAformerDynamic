# 模型模块导入
# 注意：每个模型文件内定义的是 class Model，这里重命名为具体模型名

from model.Transformer import Model as Transformer
from model.Informer import Model as Informer
from model.Reformer import Model as Reformer
from model.Flowformer import Model as Flowformer
from model.Flashformer import Model as Flashformer
from model.iTransformer import Model as iTransformer
from model.iInformer import Model as iInformer
from model.iReformer import Model as iReformer
from model.iFlowformer import Model as iFlowformer
from model.iFlashformer import Model as iFlashformer
from model.EMAformer import Model as EMAformer