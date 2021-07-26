from src.model.crossnet import CrossNetModule
from src.model.dummy import DummyModule
from src.model.rcan import RCANModule
from src.model.rwdcn import RwDCNMixedUpModule, RwDCNAttentionModule
from src.model.rwdcn_lr import RwDCNLRModule
from src.model.srntt import SRNTTModule

config_path = '/home/usrs/gzr1997/CODE/RwDCN/cfg'

model_dict = {
    'Dummy': DummyModule,
    'SRNTT': SRNTTModule,
    'CrossNet': CrossNetModule,
    'RwDCNMixedUp': RwDCNMixedUpModule,
    'RwDCNLR': RwDCNLRModule,
    'RCAN': RCANModule,
    'RwDCNAttention': RwDCNAttentionModule
}
