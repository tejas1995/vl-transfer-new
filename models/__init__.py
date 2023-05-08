from .bert import Bert
from .clip import Clip
from .t5 import T5

MODEL_MAP = {
    'bert': Bert,
    'clip': Clip,
    'bert-lxmert': Bert,
    't5': T5
}