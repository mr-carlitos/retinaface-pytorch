##### CARLOS CODE FILE ########
from torchsummary import summary
from models.retinaface import RetinaFace
from data import cfg_re50

model = RetinaFace(cfg=cfg_re50)
summary(
    model,
    (3,1640,1640),
    device="cpu"
)