import cv2, torch
from dataset import Vocabulary
from torchvision import transforms
import models

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
vocab = Vocabulary()
hidden_size = 256
model = models.Transcriptor(hidden_size, len(vocab)).to(models.device)
model.load_state_dict(torch.load("epoch60.pt"))
model.eval()


def transcribe(img_path, length=50):
    img = 255 - cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = transform(img).to(models.device).unsqueeze(0)
    outs = model(img, length=length)
    pred = vocab.decode(outs.argmax(dim=1)[0].cpu().numpy())
    return pred.replace(r" \eos", "")
