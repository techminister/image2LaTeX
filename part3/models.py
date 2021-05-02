import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Convolutional(nn.Module):
    def __init__(self, output_size):
        super(Convolutional, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, output_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)

    def step(self, inp, hid):
        out, hid = self.gru(inp, hid)
        return out, hid

    def forward(self, inp):
        inp_length, batch_size, _ = inp.shape

        outs = torch.zeros(inp_length, batch_size, self.hidden_size, device=device)
        hid = self.init_hidden(batch_size)

        for ei in range(inp_length):
            out, hid = self.step(inp[ei].unsqueeze(0), hid)
            outs[ei] = out[0]

        return outs

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class PredictiveAlignmentAttention(nn.Module):
    def __init__(self, hidden_size, sx=4, sy=4):
        super(PredictiveAlignmentAttention, self).__init__()
        self.hidden_size = hidden_size
        self.sx = sx
        self.sy = sy

        self.alignment = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Tanh(),
            nn.Linear(self.hidden_size * 2, 2),
            nn.Sigmoid(),
        )
        self.contextualizer = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size), nn.ReLU()
        )

    def forward(self, inp, hid, V):
        h, w, batch_size, _ = V.shape

        algn = self.alignment(hid)
        px, py = algn.permute(1, 0)
        x0 = (px * w).unsqueeze(1).unsqueeze(1)
        y0 = (py * h).unsqueeze(1).unsqueeze(1)

        def gaussian(x, y):
            return torch.exp(
                -(
                    (x - x0) ** 2 / (2 * self.sx ** 2)
                    + (y - y0) ** 2 / (2 * self.sy ** 2)
                )
            )

        xs = torch.cat(h * [torch.arange(w, device=device).unsqueeze(0)])
        xs = torch.cat(batch_size * [xs.unsqueeze(0)])
        ys = torch.cat(w * [torch.arange(h, device=device).unsqueeze(1)], axis=1)
        ys = torch.cat(batch_size * [ys.unsqueeze(0)])
        alpha = gaussian(xs, ys)

        alpha = alpha / alpha.sum(axis=[1, 2]).unsqueeze(1).unsqueeze(1)
        attn = torch.einsum("brc,rcbh->bh", alpha, V)

        ctx = self.contextualizer(torch.cat([inp, attn], axis=1))

        return ctx


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attention = PredictiveAlignmentAttention(self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.output_size), nn.LogSoftmax(dim=1)
        )

    def step(self, inp, hid, V):
        inp = self.embedding(inp)
        inp = self.dropout(inp)

        ctx = self.attention(inp, hid, V)

        out, hid = self.gru(ctx.unsqueeze(0), hid.unsqueeze(0))
        out = self.classifier(out[0])

        return out, hid[0]

    def forward(self, V, inp=1, eos=0, force_inp=None, length=None):
        batch_size = V.shape[2]
        inp = torch.tensor([inp] * batch_size, device=device)
        hid = V.mean(axis=[0, 1])
        outs = []
        if force_inp != None:
            for inp in torch.cat([inp.unsqueeze(0), force_inp[:, :-1].permute(1, 0)]):
                out, hid = self.step(inp, hid, V)
                outs.append(out)
        elif length != None:
            ind = 0
            while (not length) or (ind < length):
                out, hid = self.step(inp, hid, V)
                topv, topi = out.topk(1)
                outs.append(out)
                inp = topi.squeeze(1)
                ind += 1
        else:
            raise ValueError("Either force_inp or length must be specified.")

        return torch.stack(outs).permute(1, 2, 0)


class Transcriptor(nn.Module):
    def __init__(self, hidden_size, output_size, fmap_size=64):
        super(Transcriptor, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.fmap_size = fmap_size

        self.convolutional = Convolutional(self.fmap_size)
        self.encoder = EncoderRNN(self.fmap_size, self.hidden_size)
        self.decoder = AttnDecoderRNN(self.hidden_size, self.output_size)

    def forward(self, imgs, force_inp=None, length=None):
        fmap = self.convolutional(imgs).permute(2, 3, 0, 1)
        V = torch.stack([self.encoder(row) for row in fmap])
        outs = self.decoder(V, force_inp=force_inp, length=length)
        return outs
