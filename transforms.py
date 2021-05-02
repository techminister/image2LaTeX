import torch
from torch import nn
from torchvision.transforms import Compose, ToTensor, Grayscale, ColorJitter, RandomAffine, Resize, RandomCrop, Normalize
from torchvision.transforms.functional import gaussian_blur
from random import randrange, random

class RandomGaussianNoise(nn.Module):
	def __init__(self, sigma=(0.0, 1.0)):
		super().__init__()
		self.sigma = sigma

	def forward(self, tensor):
		std = random() * (self.sigma[1] - self.sigma[0]) + self.sigma[0]
		return tensor + torch.randn(tensor.size()) * std/500
	def __repr__(self):
		return self.__class__.__name__ + f"(sigma={self.sigma})"

class RandomApply(nn.Module):
	def __init__(self, transform, p=0.5):
		super().__init__()
		self.transform = transform
		self.p = p

	def forward(self, tensor):
		return self.transform(tensor) if random() <= self.p else tensor

	def __repr__(self):
		return self.__class__.__name__ + f"(transform={self.transform}, p={self.p}"

class RandomGaussianBlur(nn.Module):
	def __init__(self, kernel_size, sigma=(0.1, 2.0)):
		super().__init__()
		self.kernel_size = kernel_size
		self.sigma = sigma

	def forward(self, tensor):
		kernel_size = int(randrange(self.kernel_size[0] - 1, self.kernel_size[1])) * 2 + 1
		return gaussian_blur(tensor, (kernel_size, kernel_size), [torch.empty(1).uniform_(self.sigma[0], self.sigma[1]).item()])

	def __repr__(self):
		return self.__class__.__name__ + f"(kernel_size={self.kernel_size}, sigma={self.sigma})"


clean_transform = Compose([
	Grayscale(),
	Resize((90, 90)),
	ToTensor(),
	Normalize((0.5), (0.5))
])

noise_transform = Compose([
	Grayscale(),
	Resize((90, 90)),
	RandomApply(
		RandomAffine(
			degrees=(-20, 20),
			fillcolor=255
		)
	),
	RandomApply(
		RandomAffine(
			degrees=0,
			scale=(0.85, 1.15),
			fillcolor=255
		)
	),
	RandomApply(
		RandomAffine(
			degrees=0,
			translate=(0.1, 0.1),
			fillcolor=255
		)
	),
	RandomApply(
		ColorJitter(
			brightness=0.1,
		)
	),
	ToTensor(),
	RandomApply(
		RandomGaussianNoise()
	),
	RandomApply(
		RandomGaussianBlur(
			kernel_size=(1, 5)
		)
	),
	Normalize((0.5), (0.5))
])