from transforms import clean_transform, noise_transform
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms.functional import to_pil_image

def load_data(loc="data", replications=1, noisy=True, shuffle=True, batch_size=32):
	return DataLoader(
		ConcatDataset([ImageFolder(
			loc,
			transform=noise_transform if noisy else clean_transform
		) for _ in range(replications)]),
		shuffle=shuffle,
		batch_size=batch_size
	)
