import torch
from torch import nn, optim
from transforms import clean_transform
from utils_custom import load_data
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Train custom classifier model")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--save_model", help="save model", default = 'custom_drop.pt')
parser.add_argument("--load_model", help="load model", default=None)

args = parser.parse_args()

L_RATE= args.learning_rate
N_EPOCH = args.epochs 
device = args.gpu
save_model = args.save_model 
load = args.load_model 

labels = pd.read_csv("meta.csv").to_dict()["formula"]

class SingleCharacterClassifier(nn.Module):
	def __init__(self):
		super(SingleCharacterClassifier, self).__init__()
		# Architecture: 3x Conv layers => Max Pooling => 3x FC layers
		self.layers = nn.Sequential(
			nn.Conv2d(1, 5, 5),
			nn.BatchNorm2d(5),
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Conv2d(5, 10, 5),
			nn.BatchNorm2d(10),
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Conv2d(10, 15, 5),
			nn.BatchNorm2d(15),
			nn.ReLU(), 
			nn.Dropout(0.25),
			nn.Conv2d(15, 20, 5),
			nn.BatchNorm2d(20),
			nn.ReLU(),
			nn.MaxPool2d(5),
			nn.Flatten(),
			nn.LeakyReLU(),
			nn.Linear(3920, 2048),
			nn.BatchNorm1d(2048),
			nn.LeakyReLU(),
			nn.Dropout(0.25),
			nn.Linear(2048, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(),
			nn.Dropout(0.25),
			nn.Linear(1024, 405)
		)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
			#print(x.size())
		return x

	def predict(self, loc, is_image=False):
		self.eval()
		device = "cuda" if next(self.parameters()).is_cuda else "cpu"
		img = Image.open(loc) if is_image else loc
		img_t = clean_transform(img).to(device)
		out = self.forward(torch.unsqueeze(img_t, 0))
		return nn.functional.softmax(out, dim=1)[0]

	def classify(self, loc, top=5):
		prediction = self.predict(loc)
		_, indices = torch.sort(prediction, descending=True)
		return [(labels[ind.item()], prediction[ind.item()].item()) for ind in indices[:top]]

def graph_plot(train_loss_values, valid_loss_values):
	fig, ax = plt.subplots()
	ax.plot(train_loss_values)
	ax.plot(valid_loss_values)
	ax.set_xlabel('Epochs')
	ax.set_ylabel('Loss')
	ax.set_title('Loss Plot')
	ax.legend(['Training Loss', 'Validation Loss'])
	fig.savefig('custom_drop.png')

def train(model, device, train_loader, optimizer, epoch):
	model.train()
	loss_fn = nn.CrossEntropyLoss()
	running_loss = 0
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		pred = model(data)
		loss = loss_fn(pred, target)
		running_loss += loss.item()
		loss.backward()
		optimizer.step()

		if batch_idx % 100 == 0:
			print("Epoch:", epoch, "loss:", round(loss.item(), 5))
	return running_loss/len(train_loader)

def test(model, device, test_loader, plot=False):
	model.eval()

	loss_fn = nn.CrossEntropyLoss()
	running_loss = 0
	correct = 0
	exampleSet = False
	example_data = np.zeros([10, 90, 90])
	example_pred = np.zeros(10)
	example_target = np.zeros(10)

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			pred_loss = model(data)
			loss = loss_fn(pred_loss, target)
			running_loss += loss.item()

			pred = torch.argmax(model(data), dim=1)
			correct += sum((pred - target) == 0).item()

			if not exampleSet:
				for i in range(10):
					example_data[i] = data[i][0].to("cpu").numpy()
					example_pred[i] = pred[i].to("cpu").numpy()
					example_target[i] = target[i].to("cpu").numpy()
					exampleSet = True

	print('Valid set loss ', round(running_loss/len(test_loader), 5))
	print('Valid set accuracy: ', round(100. * correct / len(test_loader.dataset), 3), '%')

	if not plot: return round(running_loss/len(test_loader), 5)
	for i in range(10):
		plt.subplot(2,5,i+1)
		data = (example_data[i] - example_data[i].min()) * (example_data[i].max() - example_data[i].min())
		plt.imshow(data, cmap='gray', interpolation='none')
		corr = int(example_pred[i]) == int(example_target[i])
		plt.title(labels[int(example_pred[i])] + (" ✔" if corr else " ✖"))
		plt.xticks([])
		plt.yticks([])
	plt.show()

def load_model(load, device):
	model = SingleCharacterClassifier().to(device)
	if load: model.load_state_dict(torch.load(load, map_location=device))
	return model

def run(N_EPOCH, L_RATE, load=None, save=None, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
	
	train_loss = []
	valid_loss = []
	train_loader = load_data(replications=100)
	test_loader = load_data(replications=1)

	model = load_model(load, device)
	optimizer = optim.Adam(model.parameters(), lr=L_RATE)

	for epoch in range(N_EPOCH):
		v_loss = test(model, device, test_loader)
		valid_loss.append(v_loss)
		if save: torch.save(model.state_dict(), save)
		t_loss = train(model, device, train_loader, optimizer, epoch + 1)
		train_loss.append(t_loss)	

	test(model, device, test_loader)
	graph_plot(train_loss,valid_loss)

	if save: torch.save(model.state_dict(), save)
if __name__ == "__main__":
	run(N_EPOCH, L_RATE, load = load, save = save_model, device = device)