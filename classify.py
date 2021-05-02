import sys, torch, json
from model import load_model

locs = sys.argv[1:]
if not len(locs): raise ValueError("At least one input image must be specified.")

clf = load_model("scc.model", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
for loc in locs:
	pred = clf.classify(loc, top=1)
	print(f"{loc} - Prediction: {pred[0][0]}, Confidence: {round(pred[0][1]*100, 3)}%")