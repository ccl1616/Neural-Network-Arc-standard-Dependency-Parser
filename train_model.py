import sys
import numpy as np
import torch

from torch.nn import Module, Linear, Embedding, NLLLoss
from torch.nn.functional import relu, log_softmax
from torch.utils.data import Dataset, DataLoader 

from extract_training_data import FeatureExtractor

class DependencyDataset(Dataset):
	
	'''
	Function: __init__
	Input: use two .npy files created in part 2
	'''
	def __init__(self, input_filename, output_filename):
		self.inputs = np.load(input_filename)
		self.outputs = np.load(output_filename)

	def __len__(self): 
		return self.inputs.shape[0]

	def __getitem__(self, k): 
		return (self.inputs[k], self.outputs[k])


class DependencyModel(Module): 

	def __init__(self, word_types, outputs):
		super(DependencyModel, self).__init__()
		# TODO: complete for part 3
		# Define layers
		# Embedding layer. Set num_embeddings = word_types, embedding_dim=128.
		self.embedding = Embedding(num_embeddings=word_types, embedding_dim=128)
		# Torch linear hidden layer of 128 units.
		self.hidden = Linear(in_features=768, out_features=128)
		# Torch linear layer of 91 units.
		self.output_layer = Linear(in_features=128, out_features=outputs)

	def forward(self, inputs):
		# TODO: complete for part 3
		# Embedding layer. Input (batch_size, 6), Output (batch_size, 6, 128).
		embedded = self.embedding(inputs)
		# Flatten embedded from (batch_size, 6, 128) to (batch_size, 768)
		# embedded.size(0): the size of first dimension of the tensor, which is the batch_size = 16.
		embedded_flat = embedded.view(embedded.size(0), -1)
		# Pass to hidden layer with reLu. Input (batch_size, 768), Output (batch_size, 128)
		hidden_output = relu(self.hidden(embedded_flat))
		# Pass to output layer. Output (batch_size, 91).
		result = self.output_layer(hidden_output)
		return result
		# return torch.zeros(inputs.shape(0), 91)  # replace this line


def train(model, loader): 

	# loss_function = NLLLoss(reduction='mean')
	loss_function = torch.nn.CrossEntropyLoss(reduction='mean')

	LEARNING_RATE = 0.01 
	optimizer = torch.optim.Adagrad(params=model.parameters(), lr=LEARNING_RATE)

	tr_loss = 0 
	tr_steps = 0

	# put model in training mode
	model.train()


	correct = 0 
	total =  0 
	for idx, batch in enumerate(loader):
		# begin for loop
		inputs, targets = batch

		predictions = model(torch.LongTensor(inputs))

		loss = loss_function(predictions, targets)
		# loss = loss_function(predictions, torch.LongTensor(targets))
		tr_loss += loss.item()

		#print("Batch loss: ", loss.item()) # Helpful for debugging, maybe 

		tr_steps += 1

		if idx % 1000==0:
			curr_avg_loss = tr_loss / tr_steps
			print(f"Current average loss: {curr_avg_loss}")

		# To compute training accuracy for this epoch 
		correct += sum(torch.argmax(predictions, dim=1) == torch.argmax(targets, dim=1))
		# predicted_classes = torch.argmax(predictions, dim=1)  # Get predicted class for each input
		# correct += (predicted_classes == torch.LongTensor(targets)).sum().item()  # Compare with true classes
        
		total += len(inputs)
			
		# Run the backward pass to update parameters 
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		# end for loop

	epoch_loss = tr_loss / tr_steps
	acc = correct / total
	print(f"Training loss epoch: {epoch_loss},   Accuracy: {acc}")


if __name__ == "__main__":

	WORD_VOCAB_FILE = 'data/words.vocab'
	POS_VOCAB_FILE = 'data/pos.vocab'

	try:
		word_vocab_f = open(WORD_VOCAB_FILE,'r')
		pos_vocab_f = open(POS_VOCAB_FILE,'r')
	except FileNotFoundError:
		print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
		sys.exit(1)

	extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)


	model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))

	dataset = DependencyDataset(sys.argv[1], sys.argv[2])
	loader = DataLoader(dataset, batch_size = 16, shuffle = True)

	print("Done loading data")

    # Now train the model
	for i in range(10): 
		print(i)
		train(model, loader)

	torch.save(model.state_dict(), sys.argv[3])
