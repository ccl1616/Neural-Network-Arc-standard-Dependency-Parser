import sys
import copy

import numpy as np
import torch

from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        # TODO: Write the body of this loop for part 5
        while state.buffer:
            # features = self.extractor.get_input_representation(words, pos, state)
            # predictions = self.model(torch.tensor([features], dtype=torch.float32))

            # Use the get_input_representation function to get the input features from words, pos, and state
            features = self.extractor.get_input_representation(words, pos, state)
            # Convert features to a tensor (expected by the model)
            features_tensor = torch.LongTensor(features).unsqueeze(0)  # Add batch dimension
            # Get predictions from the model
            predictions = self.model(features_tensor)

            # Softmax to get action probabilities
            probabilities = torch.nn.functional.softmax(predictions, dim=1).detach().numpy()[0]
            sorted_actions = np.argsort(probabilities)[::-1].tolist()  # Sort by Probabilities. Convert to list of indices

            # Print sorted actions and their corresponding probabilities
            # print(f"Current state: stack={state.stack}, buffer={state.buffer}, deps={state.deps}")
            # print(f"Probabilities: {probabilities.tolist()}")
            # print(f"Sorted actions (indices): {sorted_actions}")
            # print(f"Sorted actions (labels): {[self.output_labels[action_idx] for action_idx in sorted_actions]}")
            # print(f"Corresponding probabilities (sorted): {[probabilities[action_idx] for action_idx in sorted_actions]}")

            # for each action
                # if stack is emoty: cannot arc-left or right-left
                # if buffer size == 1 and not stack.empty(), cannot shift
                # root cannot be target of a left-arc
            valid_actions = []
            # with valid_actions
            for action_idx in sorted_actions:
                action = self.output_labels[action_idx]
                # print(f"Predicted action: {action}")
                if action[0] == "shift" and len(state.buffer) > 0:
                    if len(state.buffer) == 1 and len(state.stack) > 0:
                        continue
                    valid_actions.append(action)
                elif action[0] == "left_arc" and len(state.stack) > 0 and state.stack[-1] != 0:
                    valid_actions.append(action)
                elif action[0] == "right_arc" and len(state.stack) > 0:
                    valid_actions.append(action)
                # break if one of the valid action is found
                if len(valid_actions):
                    break

            # If there are valid actions, choose the best one (the first in sorted_actions)
            if valid_actions:
                best_action = valid_actions[0]
                if best_action[0] == "shift":
                    state.shift()
                elif best_action[0] == "left_arc":
                    state.left_arc(best_action[1])
                elif best_action[0] == "right_arc":
                    state.right_arc(best_action[1])
            else:
                print(f"No valid action could be performed. Current state: stack={state.stack}, buffer={state.buffer}, deps={state.deps}")
                break

            # end while

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


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
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
