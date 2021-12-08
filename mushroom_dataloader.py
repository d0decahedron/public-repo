import torch
from torch.utils.data import DataLoader, Dataset

#########################################
# Vocabulary-making, data preprocessing #
#########################################


def make_inputs_output(data):
    '''
    A function that splits the data into inputs and ouputs, since the first element in a row is the output.

    :param data: list(list(str))
    :return input_data: list(list(str))
    :return output_data: list(list(str))

    '''
    input_data = [el[1:] for el in data]
    output_data = [el[0] for el in data]
    return input_data, output_data


def alt_make_vocab(data):

    '''
    A function that returns a vocabulary lookup dictionary for each unique feature (position). 
    For the mushroom dataset - 22 unique attributes for inputs, 2 for outputs.

    :param data: list(list(str))
    :return possible_features_per_attribute: dict(#:dict()); key - range(len(unique)), value - dict(): key: feature, value: index

    '''

    unique_attributes = len(data[0]) # number of possible attributes
    possible_features_per_attribute = {k:None for k in range(unique_attributes)}

    for i in range(unique_attributes):
        uniq = set([el[i] for el in data])
        possible_features_per_attribute[i] = {k:v for v,k in enumerate(uniq)}

    return possible_features_per_attribute


def numerize_data(input_data):

    '''
    A function that enumerates (input) data using the alt_make_vocab lookup dictionary.

    :param input_data: list(list(str))
    :returns new_inputs: list(Tensor)

    '''

    possible_features_per_attribute = alt_make_vocab(input_data)
    new_inputs = []
    # new_outputs = []
    for instance in input_data:
        vec = []
        for i,el in enumerate(instance):
            needed_attribute = possible_features_per_attribute[i]
            vec += [needed_attribute[el]]
        new_inputs.append(torch.Tensor(vec))
        
    return new_inputs

def one_hot(output_data):

    '''
    A function that takes a list of the outputs, and returns them as one-hot encoded vectors.

    Possible outputs: e (0), p (1)


    :param output_data: list(str)
    :return new_outputs: list(Tensor()) 

    '''

    outputs = alt_make_vocab(output_data)[0]
    new_outputs = []
    for out in output_data:
        one_hot_vec = [0]*len(outputs)
        one_hot_vec[outputs[out]] = 1
        new_outputs.append(torch.Tensor(one_hot_vec))
    return new_outputs


def enumerated_data(inp, out):

    '''
    A function that takes the enumerated inputs / one-hot encoded outputs and zips them together, so that they are accessible in the Dataset class.

    :param inp: enumerated inputs
    :param out: one-hot encoded outputs
    :returns data: dict(), data['in'] = list(input_Tensor); data['out'] = list(output_Tensor)


    '''

    data = {'in':[], 'out':[]}
    for instance, outcome in zip(inp,out):
        data['in'].append(instance)
        data['out'].append(outcome)
    return data


#################
# Dataset class #
#################

class Dataset(Dataset):
    def __init__(self, data):
        self.input_data = data['in']
        self.output_data = data['out']
        
    def __len__(self):
        return len(self.input_data)
        
    def __getitem__(self, index):
            
        select_input = self.input_data[index]
        select_output = self.output_data[index]
        
        return select_input, select_output