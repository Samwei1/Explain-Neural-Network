import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import sigmoid
import matplotlib.pyplot as plt
from torch.autograd import Variable
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from random import sample, randint

"""
Reference:
Training section, Evaluation section, Testing section are based on the code from lab2 COMP4660
GA algorithm is based on the code from lab7 COMP4660
"""

# -----------------------------------------------------------------------------
# Pre-processing code
# -----------------------------------------------------------------------------
np.random.seed(0)
torch.manual_seed(0)
# read data from the four files and add label.
# label: 0 means HighBp, 1 means normal, 2 means pneumonia, 3 means sars.
highbp = pd.read_csv("SM_HighBP.csv", header=None)
highbp['label'] = 0
normal = pd.read_csv("SM_Normal.csv", header=None)
normal['label'] = 1
pneumonia = pd.read_csv("SM_pneumonia.csv", header=None)
pneumonia['label'] = 2
sars = pd.read_csv("SM_SARS.csv", header=None)
sars['label'] = 3

# # normalise input data
# for data in [highbp, normal, pneumonia, sars]:
#     for column in data.columns[:-1]:
#         # the last column is target
#         data[column] = data.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# split data into training set and test set
msk = np.random.rand(len(highbp)) < 0.8
train_highbp = highbp[msk]
test_highbp = highbp[~msk]

train_normal = normal[msk]
test_normal = normal[~msk]

train_pneumonia = pneumonia[msk]
test_pneumonia = pneumonia[~msk]

train_sars = sars[msk]
test_sars = sars[~msk]

# combine these data into one file
train_data = [train_highbp, train_normal, train_pneumonia, train_sars]
test_data = [test_highbp, test_normal, test_pneumonia, test_sars]
train_data = pd.concat(train_data)
test_data = pd.concat(test_data)

# resample
train_data = train_data.sample(frac=1).reset_index(drop=True)
test_data = test_data.sample(frac=1).reset_index(drop=True)
train_data.reset_index(drop=True, inplace=True)
test_data.reset_index(drop=True, inplace=True)

# save as a csv file
train_data.to_csv("preprocessing_training_data.csv", index=False, header=False)
test_data.to_csv("preprocessing_test_data.csv", index=False, header=False)
# training features and target
train_data = pd.read_csv("preprocessing_training_data.csv", header=None)
test_data = pd.read_csv("preprocessing_test_data.csv", header=None)

n_features = train_data.shape[1] - 1
train_input = train_data.iloc[:, :n_features]
train_target = train_data.iloc[:, n_features]

test_input = test_data.iloc[:, :n_features]
test_target = test_data.iloc[:, n_features]

# input and output tensors
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_target.values).long()


# -----------------------------------------------------------------------------
# Define a neural network
# -----------------------------------------------------------------------------

# a simple three layer neural network
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = sigmoid(self.hidden(x))
        out = F.log_softmax(self.predict(x), dim=1)
        return out


# hyper-parameters setting
input_neurons = n_features
output_neurons = 4
num_epochs = 1000
learning_rate = 0.01
hidden_neurons = 2

net = Net(input_neurons, hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# -----------------------------------------------------------------------------
# Train this neural network
# -----------------------------------------------------------------------------

# store all losses for visualisation
all_losses = []

for epoch in range(num_epochs):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(X)
    # Compute loss
    loss = loss_func(Y_pred, Y)
    all_losses.append(loss.item())

    # print progress
    if epoch % 50 == 0:
        # convert three-column predicted Y values to one column for comparison
        _, predicted = torch.max(Y_pred, 1)

        # calculate and print accuracy
        total = predicted.size(0)
        correct = predicted.data.numpy() == Y.data.numpy()

        print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
              % (epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimizer.step()

# -----------------------------------------------------------------------------
# Evaluate this neural network
# -----------------------------------------------------------------------------

confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)

_, predicted = torch.max(Y_pred, 1)
# get confusion matrix for training
for i in range(train_data.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

# -----------------------------------------------------------------------------
# Test this neural network
# -----------------------------------------------------------------------------
# create Tensors
X_test = torch.Tensor(test_input.values).float()
Y_test = torch.Tensor(test_target.values).long()
# predications
Y_pred_test = net(X_test)
# get the label
_, predicted_test = torch.max(Y_pred_test, 1)
# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())

print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

# get confusion matrix for testing
confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)

# torch.save(net.state_dict(), './save_weights.pth')
# print(net.state_dict())

# -----------------------------------------------------------------------------
# Find characteristic ON and OFF patterns
# -----------------------------------------------------------------------------

# get predications from trained neural network
X = torch.Tensor(train_input.values).float()
Y_pred = net(X)
_, predicted = torch.max(Y_pred, 1)
predicted = predicted.numpy()

# group the input patterns by the predicted labels and calculate the mean of each group
train_input["label"] = predicted
record_on = {}
record_off = {}
for i in range(0, 4):
    on_inputs = train_input[train_input["label"] == i].iloc[:, :-1].values
    off_inputs = train_input[train_input["label"] != i].iloc[:, :-1].values
    record_on[i] = on_inputs.mean(axis=0)
    record_off[i] = off_inputs.mean(axis=0)
print("--- input ON characteristic patterns ----")
df = pd.DataFrame(columns=["HighBp", "normal", "pneumonia", "sars"])
df["HighBp"] = record_on[0]
df["normal"] = record_on[1]
df["pneumonia"] = record_on[2]
df["sars"] = record_on[3]
df.to_excel("characteristic_ON_patterns.xlsx")
print(df)

print("--- input OFF characteristic patterns ----")
df = pd.DataFrame(columns=["HighBp", "normal", "pneumonia", "sars"])
df["HighBp"] = record_off[0]
df["normal"] = record_off[1]
df["pneumonia"] = record_off[2]
df["sars"] = record_off[3]
df.to_excel("characteristic_OFF_patterns.xlsx")
print(df)

# -----------------------------------------------------------------------------
# Calculate causal index on characteristic patterns
# -----------------------------------------------------------------------------
causal_index_on = {}
for j in range(output_neurons):
    input_data = record_on[j]
    input_data = Variable(torch.tensor([input_data]).float(), requires_grad=True)
    label = torch.tensor([j]).long()
    y_p = net(input_data)
    d = torch.autograd.grad(y_p[0][j], input_data)[0]
    causal_index_on[j] = d[0].numpy()

causal_index_off = {}
for j in range(output_neurons):
    input_data = record_off[j]
    input_data = Variable(torch.tensor([input_data]).float(), requires_grad=True)
    label = torch.tensor([j]).long()
    y_p = net(input_data)
    d = torch.autograd.grad(y_p[0][j], input_data)[0]
    causal_index_off[j] = d[0].numpy()

df = pd.DataFrame(columns=["HighBp", "normal", "pneumonia", "sars"])
df["HighBp"] = causal_index_on[0]
df["normal"] = causal_index_on[1]
df["pneumonia"] = causal_index_on[2]
df["sars"] = causal_index_on[3]
df.to_excel("causal_index_on.xlsx")

df = pd.DataFrame(columns=["HighBp", "normal", "pneumonia", "sars"])
df["HighBp"] = causal_index_off[0]
df["normal"] = causal_index_off[1]
df["pneumonia"] = causal_index_off[2]
df["sars"] = causal_index_off[3]
df.to_excel("causal_index_off.xlsx")

causal_index = np.zeros((8, input_neurons))
causal_index[0] = causal_index_on[0]
causal_index[1] = causal_index_off[0]
causal_index[2] = causal_index_on[1]
causal_index[3] = causal_index_off[1]
causal_index[4] = causal_index_on[2]
causal_index[5] = causal_index_off[2]
causal_index[6] = causal_index_on[3]
causal_index[7] = causal_index_off[3]

print("--- causal index --- ")
print(causal_index)

# -----------------------------------------------------------------------------
# Extract Rules with causal index
# -----------------------------------------------------------------------------

# I picked some important features based on the causal index. Here is a table shows the features.
# HighBp : Bp Diastolic Med, Bp Systolic Med, Bp Diastolic High, Bp Systolic High
# Normal: Bp Diastolic Med, Bp Systolic Med, Temp12pm High <=0
# pneumonia:  Temp 12pm Mod, Temp 4pm Mod, Temp 8pm High
# Sars: Bp Diastolic Med, Bp Systolic Med, Abdominal Pain Yes

# find boundary
# index of each important feature
highbp_features = [13, 16, 14, 17]
normal_features = [13, 16, 5]
pneumonia_features = [5, 7, 11]
sars_features = [13, 16, 22]
data = []


# find boundary when gradient equals to 0
def find_boundary(index_column, index_label):
    a = np.arange(0, 1, 0.01)
    boundary = None
    difference = None
    for i in a:
        input_data = record_on[index_label]
        input_data[index_column] = i
        input_data = Variable(torch.tensor([input_data]).float(), requires_grad=True)
        y_p = net(input_data)
        d = torch.autograd.grad(y_p[0][index_label], input_data)[0]
        gradient_c = d[0][index_label].numpy()
        data.append(gradient_c)
        if difference is None or abs(gradient_c) < difference:
            if i > 0:
                difference = abs(gradient_c)
                boundary = i
    return boundary


print("----------------------------")
# you can change the index of feature and the output neuron, and see the boundary.
print(find_boundary(13, 0))
a = np.arange(0, 1, 0.01)
plt.plot(a, data)
plt.xlabel("Bp Systolic Med")
plt.ylabel(" Rate of Change ")
plt.savefig("rateofchange_vs_input.png", bbox_inches='tight')

# -----------------------------------------------------------------------------
# Given a input pattern, find the explanations
# -----------------------------------------------------------------------------

input_pattern = np.array(train_input.iloc[0, :-1])
min_dis = None
target = None
# find the most similar characteristic pattern
for i in range(len(causal_index)):
    pattern = causal_index[i]
    dis = np.linalg.norm(pattern - input_pattern)
    if min_dis is None or min_dis > dis:
        min_dis = dis
        target = i
print("most similar characteristic pattern", target)
# find the next most likely output
min_dis = None
for i in range(4):
    pattern = causal_index_on[i]
    dis = np.linalg.norm(pattern - input_pattern)
    if min_dis is None or min_dis > dis:
        min_dis = dis
        target = i
print("the next most likely output", target)

# -----------------------------------------------------------------------------
# Test rules extracted by causal index
# -----------------------------------------------------------------------------

# the rules I extracted are: ON High BP ((BP Diastolic Med > 0.53) Λ (BP Systolic Med >0.73)) OR ((BP Systolic High > 0.06) Λ (BP Diastolic High > 0.28))
#                            ON Normal  (Temp12pm High <=0) Λ (BP Systolic Med < 0.11)
#                                       Λ (BP Diastolic Med < 0.1)
#                            ON Pneumonia (Temp 12pm High >0) Λ (Temp 4pm Mod >0)
#                                       Λ (Temp 8pm High >0)
#                            ON SARS  (Bp Diastolic Med < 0.61) Λ (BP Systolic Med < 0.58)
#                                       Λ (Abdominal Pain Yes > 0.55)
input = np.array(train_input.iloc[:, :-1])
output = np.array(train_input.iloc[:, -1])
correct_counter = 0
c = 0
for i in input:
    o = None
    if i[5] <= 0 and i[13] < 0.11 and i[16] < 0.1:
        o = 1
    if i[5] > 0 and i[7] > 0 and i[11] > 0.5:
        o = 2
    if (i[13] < 0.58 and i[16] < 0.61) and i[22] > 0.55:
        o = 3
    if (i[13] > 0.73 and i[16] > 0.53) or (i[14] > 0.06 and i[17] > 0.28):
        o = 0
    if o == output[c]:
        correct_counter += 1
    c += 1
print("causal index accuracy: ", correct_counter / c)

# -----------------------------------------------------------------------------
# Genetic Algorithm: find optimal features
# -----------------------------------------------------------------------------

#  Define settings
DNA_SIZE = 23  # number of bits in DNA
POP_SIZE = 30  # population size
CROSS_RATE = 0.8  # DNA crossover probability
MUTATION_RATE = 0.2  # mutation probability
N_GENERATIONS = 50  # generation size
number_of_features = 2  # define number of features should be selected
num_epochs = 200  # epochs for training a network
learning_rate = 0.01
hidden_neurons = 2

Y_test = torch.Tensor(test_target.values).long()
# record the fitness value to improve training speed
record_accuracy = {}


# define target function
def F_function(x):
    accuracy = []
    for list_features in x:
        print("selected features during training GA: ",list_features)
        if tuple(list_features) in record_accuracy:
            accuracy.append(record_accuracy[tuple(list_features)])
        else:
            # train a neural network based on the selected features
            train_input = train_data.iloc[:, list_features]
            net = Net(len(list_features), hidden_neurons, output_neurons)
            loss_func = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            X = torch.Tensor(train_input.values).float()
            Y = torch.Tensor(train_target.values).long()

            for epoch in range(num_epochs):
                # Perform forward pass: compute predicted y by passing x to the model.
                Y_pred = net(X)
                # Compute loss
                loss = loss_func(Y_pred, Y)
                # Clear the gradients before running the backward pass.
                net.zero_grad()

                # Perform backward pass
                loss.backward()

                # Calling the step function on an Optimiser makes an update to its
                # parameters
                optimizer.step()
            # test accuracy is considered as the fitness value
            X = test_data.iloc[:, list_features]
            X = torch.Tensor(X.values).float()
            Y_pred = net(X)
            _, predicted = torch.max(Y_pred, 1)
            # calculate accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y_test.data.numpy()
            accuracy.append(sum(correct) / total)
            record_accuracy[tuple(list_features)] = sum(correct) / total

    return np.array(accuracy)


# translate binary DNA into a list of features
def translateDNA(pop):
    select_features = []
    row, col = pop.shape
    for r in range(row):
        features = []
        check = pop[r]
        for j in range(len(pop[r])):
            if check[j] == 1:
                features.append(j)
        select_features.append(features)
    return select_features


# define fitness function for selection
def get_fitness(prediction):
    return prediction


# define population select function based on fitness value
# population with higher fitness value has higher chance to be selected
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True, p=fitness / fitness.sum())
    return pop[idx]


# define gene crossover function
def crossover(parent, pop):
    if np.random.rand() < CROSS_RATE:
        # randomly select another individual from population
        i = np.random.randint(0, POP_SIZE, size=1)
        # choose crossover points(bits)
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool_)
        # produce one child
        parent[cross_points] = pop[i, cross_points]
    return parent


# define mutation function
def mutate(child):
    record_digit = []
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0

    # if the number of features larger than given number, randomly delete extra features
    if sum(child) > number_of_features:
        for point in range(DNA_SIZE):
            if child[point] == 1:
                record_digit.append(point)
        slice = set(sample(record_digit, number_of_features))
        for point in range(DNA_SIZE):
            if point in slice:
                child[point] = 1
            else:
                child[point] = 0

    return child


top_features = None
# ----------------------
# initialize a population
# ----------------------
pop1 = [0 for i in range(DNA_SIZE)]
explored = set()
for i in range(number_of_features):
    t = randint(0, DNA_SIZE - 1)
    while t in explored:
        t = randint(0, DNA_SIZE - 1)
    pop1[t] = 1
    explored.add(t)

pop = np.array([pop1] * POP_SIZE)
# ----------------------
# training GA algorithm
# ----------------------
for t in range(N_GENERATIONS):
    # convert binary DNA to a list of features
    pop_DNA = translateDNA(pop)
    # compute target function based on extracted DNA
    F_values = F_function(pop_DNA)

    # calculate fitness value
    fitness = get_fitness(F_values)
    # extract the best DNA
    top_features = pop[np.argmax(fitness), :]
    print("Most fitted DNA: ", top_features)
    print("Accuracy:", np.max(fitness))
    if np.max(fitness) == 1:
        break

    # select better population as parent 1
    pop = select(pop, fitness)
    # make another copy as parent 2
    pop_copy = pop.copy()

    for parent in pop:
        # produce a child by crossover operation
        child = crossover(parent, pop_copy)
        # mutate child
        child = mutate(child)
        # if there no selected features, then replace this DNA with initial one
        if sum(child) == 0:
            child = np.array(pop1)
        # replace parent with its child
        parent[:] = child

print("Most fitted DNA: ", top_features)
features = []
for i in range(len(top_features)):
    if top_features[i] == 1:
        features.append(i)
print("Most fitted feature: ", features)

# ---------------------------------------------------------------------------------
# Extract Rules from a trained neural network with optimal features by decision tree
# ---------------------------------------------------------------------------------

# there are many group of satisfied features, I just chose one set of features
features = [9, 17]

dt = DecisionTreeClassifier()
# training a neural network with the selected features
input = train_input.iloc[:, features]
nn_input = torch.tensor(input.values).float()
net = Net(len(features), hidden_neurons, output_neurons)
loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
Y = torch.Tensor(train_target.values).long()

for epoch in range(num_epochs):
    # Perform forward pass: compute predicted y by passing x to the model.
    Y_pred = net(nn_input)
    # Compute loss
    loss = loss_func(Y_pred, Y)
    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimizer.step()
Y_pred = net(nn_input)
_, predicted = torch.max(Y_pred, 1)
# train a decision tree based on predicted labels and selected features
dt_class_model = dt.fit(train_input.iloc[:, features], predicted.data.numpy())
print("decision tree accuracy", dt_class_model.score(train_input.iloc[:, features], predicted.data.numpy()))
tree.export_graphviz(dt_class_model, "tree1.dot")
