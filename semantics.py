from pathlib import Path
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import re

def clean(text):
    text = re.sub(r"<br\s*/?>", " ", text)          # strip <br /> tags
    text = re.sub(r"[^a-z0-9' ]", " ", text.lower())  # strip punctuation
    return text.split()

# get the training path on our computer
train_path = Path("path here")

# cd into the positive and negative folders which hold our files data
positive_path = train_path / "pos"
negative_path = train_path / "neg"

# make our positve reviews array
positive_reviews = []

count = 0
# iterate through the files in the positive path, read them, and append them
for file in positive_path.glob("*.txt"):
    review = file.read_text(encoding = "utf-8")
    positive_reviews.append(review)

    #count += 1

    #if count == 2500:
       # break
    #if count % 1000 == 0:
     #   print(f"Loaded {count} positive reviews")

print(f"Finished positive reviews: {len(positive_reviews)}")

# make our negatives reviews array
negative_reviews = []

count2 = 0
# iterate through the files in the negative path, read them, and append them
for file in negative_path.glob("*.txt"):
    review = file.read_text(encoding = "utf-8")
    negative_reviews.append(review)

    count2 += 1

    #if count2 == 2500:
     #   break
    #if count2 % 1000 == 0:
     #   print(f"Loaded {count2} negative reviews")

print(f"Finished negative reviews: {len(negative_reviews)}")

# print the lengths to test them
print(len(positive_reviews))
print(len(negative_reviews))

# put all our reviews in one array, and lables in another
all_reviews = []
labels = []

# add all postive to all reviews
for item in positive_reviews:
    all_reviews.append(item)
    labels.append(1)

# add all negative to all reviews
for item in negative_reviews:
    all_reviews.append(item)
    labels.append(0)

# X_train = reviews used to train the model
# X_val = reviews used to validate the model
# y_train = sentiment labels for the training reviews
# y_val = sentiment labels for the validation reviews
# test_size = 20% of the data is used for validation
# random_state = keeps the random split the same every time
# shuffle = randomly mixes the data before splitting
X_train, X_val, y_train, y_val = train_test_split(
    all_reviews,
    labels,
    test_size=0.2,
    random_state=67,
    shuffle=True
)

# print out our training split
print(len(X_train))
print(len(X_val))
print(len(y_train))
print(len(y_val))

first_review = all_reviews[0].split()
print(first_review)

# make our token dicitonary
token_dictionary = {}
dict_count = 1
# iterate through data in x train
for string in X_train:
    # split the data up
    string = clean(string)
    # for the words in that string
    for word in string:
        # add to dictionary if not in, with number
        if word not in token_dictionary:
            token_dictionary[word] = dict_count
            dict_count += 1

#print(token_dictionary)


# add an unknown token to our dictionary
token_dictionary["<UNK>"] = dict_count
dict_count += 1

# create our token list
token_list = []
# iterate thorugh x train
for string in X_train:
    # split data make new list
    string = clean(string)
    word_list = []
    # add the number to our list
    for word in string:
        if word in token_dictionary:
            word = token_dictionary[word]
            word_list.append(word)
        # if not in use the token UNK
        else:
            word_list.append(token_dictionary["<UNK>"])

    token_list.append(word_list)


#print(token_list)
print()
print()
print(len(token_list))
print(token_list[0])

#print(X_train[0])

# create our validation token list
val_token_list = []

# iterate through X_val
for string in X_val:
    # split the review into words
    string = clean(string)
    word_list = []

    # convert each word into its token number
    for word in string:
        if word in token_dictionary:
            word = token_dictionary[word]
        else:
            word = token_dictionary["<UNK>"]

        word_list.append(word)

    val_token_list.append(word_list)

print(len(val_token_list))
print(val_token_list[0])


# create an empty list to store the true lengths of our training reviews
train_lengths_list = []

# go through every review in our training token list
for review in token_list:

    # get the number of tokens in the review
    review_length = len(review)

    # if the review has more than 500 tokens,
    # we only keep the first 500 tokens
    if review_length > 500:
        review_length = 500

    # add the review's length to our list
    train_lengths_list.append(review_length)


# convert our list of training review lengths into a PyTorch tensor
train_lengths = torch.tensor(train_lengths_list)


# create an empty list to store the true lengths of our validation reviews
val_lengths_list = []

# go through every review in our validation token list
for review in val_token_list:

    # get the number of tokens in the review
    review_length = len(review)

    # if the review has more than 500 tokens,
    # we only keep the first 500 tokens
    if review_length > 500:
        review_length = 500

    # add the review's length to our list
    val_lengths_list.append(review_length)


# convert our list of validation review lengths into a PyTorch tensor
val_lengths = torch.tensor(val_lengths_list)

# make our padded token list
padded_token_list = []
MAX_LENGTH = 500
for review in token_list:
    # if len = 200 keep
    if len(review) == MAX_LENGTH:
        padded_token_list.append(review)

    # if greater than 200 just keep to 200
    elif len(review) > MAX_LENGTH:
        padded_token_list.append(review[:MAX_LENGTH])

    # pad with zeros
    else:
        padded_token_list.append(review + [0] * (MAX_LENGTH - len(review)))



# padded list for the validation list
padded_val_token_list = []
for review in val_token_list:
    # keep if 200
    if len(review) == MAX_LENGTH:
            padded_val_token_list.append(review)

    # only keep first 200
    elif len(review) > MAX_LENGTH:
            padded_val_token_list.append(review[:MAX_LENGTH])

    # pad with 0s
    else:
        padded_val_token_list.append(review + [0] * (MAX_LENGTH - len(review)))

#print("Length of padded token list and the padded validation list: ")
#print(len(padded_token_list))
#print(len(padded_val_token_list))

print("Training reviews:", len(padded_token_list))
print("Validation reviews:", len(padded_val_token_list))
print("Tokens in first training review:", len(padded_token_list[0]))
print("Tokens in first validation review:", len(padded_val_token_list[0]))

# convert our padded review lists into PyTorch tensors
x_train_tensor = torch.tensor(padded_token_list)
x_val_tensor = torch.tensor(padded_val_token_list)

# convert our labels into PyTorch tensors
y_train_tensor = torch.tensor(y_train)
y_val_tensor = torch.tensor(y_val)

# check the shapes
print("Training data:", x_train_tensor.shape)
print("Validation data:", x_val_tensor.shape)
print("Training labels:", y_train_tensor.shape)
print("Validation labels:", y_val_tensor.shape)

# Create a dataset that pairs each training review with its correct sentiment label
# x_train_tensor = 160 training reviews, each represented by 200 token numbers
# y_train_tensor = 160 labels, where 1 = positive and 0 = negative
train_dataset = TensorDataset(x_train_tensor, train_lengths,  y_train_tensor)

# Create a dataset that pairs each validation review with its correct sentiment label
# x_val_tensor = 40 validation reviews, each represented by 200 token numbers
# y_val_tensor = 40 labels, where 1 = positive and 0 = negative
val_dataset = TensorDataset(x_val_tensor, val_lengths, y_val_tensor)

# Create a DataLoader that gives the model the training data in batches of 64 reviews
# shuffle=True randomly changes the order of the training reviews each time
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)

# Create a DataLoader that gives the model the validation data in batches of 64 reviews
# shuffle=False keeps the validation data in its current order
val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)

# get the size of our token dicitonary
print(len(token_dictionary))

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim,padding_idx=0)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.dropout = nn.Dropout(0.5)

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x, lengths):

        x = self.embedding(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, (hidden, cell) = self.lstm(packed)

        x = hidden[-1]

        x = self.dropout(x)

        x = self.fc(x)

        return x


model = SentimentModel(
    vocab_size=len(token_dictionary) + 1,
    embedding_dim=64,
    hidden_dim=128
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print("Using device:", device)

# get our first batch
#reviews, labels = next(iter(train_loader))

# lets check them
#print(reviews.shape)
#print(labels.shape)

#outputs = model(reviews)
#print(outputs.shape)
#print(outputs)

loss_function = nn.CrossEntropyLoss()

#loss = loss_function(outputs, labels)

#print("Outputs:", outputs)
#print("Labels:", labels)
#print("Loss:", loss)

# make our optimizer with a learning rate of .001
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
# clear our old gradients
#optimizer.zero_grad()
# calculate how each weight contributed to error
#loss.backward()
# update the weights
#optimizer.step()

best_val_accuracy = 0.0
for epoch in range(15):

    total_loss = 0
    batch_number = 0
    for reviews, lengths, labels in train_loader:
        batch_number += 1
        reviews, labels = reviews.to(device), labels.to(device)
        #if batch_number % 20 == 0:
         #   print(f"  Epoch {epoch + 1}, batch {batch_number}/{len(train_loader)}")


        # make predictions
        outputs = model(reviews, lengths)

        # calculate our loss
        loss = loss_function(outputs, labels)

        # clear our old gradients
        optimizer.zero_grad()

        # calculate how each weight contributed to error
        loss.backward()

        #update our weights
        optimizer.step()

        # sum up the loss per item to help with our average
        total_loss += loss.item()

    #calculate the average loss for the epoch
    average_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch + 1}, Average Loss: {average_loss:.4f}")

    # check validation accuracy after this epoch
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for reviews, lengths, labels in val_loader:
            reviews, labels = reviews.to(device), labels.to(device)
            outputs = model(reviews, lengths)
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    epoch_val_accuracy = correct / total
    print(f"  -> Validation Accuracy after epoch {epoch + 1}: {correct/total*100:.2f}%")
    if epoch_val_accuracy > best_val_accuracy:
        best_val_accuracy = epoch_val_accuracy
        torch.save(model.state_dict(), "best_model.pt")
        print(f"  -> New best model saved ({epoch_val_accuracy*100:.2f}%)")
    model.train()



correct = 0
total = 0

with torch.no_grad():

    for reviews, lengths, labels in val_loader:
        reviews, labels = reviews.to(device), labels.to(device)

        # Make predictions
        outputs = model(reviews,lengths)

        # Choose the class with the highest score
        predictions = torch.argmax(outputs, dim=1)

        # Count correct predictions
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = correct / total

print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Take the first validation review
review = x_val_tensor[0].unsqueeze(0).to(device)
review_length = val_lengths[0].unsqueeze(0)

# Get the model's prediction
with torch.no_grad():
    output = model(review, review_length)

# Get the predicted class
prediction = torch.argmax(output, dim=1)

print("Model output:", output)
print("Model prediction:", prediction.item())
print("Actual label:", y_val_tensor[0].item())


# Load the best checkpoint
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Load and tokenize the real test set the same way you did train/val
test_path = Path("path here")
test_positive_path = test_path / "pos"
test_negative_path = test_path / "neg"

test_reviews = []
test_labels = []

for file in test_positive_path.glob("*.txt"):
    test_reviews.append(file.read_text(encoding="utf-8"))
    test_labels.append(1)

for file in test_negative_path.glob("*.txt"):
    test_reviews.append(file.read_text(encoding="utf-8"))
    test_labels.append(0)

print(f"Test reviews: {len(test_reviews)}")

# Tokenize using the SAME token_dictionary built from training data
# (do NOT rebuild the dictionary from test data — that would leak info)
test_token_list = []
for string in test_reviews:
    string = clean(string)
    word_list = []
    for word in string:
        if word in token_dictionary:
            word_list.append(token_dictionary[word])
        else:
            word_list.append(token_dictionary["<UNK>"])
    test_token_list.append(word_list)

# Track true lengths, capped at 500
test_lengths_list = []
for review in test_token_list:
    length = min(len(review), 500)
    test_lengths_list.append(length)
test_lengths = torch.tensor(test_lengths_list)

# Pad to 500
padded_test_token_list = []
for review in test_token_list:
    if len(review) >= 500:
        padded_test_token_list.append(review[:500])
    else:
        padded_test_token_list.append(review + [0] * (500 - len(review)))

x_test_tensor = torch.tensor(padded_test_token_list)
y_test_tensor = torch.tensor(test_labels)

test_dataset = TensorDataset(x_test_tensor, test_lengths, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Final evaluation
correct = 0
total = 0
with torch.no_grad():
    for reviews, lengths, labels in test_loader:
        reviews, labels = reviews.to(device), labels.to(device)
        outputs = model(reviews, lengths)
        predictions = torch.argmax(outputs, dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

test_accuracy = correct / total
print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")