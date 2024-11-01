def read_conll_file(file_path):
    sentences = []
    sentence = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if '#' in line:
                continue
            # Strip any surrounding whitespace or newline
            line = line.strip()

            # If line is empty, it marks the end of a sentence
            if not line:
                if sentence:  # Add sentence if not empty
                    sentences.append(sentence)
                    sentence = []  # Reset for next sentence
            else:
                # Split the line by whitespace to get columns (e.g., token, POS tag, etc.)
                columns = line.split()
                sentence.append(columns)

        # Add the last sentence if file does not end with an empty line
        if sentence:
            sentences.append(sentence)

    return sentences


train = read_conll_file('../lid_hineng/train.conll')
val = read_conll_file('../lid_hineng/dev.conll')
test = read_conll_file('../lid_hineng/test.conll')

# Check all labels
labels = set()
for sentence in train:
    for token in sentence:
        labels.add(token[1])
for sentence in val:
    for token in sentence:
        labels.add(token[1])

print(labels)

# See the distribution of labels
label_distribution = {}
for sentence in train:
    for token in sentence:
        label = token[1]
        if label in label_distribution:
            label_distribution[label] += 1
        else:
            label_distribution[label] = 1

print(label_distribution)
# plot the distribution
import matplotlib.pyplot as plt

plt.bar(label_distribution.keys(), label_distribution.values())
plt.xlabel('Label')
plt.ylabel('Frequency')
plt.title('Label Distribution')

plt.show()