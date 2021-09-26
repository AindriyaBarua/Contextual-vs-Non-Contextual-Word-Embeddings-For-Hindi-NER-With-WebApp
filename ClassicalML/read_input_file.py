# Created by Aindriya Barua at

def get_train_data(path):
    texts = []
    labels = []
    read_file = open(path, 'r', encoding="utf8")
    for line in read_file:
        line = line.replace('\n', '')
        if line == 'newline':
            pass
        else:
            items = line.split('\t')
            texts.append(items[0])
            labels.append(items[1])
    read_file.close()

    return texts, labels
