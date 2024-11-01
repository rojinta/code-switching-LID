import fasttext
import fasttext.util
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from tqdm import tqdm
import h5py
import numpy as np
import os

def hinglish_to_hindi(text):
    # Transliterate Hinglish (Latin script) to Hindi (Devanagari script)
    hindi_text = transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
    return hindi_text


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


def build_datasets(file_path, language_pair, type='train', max_length=20):
    texts = read_conll_file(file_path)

    if language_pair == 'en-es':
        ft_lan1 = fasttext.load_model('cc.en.300.bin')
        ft_lan2 = fasttext.load_model('cc.es.300.bin')
        mask_embed_1 = ft_lan1.get_word_vector('<mask>')
        mask_embed_2 = ft_lan2.get_word_vector('<mask>')
        pad_embed = ft_lan1.get_word_vector('<pad>')
    elif language_pair == 'en-hi':
        ft_lan1 = fasttext.load_model('cc.en.300.bin')
        ft_lan2 = fasttext.load_model('cc.hi.300.bin')
        mask_embed_1 = ft_lan1.get_word_vector('<mask>')
        mask_embed_2 = ft_lan2.get_word_vector('<mask>')
        pad_embed = ft_lan1.get_word_vector('<pad>')
    # No other language pairs are supported
    else:
        raise ValueError('Invalid language pair')

    inputs = []
    labels = []
    attention_masks = []

    vec_dict1 = {}  # Cache word vectors to avoid redundant computation
    vec_dict2 = {}

    label_map = {'lang1': 0, 'lang2': 1, 'mixed': 2, 'ne': 3, 'fw': 4, 'ambiguous': 5, 'other': 6, 'unk': 7}
    for sentence in tqdm(texts):
        input_vec = []
        label_id = []
        for word_label in sentence:
            if language_pair != 'en-hi':
                token = word_label[0]
                label = word_label[1]
                if label != 'lang2': # Use English embeddings for the rest of the labels
                    if token not in vec_dict1:
                        vec_dict1[token] = ft_lan1.get_word_vector(token)
                        token_vec = vec_dict1[token]
                    else:
                        token_vec = vec_dict1[token]
                else:
                    if token not in vec_dict2:
                        vec_dict2[token] = ft_lan2.get_word_vector(token)
                        token_vec = vec_dict2[token]
                    else:
                        token_vec = vec_dict2[token]

                input_vec.append(token_vec)

                label_index = label_map.get(label)
                label_id.append(label_index)
            else:
                token = hinglish_to_hindi(word_label[0])  # Transliterate Hinglish to Hindi
                label = word_label[1]
                if label != 'lang2':
                    if token not in vec_dict1:
                        vec_dict1[token] = ft_lan1.get_word_vector(token)
                        token_vec = vec_dict1[token]
                    else:
                        token_vec = vec_dict1[token]
                else:
                    if token not in vec_dict2:
                        vec_dict2[token] = ft_lan2.get_word_vector(token)
                        token_vec = vec_dict2[token]
                    else:
                        token_vec = vec_dict2[token]

                input_vec.append(token_vec)

                label_index = label_map.get(label)
                label_id.append(label_index)

        if len(input_vec) < max_length:
            input_vec += [pad_embed] * (max_length - len(input_vec))
            label_id += [label_map['unk']] * (max_length - len(label_id))
            attention_mask = [1] * len(input_vec) + [0] * (max_length - len(input_vec))
        else:
            input_vec = input_vec[:max_length]
            label_id = label_id[:max_length]
            attention_mask = [1] * max_length
        inputs.append(input_vec)
        labels.append(label_id)
        attention_masks.append(attention_mask)

    os.makedirs(f"{language_pair}", exist_ok=True)
    with h5py.File(f"{language_pair}/{type}", 'w') as f:
        f.create_dataset('inputs', data=np.array(inputs))
        f.create_dataset('labels', data=np.array(labels))
        f.create_dataset('attention_masks', data=np.array(attention_masks))

    with h5py.File(f"{language_pair}/mask_vec", 'w') as f:
        if language_pair == 'en-es':
            f.create_dataset('mask_embed_1', data=np.array(mask_embed_1))
            f.create_dataset('mask_embed_2', data=np.array(mask_embed_2))
        elif language_pair == 'en-hi':
            f.create_dataset('mask_embed_1', data=np.array(mask_embed_1))
            f.create_dataset('mask_embed_2', data=np.array(mask_embed_2))


if __name__ == "__main__":
    build_datasets('../lid_spaeng/train.conll', 'en-es', 'train')
    build_datasets('../lid_spaeng/dev.conll', 'en-es', 'dev')

    build_datasets('../lid_hineng/train.conll', 'en-hi', 'train')
    build_datasets('../lid_hineng/dev.conll', 'en-hi', 'dev')
