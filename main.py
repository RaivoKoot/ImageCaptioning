from dataset_utils import Vocab
from dataset import ImageCaptionDataset
import os


if __name__ == '__main__':
    vocab = Vocab()
    vocab_file = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'word2index.txt')
    vocab.load_vocab(vocab_file)

    sample_list_train = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'train_list.txt')
    sample_list_val = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'val_list.txt')
    images_root = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'images')

    train_data = ImageCaptionDataset(sample_list_train, vocab, images_root, transform=None)
    val_data = ImageCaptionDataset(sample_list_val, vocab, images_root, transform=None)
