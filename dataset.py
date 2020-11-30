import torch.utils.data
import os
from PIL import Image
from dataset_utils import Vocab
from collections import OrderedDict
from random import shuffle
from torchvision import transforms
import random

class ImageCaptionDataset():
    def __init__(self, sample_list_path, vocab, images_root, transform,
                 imagename_template="COCO_train2014_{:012d}.jpg"):
        """
        Holds a list of samples where each sample is a dictionary containing
        the image file id and the caption of that image as a list of word indices.
        :param sample_list_path: filepath to the txt file that contains
        a row of "image_id, id, caption" per sample.
        :param vocab: a prebuilt Vocab object for splitting string sentences and
        performing word2index.
        """
        self.samples = []
        self.images_root = images_root
        self.transform = transform
        self.img_template = imagename_template

        with open(sample_list_path) as file:
            for i, line in enumerate(file):
                if i == 0:
                    continue  # header line
                sample = line.strip().split(" xSEPERATORx ")

                image_id = int(sample[0])
                caption = sample[2]
                caption = '<sos> ' + caption + ' <eos>'
                words = vocab.splitter(caption)
                word_ids = [vocab.word_to_index(word) for word in words]

                sample = {
                    "image_id": image_id,
                    "caption": torch.LongTensor(word_ids)
                }
                self.samples.append(sample)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_name = self.img_template.format(sample['image_id'])
        img_name = os.path.join(self.images_root, img_name)

        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            print(f"WARNING: Could not find image '{img_name}'. ")
            image = Image.new('RGB', (299, 299))

        if self.transform:
            image = self.transform(image)

        return image, sample['caption']

class VariableSeqLenBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sequence_samples, batch_size):
        """
        A custom dataloader index-sampler that, for each batch,
        samples indices of instances that have the exact same
        sequence length.
        thanks to : https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/11
        :param sequence_samples: A list of sequences where the
        sequence items have a 'shape' attribute where the seq
        dimension comes first. This list of samples must have
        the same order as the samples inside of the Dataset
        object that will be used with this class.
        :param batch_size: number of samples per batch.
        """
        self.batch_size = batch_size
        index_lengths = []
        for i, seq in enumerate(sequence_samples):
            index_lengths.append((i, seq.shape[0]))
        self.index_lengths = index_lengths
        self.batch_list = self._generate_batch_indices()
        self.num_batches = len(self.batch_list)

    def _generate_batch_indices(self):
        """
        Computes the indices to be used in each batch and ensures that
        all indices in a batch belong to sequences with the same length.
        :return:
        1) a list of lists where each inner list holds one set
        of batch indices.
        """
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.index_lengths)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.index_lengths:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.index_lengths)

    def __iter__(self):
        self.batch_list = self._generate_batch_indices()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i

def preprocessing_transforms():
    return transforms.Compose([
        transforms.Resize((299,299)),
        #transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def denormalize(image):
    """
    args:
        video: a (frames, channels, height, width) tensor
    """
    inv_normalize = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )
    return (inv_normalize(image) * 255.).type(torch.uint8).permute(1, 2, 0).numpy()

def take_n(dataset, n):
    """
    Given a dataset with x number of samples, remove random samples from the dataset
    so that n samples are left.
    """
    dataset.samples = random.choices(dataset.samples, k=n)
    return dataset

if __name__ == '__main__':
    """ 1) load vocab object that can do word2index, index2word, and split sentences into words """
    vocab = Vocab()
    vocab_file = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'word2index.txt')
    vocab.load_vocab(vocab_file)

    """ 2) txt files that contain a row for each sample """
    sample_list_train = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'train_list.txt')
    sample_list_val = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'val_list.txt')
    images_root = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'images')

    """ 3) initialize dataset objects """
    train_data = ImageCaptionDataset(sample_list_train, vocab, images_root, transform=preprocessing_transforms())
    val_data = ImageCaptionDataset(sample_list_val, vocab, images_root, transform=preprocessing_transforms())

    """ 4) dataloader parameters """
    BATCH_SIZE = 8
    NUM_WORKERS = 8

    """ 5) custom batch-ids sampler that makes sure a batch contains caption sequences of equal length """
    caption_list = [sample['caption'] for sample in train_data.samples]
    batch_sampler = VariableSeqLenBatchSampler(caption_list, batch_size=BATCH_SIZE)

    """ 6) initialize and test dataloader """
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler=batch_sampler, num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    x, y = next(iter(train_loader))
    print(x.size(), x.dtype)
    print(y.size(), y.dtype)

    import matplotlib.pyplot as plt

    for image, caption in zip(x, y):
        image = denormalize(image)
        caption = [vocab.index_to_word(int(word_id)) for word_id in caption]

        plt.imshow(image)
        plt.title(caption)
        plt.show()
        plt.pause(1)

