import torch.utils.data
import os
from skimage import io
from dataset_utils import Vocab

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

        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)

        return image, sample['caption']


if __name__ == '__main__':
    vocab = Vocab()
    vocab_file = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'word2index.txt')
    vocab.load_vocab(vocab_file)

    sample_list = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'train_list.txt')
    images_root = os.path.join(os.path.expanduser('~'), 'data', 'raivo', 'coco', 'images')
    dataset = ImageCaptionDataset(sample_list, vocab, images_root, None)

    img, caption_ids = dataset[500]

    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.title([vocab.index_to_word(int(id)) for id in caption_ids])
    plt.show()

