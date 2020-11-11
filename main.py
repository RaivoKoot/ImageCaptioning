from dataset_utils import Vocab
from dataset import ImageCaptionDataset, VariableSeqLenBatchSampler, preprocessing_transforms, denormalize
import os
import torch


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
    caption_list_train = [sample['caption'] for sample in train_data.samples]
    caption_list_val = [sample['caption'] for sample in val_data.samples]
    batch_sampler_train = VariableSeqLenBatchSampler(caption_list_train, batch_size=BATCH_SIZE)
    batch_sampler_val = VariableSeqLenBatchSampler(caption_list_val, batch_size=BATCH_SIZE)

    """ 6) initialize dataloaders """
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_sampler=batch_sampler_train, num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_sampler=batch_sampler_val, num_workers=NUM_WORKERS,
        pin_memory=True,
    )