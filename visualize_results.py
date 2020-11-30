from dataset_utils import Vocab
from dataset import ImageCaptionDataset, VariableSeqLenBatchSampler, preprocessing_transforms, denormalize, take_n
from trainer import generate_caption
from model import ImageEncoder, CaptionRNN
import matplotlib.pyplot as plt
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
    train_data = ImageCaptionDataset(sample_list_train, vocab, images_root, transform=preprocessing_transforms(),
                                     imagename_template="COCO_train2014_{:012d}.jpg")
    val_data = ImageCaptionDataset(sample_list_val, vocab, images_root, transform=preprocessing_transforms(),
                                   imagename_template="COCO_val2014_{:012d}.jpg")
    val_data = take_n(val_data, n=25000)  # of 200 000
    #train_data = take_n(train_data, n=200000)  # of 400 000

    """ 4) dataloader parameters """
    BATCH_SIZE = 32
    NUM_WORKERS = 3

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

    """ 7) Create Model """
    VOCAB_SIZE = len(vocab.word2index)
    IMAGE_EMB_DIM = 256
    WORD_EMB_DIM = 256
    HIDDEN_DIM = 512
    word_embedding = torch.nn.Embedding(
        num_embeddings=VOCAB_SIZE,
        embedding_dim=WORD_EMB_DIM,
    )
    image_encoder = ImageEncoder(out_dim=IMAGE_EMB_DIM)
    image_decoder = CaptionRNN(
        num_classes=VOCAB_SIZE,
        word_emb_dim=WORD_EMB_DIM,
        img_emb_dim=IMAGE_EMB_DIM,
        hidden_dim=HIDDEN_DIM
    )
    word_embedding.eval()
    image_encoder.eval()
    image_decoder.eval()


    """ 9) Load Weights """
    LOAD_WEIGHTS = True
    EMBEDDING_WEIGHT_FILE = 'checkpoints/BIGDATASET-weights-embedding-epoch-3.pt'
    ENCODER_WEIGHT_FILE = 'checkpoints/BIGDATASET-weights-encoder-epoch-3.pt'
    DECODER_WEIGHT_FILE = 'checkpoints/BIGDATASET-weights-decoder-epoch-3.pt'
    if LOAD_WEIGHTS:
        print("Loading pretrained weights...")
        word_embedding.load_state_dict(torch.load(EMBEDDING_WEIGHT_FILE))
        image_encoder.load_state_dict(torch.load(ENCODER_WEIGHT_FILE))
        image_decoder.load_state_dict(torch.load(DECODER_WEIGHT_FILE))

    """ 10) Device Setup"""
    device = 'cuda:1'
    device = torch.device(device)

    word_embedding = word_embedding.to(device)
    image_encoder = image_encoder.to(device)
    image_decoder = image_decoder.to(device)

    print(vocab.word_to_index('yooooo'))

    for i, batch in enumerate(val_loader):
        image_batch, word_ids_batch = batch[0].to(device), batch[1].to(device)

        for image in image_batch:
            sentence = generate_caption(image, image_encoder, image_decoder, word_embedding, vocab, device)
            image = denormalize(image.cpu())
            plt.imshow(image)
            plt.title(sentence)
            plt.show()
            plt.pause(1)