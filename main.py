from dataset_utils import Vocab
from dataset import ImageCaptionDataset, VariableSeqLenBatchSampler, preprocessing_transforms, denormalize, take_n
from trainer import run_epoch
from model import ImageEncoder, CaptionRNN
import os
import torch
import logging


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
    IMAGE_EMB_DIM = 512
    WORD_EMB_DIM = 512
    HIDDEN_DIM = 1024
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

    """ 8) Create Optimizer and Loss Function """
    LR = 0.001
    WEIGHT_DECAY = 0.
    loss_fn = torch.nn.NLLLoss()
    parameters = list(image_decoder.parameters()) + list(word_embedding.parameters())
    optim = torch.optim.Adam(
        params=parameters,
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    """ 9) Load Weights """
    LOAD_WEIGHTS = False
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

    """ 11) text file logging """
    log_filename = f"logs/training_log-BIGDATASET-BIGMODEL.log"
    logging.basicConfig(filename=log_filename, level=logging.DEBUG)

    EPOCHS = 100
    START_EPOCH = 0

    print("Beginning Training")
    for epoch in range(START_EPOCH, EPOCHS):
        # TRAIN
        results = run_epoch(epoch, train_loader, image_encoder, image_decoder, word_embedding, loss_fn, optim, device,
                            train=True)
        print(results.to_string(-1))
        logging.debug(results.to_string(-1))

        # VAL
        results = run_epoch(epoch, val_loader, image_encoder, image_decoder, word_embedding, loss_fn, optim, device,
                            train=False)
        print('Val ' + results.to_string(-1))
        logging.debug('Val ' + results.to_string(-1))

        # SAVE
        torch.save(word_embedding.state_dict(), f"checkpoints/BIGMODEL-BIGDATASET-weights-embedding-epoch-{epoch}.pt")
        torch.save(image_encoder.state_dict(), f"checkpoints/BIGMODEL-BIGDATASET-weights-encoder-epoch-{epoch}.pt")
        torch.save(image_decoder.state_dict(), f"checkpoints/BIGMODEL-BIGDATASET-weights-decoder-epoch-{epoch}.pt")

