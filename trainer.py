import torch
from utils import AverageMeter, ProgressMeter
import time
import sys


def run_epoch(epoch_index, dataloader, image_encoder, image_decoder, embedding_block, criterion, optimizer, device, train):
    if train:
        image_encoder.eval()
        image_decoder.train()
        embedding_block.train()
    else:
        image_encoder.eval()
        image_decoder.eval()
        embedding_block.eval()

    # set up metrics
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(dataloader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch_index))

    end = time.time()
    for i, batch in enumerate(dataloader):
        data_time.update(time.time() - end)

        image_batch, word_ids_batch = batch[0].to(device), batch[1].to(device)

        if train:
            train_step(image_batch, word_ids_batch, image_encoder, image_decoder,
                       embedding_block, criterion, top1, losses, optimizer, device)
        else:
            val_step(image_batch, word_ids_batch, image_encoder, image_decoder, embedding_block,
                     criterion, top1, losses, device)

        # logs
        batch_time.update(time.time() - end)
        end = time.time()
        sys.stdout.write('\r'+progress.to_string(i))
        sys.stdout.flush()

    return progress


def train_step(image_batch, word_ids_batch, image_encoder, image_decoder, embedding_block, criterion, acc_metric,
               loss_metric, optimizer, device):
    """
    :param image_batch: shape BATCH x CHANNELS x HEIGHT x WIDTH
    :param word_ids_batch: BATCH x IDS
    :return:
    """
    optimizer.zero_grad()
    # UNROLLED_SPACE x BATCH x OUT_DIM aka SEQ x BATCH x HIDDEN
    image_encodings = image_encoder(image_batch)

    # BATCH x NUM_WORDS
    decoder_targets = word_ids_batch[:, 1:]  # skip <sos> token
    decoder_inputs = word_ids_batch[:, :-1]  # skip <eos> token

    TIME, BATCH, DIM = image_encodings.size()
    TARGET_LEN = decoder_targets.size(1)

    hidden_state = image_decoder.hidden_state_0.repeat((1, BATCH, 1))

    loss = torch.zeros(size=(1,)).to(device)
    for i in range(TARGET_LEN):
        # shape 1
        input_word_id = decoder_inputs[:, i]
        #
        input_word_emb = embedding_block(input_word_id)[None, ...]  # add axis 1 at front

        next_word_pred, hidden_state, att_weights = image_decoder(input_word_emb, image_encodings, hidden_state)
        next_word_pred = next_word_pred[0]  # remove seq axis

        target_word = decoder_targets[:, i]
        loss = loss + criterion(next_word_pred, target_word)
        acc = get_acc(next_word_pred, target_word).item()
        acc_metric.update(acc, BATCH)


    loss.backward()
    optimizer.step()
    loss_metric.update(loss.item(), BATCH)


def val_step(image_batch, word_ids_batch, image_encoder, image_decoder, embedding_block, criterion, acc_metric,
               loss_metric, device):
    # UNROLLED_SPACE x BATCH x OUT_DIM aka SEQ x BATCH x HIDDEN
    image_encodings = image_encoder(image_batch)

    # BATCH x NUM_WORDS
    decoder_targets = word_ids_batch[:, 1:]  # skip <sos> token
    decoder_inputs = word_ids_batch[:, :-1]  # skip <eos> token

    TIME, BATCH, DIM = image_encodings.size()
    TARGET_LEN = decoder_targets.size(1)

    hidden_state = image_decoder.hidden_state_0.repeat((1, BATCH, 1))

    loss = torch.zeros(size=(1,)).to(device)
    for i in range(TARGET_LEN):
        # shape 1
        input_word_id = decoder_inputs[:, i]
        #
        input_word_emb = embedding_block(input_word_id)[None, ...]  # add axis 1 at front

        next_word_pred, hidden_state, att_weights = image_decoder(input_word_emb, image_encodings, hidden_state)
        next_word_pred = next_word_pred[0]  # remove seq axis

        target_word = decoder_targets[:, i]
        loss = loss + criterion(next_word_pred, target_word)
        acc = get_acc(next_word_pred, target_word).item()
        acc_metric.update(acc, BATCH)

    loss_metric.update(loss.item(), BATCH)

def generate_caption(image, image_encoder, image_decoder, embedding_block, vocab, device):
    """
    Given an image, predict a caption for it.
    :param image: a CHANNELS x HEIGHT x WIDTH tensor
    :param image_encoder:
    :param image_decoder:
    :param embedding_block:
    :return: the caption as a list of word strings.
    """
    image = image[None, ...]  # add batch dimension
    image_encoding = image_encoder(image)

    sentence = []
    END_OF_SENTENCE_WORD = '<eos>'
    previous_word = '<sos>'
    hidden_state = image_decoder.hidden_state_0

    while previous_word != END_OF_SENTENCE_WORD:
        input_word_id = vocab.word_to_index(previous_word)

        input_word_tensor = torch.tensor([input_word_id])[None, ...].to(device)  # add sequence and batch dimension
        input_word_tensor = embedding_block(input_word_tensor)

        next_word_pred, hidden_state, att_weights = image_decoder(input_word_tensor, image_encoding, hidden_state)

        # turn logprobabilites to word string
        next_word_pred = next_word_pred[0,0, :]  # remove sequence and batch dimension
        next_word_pred = torch.argmax(next_word_pred)
        next_word_pred = vocab.index_to_word(next_word_pred.item())
        sentence.append(next_word_pred)

        previous_word = next_word_pred

    return sentence

def get_acc(log_proba, targets):
    proba = torch.exp(log_proba)
    equality = (targets == proba.max(dim=1)[1])
    return equality.float().mean()