import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ImageEncoder(nn.Module):
    """
    Encodes an image into a sequence of vectors produced
    by unrolling the output feature-cube of a CNN.
    """
    def __init__(self, out_dim=256):
        super(ImageEncoder, self).__init__()
        self.feature_extractor = torch.hub.load('pytorch/vision:v0.6.0', 'inception_v3', pretrained=True)
        self.feature_extractor.avgpool = nn.Identity()
        self.feature_extractor.dropout = nn.Identity()
        self.feature_extractor.fc = nn.Identity()
        INCEPTIONV3_OUT_DIM = 2048
        self.IMAGE_FEATS = INCEPTIONV3_OUT_DIM
        self.out = nn.Linear(INCEPTIONV3_OUT_DIM, out_dim)

    def forward(self, image_batch):
        # BATCH x 3 x 299 x 299
        image_feats = self.feature_extractor(image_batch)
        # BATCH x FEATURES x 8 x 8
        image_feats = image_feats.view((-1, self.IMAGE_FEATS, 8 * 8))
        # BATCH x FEATURES x 64
        image_feats = image_feats.permute((0, 2, 1))
        # BATCH x 64 x FEATURES
        image_feats = self.out(image_feats)
        # BATCH x 64 FEATURES_OUT
        image_feats = image_feats.permute((1, 0, 2))
        # 64 x BATCH x FEATURES_OUT
        return image_feats

class Attention(nn.Module):
    """
    Given a context sequence and a query vector, uses the query
    to compute a weighed sum of the context sequence, based on
    an attention mechanism.
    """

    def __init__(self, context_dim=256, query_dim=512, attention_dim=256):
        super(Attention, self).__init__()
        self.query_proj = nn.Linear(query_dim, attention_dim)
        self.context_proj = nn.Linear(context_dim, attention_dim)
        self.register_buffer('scaling_term', torch.Tensor([1/math.sqrt(attention_dim)]))

    def forward(self, features, query):
        """
        Computes an attention-weighted sum of an image feature-vector sequence
        based on a query vector.
        :param features: feature sequence to attend over of size
        SEQ x BATCH x INPUT_DIM. SEQ is typically 8*8=64 for example, from the unrolled
        spatial dimensions of a cnn feature cube that is INPUT_DIM x 8 x 8.
        :param query: vectors to attend from of size
        1 x BATCH x INPUT_DIM
        :return:
        1) weighted sums of features of size 1 x BATCH x INPUT_DIM
        2) attention weights of size BATCH x SEQ x 1
        """
        # project query and keys into a learnt space for dot product attention
        keys = self.context_proj(features)
        # SEQ x BATCH x ATT_DIM
        query = self.query_proj(query)
        # 1 x BATCH x ATT_DIM

        # prepare query and key batches for dot-product attention
        keys = keys.permute((1, 0, 2))
        # BATCH x SEQ x ATT_DIM
        query = query.permute((1, 2, 0))
        # BATCH x ATT_DIM x 1
        attention_matrix = torch.matmul(keys, query) * self.scaling_term # dot-product attention
        # BATCH x SEQ x 1
        attention_weights = F.softmax(attention_matrix, dim=1)
        # BATCH x SEQ x 1
        # softmax-normalize over sequence dimension

        features = features.permute((1, 0, 2))
        # BATCH x SEQ x FEAT_DIM
        att_output = torch.sum((features * attention_weights), dim=1)
        # BATCH x FEAT_DIM
        att_output = torch.unsqueeze(att_output, dim=0)
        # 1 x BATCH x FEAT_DIM

        return att_output, attention_weights

    def ___init__backup(self, context_dim=256, query_dim=512):
        # TODO: SOLVE THIS SHIT
        super(Attention, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=1,
            kdim=context_dim,
            vdim=context_dim
        )

    def _forward_backup(self, features, query):
        """
        Computes an attention-weighted sum of an image feature-vector sequence
        based on a query vector.
        :param features: feature sequence to attend over of size
        SEQ x BATCH x INPUT_DIM
        :param query: vectors to attend from of size
        1 x BATCH x INPUT_DIM
        :return:
        1) weighted sums of features of size 1 x BATCH x INPUT_DIM
        2) attention weights of size BATCH x 1 x SEQ
        """
        # TODO: code MHA from scratch
        return self.attention(query, features, features)




class CaptionRNN(nn.Module):
    """
    Model that predicts one word at a time by repeatedly attending
    over patches of an image.
    """
    def __init__(self, num_classes, word_emb_dim=256, img_emb_dim=256, hidden_dim=512):
        super(CaptionRNN, self).__init__()
        num_layers = 1
        self.hidden_state_0 = nn.Parameter(torch.randn((1, 1, hidden_dim)))

        input_dim = word_emb_dim + img_emb_dim
        self.attention = Attention(context_dim=img_emb_dim, query_dim=hidden_dim, attention_dim=256)
        self.GRU = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, num_classes),
            nn.LogSoftmax(dim=2)  # TODO: double check this
        )

    def forward(self, word_emb, img_feat, prev_hidden_state):
        """
        Predicts the next word in a sequence given the previous hidden
        state, the previous word embedding, and a sequence of image feature-vectors
        for context to perform attention on.
        :param word_emb: word embedding of the previously predicted word of shape
        1 x BATCH x WORD_DIM
        :param img_feat: image embeddings as a sequence of feature-vectors of
        shape SEQ x BATCH x IMG_DIM
        :param prev_hidden_state: the previous hidden state of this model or the initial
        hidden state self.hidden_state_0 of shape (1 x BATCH x IMG_DIM)
        :return:
        1) Logsoftmaxed class activations for the next word of size 1 x BATCH x NUM_CLASSES
        2) Current Hidden state of the GRU for each layer of size LAYERS x BATCH x HIDDEN_SIZE
        3) Image attention weights for predicting the current word of size BATCH x SEQ x 1
        """
        img_feat, att_weights = self.attention(img_feat, prev_hidden_state)
        # 1 x BATCH x IMG_DIM
        # att_weights are BATCH x SEQ x 1 but not used anymore

        model_input = torch.cat((word_emb, img_feat), dim=2)
        # 1 x BATCH x IMG_DIM+WORD_DIM

        output, hidden_state = self.GRU(model_input, prev_hidden_state)
        # 1 x BATCH x HIDDEN_DIM
        # LAYERS x BATCH x HIDDEN_DIM

        output = self.head(output)
        return output, hidden_state, att_weights

if __name__ == '__main__':
    net = CaptionRNN(num_classes=3, word_emb_dim=256, img_emb_dim=256, hidden_dim=512)

    output, hidden_state, att_weights = net(
        word_emb=torch.randn((1,5,256)),
        img_feat=torch.randn((64, 5, 256)),
        prev_hidden_state=net.hidden_state_0.repeat(1, 5, 1)
    )

    print(output.size())
    print(hidden_state.size())
    print(att_weights.size())


    """
    attention = Attention(context_dim=256, query_dim=512, attention_dim=256)
    features = torch.randn((64, 32, 256))
    hidden = torch.randn((1, 32, 512))

    output, att_weights = attention(features, hidden)
    print(output.size())
    print(att_weights.size())
    """