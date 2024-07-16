import torch
import torch.nn as nn


class Cbow(nn.Module):

    def __init__(self, vocab_size:int, embed_size:int,*args, **kwargs):
        """
        :param vocab_size: vocabulary size
        :param embed_size: embedding size
        :param args:
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(embed_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:

        # x : (B X window) -> (B X window X embed_size)
        x = self.embed(x)

        # x : (B X window X embed_size) -> (B X 1 X embed_size) -> (B X embed_size)
        x = x.sum(1).squeeze(1)

        # x : (B X embed_size) -> (B X vocab_size)
        x = self.fc(x)

        return self.log_softmax(x)

    def get_embed(self, x:torch.Tensor) -> torch.Tensor:
        """
        :param x: torch.Tensor
        :return:  returns embedding for the input tokens
        """
        # converting to Long type
        x = x.type(torch.int64)

        with torch.inference_mode():
            x = self.embed(x)

        return x

