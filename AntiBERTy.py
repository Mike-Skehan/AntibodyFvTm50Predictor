from igfold import IgFoldRunner
import torch
import data_parser as dp
import pandas as pd

igfold = IgFoldRunner()


def seq2BERTy(heavy, light):
    embed_list = []
    length = len(light)

    for i in range(0, length):
        sequences = {
            "H": heavy[i],
            "L": light[i]
        }

        emb = igfold.embed(
            sequences=sequences,
        )
        berty = emb.bert_embs
        encoded = torch.sum(berty, dim=1)

        embed_list.append(encoded)
        final = torch.cat(embed_list, dim=0)

    return final


if __name__ == '__main__':
    light, heavy, temp = dp.data_extract_Jain("./data/combined_datasets.csv")
    tensor = seq2BERTy(heavy, light)
    X = pd.DataFrame(tensor.detach().numpy())
    X.to_csv('combined_bert_df.csv', index=0)