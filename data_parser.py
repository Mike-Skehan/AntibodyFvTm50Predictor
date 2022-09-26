import pandas as pd
import numpy as np

def remove_special_chars(seq_list):
        chars = ' -?'
        new_list = []
        for seq in seq_list:
            for char in chars:
                seq = seq.replace(char,'')
            new_list.append(seq)

        return new_list


def data_extract(data_file):
        df = pd.read_csv(data_file)

        df.rename(columns = {' Light':'Light'}, inplace = True)
        df.rename(columns = {' Heavy':'Heavy'}, inplace = True)

        light_seq = df['Light'].values.tolist()
        heavy_seq = df['Heavy'].values.tolist()
        names = df['Name'].values.tolist()
        source = df['Source'].values.tolist()

        l_seq_list = remove_special_chars(light_seq)
        h_seq_list = remove_special_chars(heavy_seq)
        source_list = remove_special_chars(source)


        return l_seq_list, h_seq_list, names, source

def data_extract_Jain(data_file):
    df = pd.read_csv(data_file)
    df.drop([0, 4])
    df.rename(columns={'VL': 'Light'}, inplace=True)
    df.rename(columns={'VH': 'Heavy'}, inplace=True)
    df.rename(columns={"Fab Tm by DSF (°C)": 'Temp'}, inplace=True)

    light_seq = df['Light'].values.tolist()
    heavy_seq = df['Heavy'].values.tolist()
    temp = df['Temp'].values.tolist()

    # l_seq_list = remove_special_chars(light_seq)
    # h_seq_list = remove_special_chars(heavy_seq)

    return light_seq, heavy_seq, temp

def data_extract_abY(data_file):
    df = pd.read_csv(data_file)

    df['light'].replace('', np.nan, inplace=True)
    df['heavy'].replace('', np.nan, inplace=True)

    df = df.dropna()

    df = df[(df.organism == 'mus musculus') | (df.organism == 'homo sapiens')]
    df["heavy_length"] = df["heavy"].str.len()
    df["light_length"] = df["light"].str.len()

    df = df[(df.heavy_length <= 150) & (df.heavy_length >= 80) & (df.light_length <= 150) & (df.light_length >= 80)]

    light_seq = df['light'].values.tolist()
    heavy_seq = df['heavy'].values.tolist()
    names     = df['id'].values.tolist()
    source    = df['organism'].values.tolist()

    return light_seq, heavy_seq, source, names


if __name__ == '__main__':
        #light, heavy, source, name = data_extract("./data/AbFv_animal_source.csv")
        #print (type(heavy))

        light, heavy,temp = data_extract_Jain("./data/Jain_Ab_dataset.csv")
        for x in heavy:
            print (len(x))
