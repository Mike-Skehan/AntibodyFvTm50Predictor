import pandas as pd
import numpy as np


def remove_special_chars(seq_list):
    """
    :param seq_list:    list of aminos acid sequence strings.
    :return:            list of original strings with characters removed.
    """
    chars = ' -?BJOUXZ'
    new_list = []
    for seq in seq_list:
        for char in chars:
            seq = seq.replace(char, '')
        new_list.append(seq)

    return new_list


def data_extract(data_file):
    """
    :param data_file:   csv file containing light sequences, heavy sequences and their tm50 values.
    :return:            lists of light sequences, heavy sequences and tm50 values.
    """

    df = pd.read_csv(data_file)
    df.rename(columns={'VL': 'Light'}, inplace=True)
    df.rename(columns={'VH': 'Heavy'}, inplace=True)
    df.rename(columns={"Fab Tm by DSF (°C)": 'Temp'}, inplace=True)

    light_seq = df['Light'].values.tolist()
    heavy_seq = df['Heavy'].values.tolist()
    temp = df['Temp'].values.tolist()

    light_seq = remove_special_chars(light_seq)
    heavy_seq = remove_special_chars(heavy_seq)

    return light_seq, heavy_seq, temp


def data_extract_class(data_file):
    """
    :param data_file:   csv file containing light sequences, heavy sequences and their tm50 values.
    :return:            lists of light sequences, heavy sequences and tm50 values.
    """

    df = pd.read_csv(data_file)
    df.rename(columns={'VL': 'Light'}, inplace=True)
    df.rename(columns={'VH': 'Heavy'}, inplace=True)
    df.rename(columns={"Fab Tm by DSF (°C)": 'Temp'}, inplace=True)

    light_seq = df['Light'].values.tolist()
    heavy_seq = df['Heavy'].values.tolist()
    temp = df['Temp'].values.tolist()
    bin = df['bin'].values.tolist()

    light_seq = remove_special_chars(light_seq)
    heavy_seq = remove_special_chars(heavy_seq)

    return light_seq, heavy_seq, temp, bin


def data_extract_abY(data_file):
    """

    :param data_file:   csv file containing light sequences, heavy sequences, their scientific names and species.
    :return:            lists of light sequences, heavy sequences, names and species, filtered for mouse & human,
                        less than 150 amino acids
    """
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
    names = df['id'].values.tolist()
    source = df['organism'].values.tolist()

    light_seq = remove_special_chars(light_seq)
    heavy_seq = remove_special_chars(heavy_seq)

    return light_seq, heavy_seq, source, names


def parse_nor_data(data_file):
    """

        :param data_file:   Postgre SQL file output containing light sequences, heavy sequences, Tm50 and sd. Separated
                            by |
        :return:            csv file containing light sequences, heavy sequences, Tm50 values
        """

    df = pd.read_csv(data_file, delimiter='|')
    df.columns = df.columns.str.replace(' ', '')
    df = df.drop([0, 69])
    df['tm'] = df['tm'].str.replace(' ', '')
    df['tm'] = pd.to_numeric(df['tm'])
    df['sd'] = df['sd'].str.replace(' ', '')
    df['sd'] = pd.to_numeric(df['sd'])
    df['heavy'] = df['heavy'].str.replace(' ', '')
    df['heavy'] = df['heavy'].str.replace(' ', '')
    df['heavy'] = df['heavy'].str.replace(' ', '')
    df['tm'].replace('', np.nan, inplace=True)
    df = df.dropna()
    df = df.reset_index()
    df = df.drop('index', axis=1)
    df2 = df.drop('sd', axis=1)
    return df2.to_csv('/CleanedNortheyTmData.csv')
