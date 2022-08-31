import pandas as pd

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


def data_extract_abY(data_file):
    df = pd.read_csv(data_file)

    light_seq = df['light'].values.tolist()
    heavy_seq = df['heavy'].values.tolist()
    names     = df['id'].values.tolist()
    source    = df['organism'].values.tolist()

    return light_seq, heavy_seq, source, names


if __name__ == '__main__':
        light, heavy, source, name = data_extract("./data/AbFv_animal_source.csv")
        print (type(heavy))

        light, heavy, source, name = data_extract_abY("./data/abYsis_data.csv")
        print(type(heavy))

