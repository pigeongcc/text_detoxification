import pandas as pd

def prepare_data():
    df = pd.read_csv('data/raw/filtered.tsv', delimiter='\t', index_col=0)

    columns = ['reference', 'translation', 'ref_tox', 'trn_tox']
    df['source'] = df[columns].apply(lambda x: x['reference'] if x['ref_tox'] > x['trn_tox'] else x['translation'], axis=1)
    df['target'] = df[columns + ['source']].apply(lambda x: x['translation'] if x['source'] == x['reference'] else x['reference'], axis=1)

    df.to_csv('data/interim/distinguished.tsv', sep='\t', index=False)

    print("Data has been processed and saved successfully.")