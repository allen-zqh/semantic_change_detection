import pandas as pd
from tqdm import tqdm

# patent number / title / abstract
todo_list = pd.read_csv(r'g_application.tsv', sep='\t')
todo_list['patent_application_type'] = pd.to_numeric(todo_list['patent_application_type'], errors = 'coerce')
todo_list = todo_list[todo_list['patent_application_type'].between(0, 17)]
todo_list['filing_date'] = pd.to_datetime(todo_list['filing_date'], errors = 'coerce')
todo_list = todo_list[(todo_list['filing_date'] >= '1960-01-01') & (todo_list['filing_date'] <= '2022-12-31')]
todo_list = todo_list[todo_list['patent_id'].str.isdigit()]
todo_list.reset_index(drop=True, inplace=True)
todo_list['patent_id'] = todo_list['patent_id'].astype(int)
todo_list = todo_list[['patent_id', 'filing_date']]

title_abs = pd.read_csv(r'g_patent.tsv', sep='\t')
todo_list = pd.merge(todo_list, title_abs[['patent_id', 'patent_title', 'patent_abstract']], on='patent_id', how='inner')

# patent claims
claims = []
for i in tqdm(range(1976, 2025)): # the file names are from 1976, see https://patentsview.org/download/claims
    dir_name = 'g_claims_' + str(i) + '.tsv'
    temp_file = pd.read_csv(dir_name, sep='\t')
    claims.append(temp_file)
    
claims = pd.concat(claims)
claims = claims.groupby('patent_id')['claim_text'].apply(lambda x: ' '.join(x)).reset_index()

todo_list = pd.merge(todo_list, claims, on='patent_id', how='left')
new_column = {'patent_id': 'patent', 'patent_title': 'title', 'patent_abstract': 'abstract', 'claim_text': 'claims'}
todo_list.rename(columns=new_column, inplace=True)

todo_list[['patent', 'claims']].to_csv(r'claims.csv')
todo_list[['patent', 'title', 'abstract', 'filing_date']].to_csv(r'patent_title_abstract.csv')

#%% 0 - concatenate patent number / title / abstract with claims
import pandas as pd
import math as mt
import re
from datetime import datetime
from tqdm import tqdm

data_dir = './data/'  # Original data directory
out_dir = './output/'  # Output directory

# Input files
claims_file = data_dir+'claims6.csv'
title_abstract_file = data_dir+'patent_title_abstract.csv'

# Output files
pno_file = out_dir+'patent_number.txt'
concat_file = out_dir+'patent_concatenated.txt'
adate_file = out_dir+'patent_adate.txt'

print('Reading claims from CSV...')
claims_data = pd.read_csv(claims_file)
claims_data['claims'] = claims_data['claims'].astype(str)
print('Claims read!')

d_text = {}

print('Concatenating claims from CSV...')
for row in tqdm(claims_data.itertuples()):
    line = ''
    if row.claims != 'nan':
        tokens = row.claims.split()
        first = tokens[0]
        if re.match('\'?.?[0-9]+.?', first) or re.match(';?[a-z].', first):
            tokens = tokens[1:]
        line = ' '.join([token for token in tokens])
    if row.patent not in d_text:
        d_text[row.patent] = ''
    d_text[row.patent] += line+' '
print('Claims concatenated!')

print('Reading title and abstract from CSV...')
title_data = pd.read_csv(title_abstract_file)
title_data['filing_date'] = title_data['filing_date'].astype(str)
title_data['abstract'] = title_data['abstract'].astype(str)
title_data['title'] = title_data['title'].astype(str)
print('Title and abstract read!')


print('Concatenating title and abstract...')
adates = {}
#ayears = {}
l_tuples = []
for row in tqdm(title_data.itertuples()):
    line = ''
    if row.title != 'nan':
        line += ' '+row.title
    if row.abstract != 'nan':
        line += ' '+row.abstract
    if row.patent in d_text:
        line += ' '+d_text[row.patent]
    d_text[row.patent] = line
    adates[row.patent] = row.filing_date
    # Form a tuple with adate, patent number and ayear to sort the patents
    # first by adate and then by patent number
    l_tuples.append((row.filing_date, row.patent))
print('Title and abstract concatenated!')

l_tuples.sort()

print('Saving patent data sorted by adate and patent number...')
with open(pno_file, 'w', encoding='utf-8') as pno_writer,\
        open(concat_file, 'w', encoding='utf-8') as concat_writer,\
        open(adate_file, 'w', encoding='utf-8') as adate_writer:
    for tup in tqdm(l_tuples):
        patent = tup[1]
        filing_date = tup[0]
        adate_writer.write(str(filing_date)+'\n')
        pno_writer.write(str(patent)+'\n')
        concat_writer.write(d_text[patent]+'\n')

print('Patent data saved!')


#%% 1 - clean concatented patents
from nltk.corpus import stopwords
import operator
import re
from tqdm import tqdm


def checkRoman(token):
    """
    Check if a token is a roman numeral.


    Parameters
    ----------
    token : A string.

    Returns
    -------
    True/False : A true value

    """
    re_pattern = '[mdcxvi]+[a-z]'
    if re.fullmatch(re_pattern, token):
        return True
    return False

aux_dir = 'E:/Zhao/patent_novelty/todo/data/' # Original data
data_dir = 'E:/Zhao/patent_novelty/todo/output/' # Processed data

# Input common files
greek_file = aux_dir+'greek.txt'
symbol_file = aux_dir+'symbols.txt'
stop_file = aux_dir+'stopwords.txt' # specific for patent stopwords
concat_file = data_dir+'patent_concatenated.txt'
pno_file = data_dir+'patent_number.txt'
uni_file = data_dir+'cleaned_content.txt'

print('Reading patent numbers...')
patents = []
with open(pno_file, 'r') as pno_reader:
    for line in pno_reader:
        patents.append(line.strip())
print('Patent numbers read!')

print('Reading greek letters, symbols and stop words to remove...')
list_replace = []
with open(greek_file, 'r', encoding='utf-8') as greek_reader:
    greek_reader.readline()
    for line in greek_reader:
        tokens = line.strip().split(',')
        for token in tokens[1:-1]:
            if token != '-':
                tup = tuple((token, tokens[-1]))
                list_replace.append(tup)

with open(symbol_file, 'r', encoding='utf-8') as symbol_reader:
    for line in symbol_reader:
        tokens = line.strip().split(',')
        for token in tokens:
            tup = tuple((token, ' '))
            list_replace.append(tup)

stwrds = stopwords.words('english')
words = []
with open(stop_file, 'r', encoding='utf-8') as stop_reader:
    for line in stop_reader:
        words.append(line.strip())
stwrds.extend(words)
stwrds = set(stwrds)
print('Greek letters, symbols and stop words read!')


print('Cleaning patents...')
# This process could take several hours, depending on the computer
clean_patents = []
with open(concat_file, 'r', encoding='utf-8') as concat_reader:
    for line in tqdm(concat_reader):
        line = line.strip().lower()
        # Standardize greek letters and eliminate symbols
        for r in list_replace:
            line = line.replace(*r)
        # Replace .sub. and .sup. in each patent
        line = line.replace('.sub.', '')
        line = line.replace('.sup.', '')
        # Extract tokens using a regular expression
        tokens = re.findall('[a-z0-9][a-z0-9-]*[a-z0-9]+|[a-z0-9]', line)
        # Remove stopwords, and words of only one char and compossed only
        # of numbers
        tokens = [token for token in tokens if len(token) > 1 and
                  token not in stwrds and
                  not token.replace('-', '').isnumeric()]
        tokens = [token for token in tokens if len(token) > 1 and
                  token not in stwrds and
                  not checkRoman(token)]
        clean_patents.append(tokens)
print('Patents cleaned!')

print('Saving cleaned patent data...')
# Save patent number and patent text
with open(uni_file, 'w', encoding='utf-8') as uni_writer:
    for tokens, patent in zip(clean_patents, patents):
        line = ' '.join(tokens)
        uni_writer.write(patent+','+line+'\n')
print('Patent data saved!')


#%% divide into slices by filing year
import os
from tqdm import tqdm
file_dire = "./data_byyear"
directory = "./split_data"

# e.g., example slices
output_files = {
    "1980_1984.txt": range(1980, 1985),
    "1985_1989.txt": range(1985, 1990),
    "1990_1994.txt": range(1990, 1995),
    "1995_1999.txt": range(1995, 2000),
    "2000_2004.txt": range(2000, 2005),
    "2005_2009.txt": range(2005, 2010),
    "2010_2014.txt": range(2010, 2015),
    "2015_2019.txt": range(2015, 2020),
    "2020_2022.txt": range(2020, 2023)
}


for output_file, years_range in tqdm(output_files.items()):
    with open(os.path.join(directory, output_file), 'w', encoding='utf-8') as outfile:
        for year in years_range:
            filename = "{}.txt".format(year)
            filepath = os.path.join(file_dire, filename)
            with open(filepath, 'r', encoding='utf-8') as infile:
                lines = infile.readlines()
                if lines:
                    lines[-1] = lines[-1].rstrip('\n')
                outfile.write(''.join(lines))
                outfile.write('\n')