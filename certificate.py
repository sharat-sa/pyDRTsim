import pandas as pd
from fillpdf import fillpdfs

data = pd.read_excel('names.xlsx')
for name in data['Name']:
    fillpdfs.write_fillable_pdf('template.pdf', f'{name}.pdf', {'field_name': name})