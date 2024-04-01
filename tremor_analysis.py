import pandas as pd
import numpy as np
import streamlit as st
import io
import re

def process_excel_sheet(sheet, parameters):
    
    data = sheet.iloc[0:256, :].set_index('f (Hz)').iloc[:, 2:]
    new_cols = np.linspace(0, parameters['col_time'] * (sheet.shape[1]-1), sheet.shape[1])/60
    data.columns = new_cols[2:-1]

    col_bins = np.linspace(0, parameters['exp_length'] , int(parameters['exp_length'] / parameters['bin_time']) +1)
    frequency_bins  = pd.cut(data.index, bins = parameters['hz'])
    time_bins = pd.cut(data.columns, bins = col_bins) 
    a = data.groupby([frequency_bins], observed = False).mean().T.groupby([time_bins], observed = False).mean() * 1000

    non_target_columns = [col for col in range(len(a.columns)) if col != parameters['tremor_freq']]
    tpr = 2 * a.iloc[:, parameters['tremor_freq']] / a.iloc[:, non_target_columns].sum(axis=1)
    # tpr = 2 * a.iloc[:,parameters['tremor_freq']] / (a.iloc[:,0]+ a.iloc[:,2])

    return a, tpr
   
   

def compile_excel_sheets(excel_tables, animals, animal_key, parameters):
    columns = pd.MultiIndex(levels=[[], []], codes=[[], []], names=[ 'Group', 'Subject'])
    tprs = pd.DataFrame(columns = columns)

    columns2 = pd.MultiIndex(levels=[[], [], []], codes=[[], [],[]], names=[ 'Group', 'Subject', 'Frequency'])
    freqs = pd.DataFrame(columns=columns2)

    all_freq = []
    for an in animals:
        if int(an) in animal_key.index:
            st.write('Calcuating animal :'+str(an))
            group = animal_key.loc[int(an), 'GROUP']
            column_index = (group, an)

            freq, trp = process_excel_sheet(excel_tables[an], parameters)
            
            multi_index = pd.MultiIndex.from_arrays([[group] * 3, [an] * 3, freq.columns], names=['Group', 'Subject', 'Frequency'])
            freq.columns = multi_index
            # freqs[multi_index] = freq

            all_freq.append(freq)
            tprs[column_index] = trp

    freqs = pd.concat(all_freq, axis = 1)

    return freqs, tprs


st.title('Tremor and Movement Data Analysis')
excel_file = st.file_uploader(label = "Select Excel File for Tremor Analysis", accept_multiple_files= False)
key_file = st.file_uploader(label = "Select Excel file with Animal/Group Key", accept_multiple_files = False)

if (excel_file is not None) and (key_file is not None):
    # num_bands = st.number_input(label = 'Enter number of frequency bands', min_value = 2, value = 3)
    # for a in range(num_bands):
    lower = st.number_input(label = 'Enter lower bound of tremor band', min_value = 2, max_value = 25, value = 8)
    higher = st.number_input(label = 'Enter higher bound of tremore band', min_value = 2, max_value = 25, value = 14)

    hz = [2,lower, higher, 25]

    bin_mins = st.number_input(label = 'Enter bin size in minutes', value = 5)
    exp_mins = st.number_input(label = 'Enter the length of experiment in minutes', value = 60)

    parameters = {'hz': hz, 'bin_time': bin_mins, 'col_time':10.24, 'exp_length' : exp_mins, 'tremor_freq': 1}

    excel_tables = pd.read_excel(excel_file, sheet_name = None,  engine='openpyxl')
    animal_key = pd.read_excel(key_file, index_col = 0,  engine='openpyxl')

    animals = [a for a in excel_tables.keys() if a != 'Summary']

    freqs, tprs = compile_excel_sheets(excel_tables, animals, animal_key, parameters)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        freqs.to_excel(writer, sheet_name = "All Frequencies")
        tprs.to_excel(writer, sheet_name = 'Tremor Power Ratios')
    

    st.download_button(
        label="Download Excel workbook",
        data= buffer,
        file_name="workbook.xlsx",
        mime="application/vnd.ms-excel"
    )