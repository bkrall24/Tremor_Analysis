import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import io
import re

def get_group_responses(excel_tables, animal_key, group = 2):
    target_group = animal_key.index[animal_key['GROUP'] == group].tolist()

    for ind, an in enumerate(target_group):
        sheet = excel_tables[str(an)]
        data = sheet.iloc[0:256,:].set_index(sheet.columns[0]).iloc[:,2:]
        new_cols = np.linspace(0, parameters['col_time'] * (sheet.shape[1]-1), sheet.shape[1])/60
        data.columns = new_cols[2:-1]

        col_bins = np.linspace(0, parameters['exp_length'] , int(parameters['exp_length'] / parameters['bin_time']) +1)
        time_bins = pd.cut(data.columns, bins = col_bins) 
        if ind == 0:
            sum_df = data.T.groupby([time_bins], observed = False).mean().T
        else:
            sum_df += data.T.groupby([time_bins], observed = False).mean().T

    average_df = sum_df / len(target_group)

    return average_df

def plot_group_responses(average_df):
    cmap = plt.get_cmap('turbo')
    colors = [cmap(i) for i in np.linspace(0, 1, 13)]  # Sample 10 colors

    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    fig, ax = plt.subplots()
    ax.plot(average_df.index, average_df)
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which = 'minor', linestyle=':', color='gray')
    ax.grid(axis='x', which = 'major', linestyle=':', color='black')

    plt.legend(average_df.columns)
    st.pyplot(fig) 

    return fig


def process_excel_sheet(sheet, parameters):
    
    # data = sheet.iloc[0:256, :].set_index('f (Hz)').iloc[:, 2:]
    data = sheet.iloc[0:256,:].set_index(sheet.columns[0]).iloc[:,2:]
    new_cols = np.linspace(0, parameters['col_time'] * (sheet.shape[1]-1), sheet.shape[1])/60
    data.columns = new_cols[2:-1]

    col_bins = np.linspace(0, parameters['exp_length'] , int(parameters['exp_length'] / parameters['bin_time']) +1)
    frequency_bins  = pd.cut(data.index, bins = parameters['hz'])
    time_bins = pd.cut(data.columns, bins = col_bins) 
    a = data.groupby([frequency_bins], observed = False).mean().T.groupby([time_bins], observed = False).mean() * 1000

    non_target_columns = [col for col in range(len(a.columns)) if col != parameters['tremor_freq']]
    tpr = 2 * a.iloc[:, parameters['tremor_freq']] / a.iloc[:, non_target_columns].sum(axis=1, skipna = False)
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
            # st.write_stream('Calculating animal :'+an)
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

st.title('Tremor Data Analysis')

# File uploader for Excel data and key
excel_file = st.file_uploader(label="Select Excel File for Tremor Analysis", accept_multiple_files=False)
key_file = st.file_uploader(label="Select Excel file with Animal/Group Key", accept_multiple_files=False)

@st.cache_data
def load_excel_data(file):
    return pd.read_excel(file, sheet_name=None)

@st.cache_data
def load_animal_key(file):
    return pd.read_excel(file, index_col=0)

# Check for file uploads
if excel_file and key_file:
    # Load data from files with caching
    excel_tables = load_excel_data(excel_file)
    animal_key = load_animal_key(key_file)
    st.text("Excel Files Loaded")

    # Set up user inputs for parameters
    

    bin_mins = st.number_input(label='Enter bin size in minutes', value=5)
    exp_mins = st.number_input(label='Enter the length of experiment in minutes', value=60)


    parameters = {
        'bin_time': bin_mins,
        'col_time': 10.24,
        'exp_length': exp_mins,
        'tremor_freq': 1,
    }

    group_choice = st.number_input(label = 'Choose group to determine frequency bands', value = 2)
    # Plot button and group responses
    if st.button('Plot Frequency Responses'):
        if group_choice in animal_key['GROUP'].unique():
            # Generate average_df only if the group is found and store it in session state
            st.session_state['average_df'] = get_group_responses(excel_tables, animal_key, group=group_choice)
        else:
            st.text(f'Group {group_choice} not found in Animal/Group Key')

    # Display plot if it exists in session state
    if 'average_df' in st.session_state:
        fig = plot_group_responses(st.session_state['average_df'])
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)  # Move to the start of the buffer

        # Download button for the plot
        st.download_button(
            label="Download Plot as PNG",
            data=buffer,
            file_name="plot.png",
            mime="image/png"
        )

    lower = st.number_input(label='Enter lower bound of tremor band', min_value=2, max_value=25, value=8)
    higher = st.number_input(label='Enter upper bound of tremor band', min_value=2, max_value=25, value=14)
    parameters['hz'] =  [2, lower, higher, 25]

    # Calculate button and frequency data
    if st.button('Calculate'):
        animals = list(excel_tables.keys())[1:]  # Exclude 'Summary' sheet

        # Cache-intensive calculations based on inputs and file content
        @st.cache_data
        def compute_frequencies_and_tremor_ratios(excel_tables, animals, animal_key, parameters):
            freqs, tprs = compile_excel_sheets(excel_tables, animals, animal_key, parameters)
            return freqs.sort_index(axis=1), tprs.sort_index(axis=1)

        freqs, tprs = compute_frequencies_and_tremor_ratios(excel_tables, animals, animal_key, parameters)

        # Save to Excel and allow download
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
            freqs.to_excel(writer, sheet_name="All Frequencies")
            tprs.to_excel(writer, sheet_name='Tremor Power Ratios')
            if 'average_df' in st.session_state:
                st.session_state['average_df'].to_excel(writer, sheet_name = 'Average Responses Group '+str(group_choice))

        st.download_button(
            label="Download Excel workbook",
            data=buffer,
            file_name="workbook.xlsx",
            mime="application/vnd.ms-excel"
        )