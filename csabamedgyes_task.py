from groq import Groq
import datetime
import re
import os
import pandas as pd
import csv

os.chdir('C:\\Users\\csaba\\Desktop\\task\\task\\')

questions = pd.read_csv("questions.csv", delimiter=';')['QUESTION']

'''
Let's be lazy now. I have no elegant solution for grabbing this information
so we will use models which were created by way smarter mathematicians or 
software engineers than I am.
'''

# %% Create a client and get current date

client = Groq(
    api_key='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'
    )
current_date = datetime.datetime.strptime("2024-10-30", "%Y-%m-%d")

# %% We will insert our prompt for each questiion
      
prompt = '''
    I will need the start and end date from date or time period related elements
    from question.
    
    Sometimes the time periods are not directly included in question, 
    some examples:
        - last year should be tconsidered as 12 month substraction from current date, 
        - if it ask last 3 months then the 3 months should be substracted from current date,
        - Same with 2 quaters should be 6 months substracted from current date
        - Sometimes there is 201# or 202# it refers to 2010s or 2020s but do not exceed current date defined at the end
    But watch out if you have
        - year-over-year or month-over-month it does not count as valid time range
        - if its a general question like this dont return a time range

    only return the date / time period start and end date like this, no need for explanation
    if there exist a time range:
        DD/MM/YYYY, DD/MM/YYYY
    if it does not exist return
        NO_TIME_RANGE, NO_TIME_RANGE
    '''

model_outputs = []

# Process each question and retrieve the responses and save it to model_outputs
for question in questions:

    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "user",
                "content": prompt + \
                    f"Current date is {current_date}.\n Question: {question}\n"
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response_full = ""

    for chunk in completion:

        response_full += chunk.choices[0].delta.content or ""
    
    print(question)
    print(response_full)
    
    model_outputs.append([question, response_full])
    
# %% Save model outputs for checking performance, confusion matrix and final
#  date cleaning.

csv_filename = 'model_outputs.csv'

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Response"])
    writer.writerows(model_outputs)
    
# %% Load the csv here

# TODO

# %% Convert it to dataframe, and clean the dates:
    
model_outputs_df = pd.DataFrame(
    model_outputs, columns=["Question", "Response"])

# %% Getting start and end dates


def separate_dates(row):
    '''
    Parameters
    ----------
    row : dataframe row
        Contains questions and responses as str.

    Returns
    -------
    Two new column values called start date and end date.
        It only returns value if it is
            - "NO_TIME_RANGE", "NO_TIME_RANGE"
            - DD/MM/YYYY, DD/MM/YYYY, 
        formatted.

    The outlier 6 cases will be handled elsewhere.
    '''

    parts = row['Response'].split(", ")
    
    if len(parts) == 2 and all(part != "NO_TIME_RANGE" for part in parts):
        return pd.Series(
            [parts[0].split('\n')[-1],
             parts[1]]
            )
    elif len(parts) == 2 and all(part == "NO_TIME_RANGE" for part in parts):
        return pd.Series(["NO_TIME_RANGE", "NO_TIME_RANGE"])
    
    # Currently dont do anything with outlier responses
    return pd.Series([None, None])


model_outputs_df[['START_DATE', 'END_DATE']] = model_outputs_df.apply(
    separate_dates, axis=1)

# %% Getting outlier responses

na_rows_responses = model_outputs_df[
    model_outputs_df[
        ['START_DATE', 'END_DATE']].isna().any(axis=1)]

print(na_rows_responses)

# %% Fixing outlier responses


def extract_dates(response):
    '''
    Parameters
    ----------
    response : str
        An outlier response.

    Returns
    -------
    Dates as str
        If it has multiple then DD/MM/YYYY, DD/MM/YYYY for each column.
    '''
    # Check for the presence of date patterns
    date_pattern = r'(\d{2}/\d{2}/\d{4})'
    dates = re.findall(date_pattern, response)

    if len(dates) == 4:
        # Start and end dates
        return pd.Series(
            [dates[0] + ' ' + dates[1], 
             dates[2] + ' ' + dates[3]])

    else:
        '''
        Example response:
            'Based on the question, I would extract the time period as follows
                :\n\n01/01/2020, 31/12/2029'
        '''

        start_date = response.replace('\n', '')[-22:-12].strip()
        end_date = response.replace('\n', '')[-10:].strip()
        
        print(start_date, end_date)
        return pd.Series([start_date, end_date])


model_outputs_df[['START_DATE', 'END_DATE']] = model_outputs_df.apply(
    lambda row: extract_dates(row['Response']) 
    if row['START_DATE'] is None and
    row['END_DATE'] is None 
    else pd.Series([row['START_DATE'], row['END_DATE']]), 
    axis=1
)

# %% Final conversion of bad cases


def convert_date_format(date_str):
    try:
        # Attempt to parse the date
        date = pd.to_datetime(date_str, format='%Y-%m-%d')
        return date.strftime('%d/%m/%Y')
    except ValueError:
        return date_str


model_outputs_df['START_DATE'] = model_outputs_df['START_DATE'].apply(
    convert_date_format)
model_outputs_df['END_DATE'] = model_outputs_df['END_DATE'].apply(
    convert_date_format)

# %% Saving model outputs df

model_outputs_df.to_csv('model_outputs_df.csv', index=False, sep=';')
