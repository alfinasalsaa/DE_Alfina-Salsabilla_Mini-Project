import pandas as pd
from bs4 import BeautifulSoup
import sqlite3
import numpy as np
import requests
from sqlalchemy import create_engine
from pycountry import countries
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from collections import defaultdict 
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime 
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator

def clean_column_names(df, non_countries):
    """
    Clean column names by trimming whitespace and replacing spaces with underscores.

    Args:
    - df (DataFrame): Input DataFrame.
    - non_countries (list): List of columns that are not country names.

    Returns:
    - DataFrame: DataFrame with cleaned column names.
    """
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns] 
    return df

def convert_to_datetime(df, columns):
    """
    Convert specified columns to datetime format.

    Args:
    - df (DataFrame): Input DataFrame.
    - columns (list): List of columns to convert to datetime.

    Returns:
    - None: Modifies DataFrame in place.
    """
    for col in columns:
        df[col] = pd.to_datetime(df[col])

def remove_commas_and_convert_to_float(df, columns):
    """
    Remove commas and convert specified columns to float format.

    Args:
    - df (DataFrame): Input DataFrame.
    - columns (list): List of columns to clean and convert to float.

    Returns:
    - None: Modifies DataFrame in place.
    """
    for col in columns:
        df[col] = df[col].str.replace(',', '').astype(float)

def clean_string_columns(df, columns):
    """
    Clean string columns by removing specified characters.

    Args:
    - df (DataFrame): Input DataFrame.
    - columns (list): List of columns to clean.

    Returns:
    - None: Modifies DataFrame in place.
    """
    for col in columns:
        df[col] = df[col].apply(lambda x: x.split('!$!')[0] if isinstance(x, str) else x)

def fill_na_with_value(df, columns, value):
    """
    Fill missing values in specified columns with a specified value.

    Args:
    - df (DataFrame): Input DataFrame.
    - columns (list): List of columns to fill missing values.
    - value: Value to fill missing values with.

    Returns:
    - None: Modifies DataFrame in place.
    """
    for col in columns:
        df[col].fillna(value, inplace=True)

def replace_nan_with_zero(df, columns):
    """
    Replace NaN values in specified columns with zero.

    Args:
    - df (DataFrame): Input DataFrame.
    - columns (list): List of columns to replace NaN values.

    Returns:
    - None: Modifies DataFrame in place.
    """
    for col in columns:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

def fill_missing_values_gdp_population(gdp_df, population_df, columns_to_fill):
    """
    Fill missing values in GDP and population DataFrames using backward and forward filling methods.

    Args:
    - gdp_df (DataFrame): GDP DataFrame.
    - population_df (DataFrame): Population DataFrame.
    - columns_to_fill (list): List of columns to fill missing values.

    Returns:
    - tuple: Tuple of modified GDP and population DataFrames.
    """
    gdp_df[columns_to_fill] = gdp_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    population_df[columns_to_fill] = population_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    return gdp_df, population_df

def impute_mean_and_mode_projects(projects_df):
    """
    Impute missing values in projects DataFrame with mean for numerical columns and mode for categorical columns.

    Args:
    - projects_df (DataFrame): Projects DataFrame.

    Returns:
    - DataFrame: Modified projects DataFrame.
    """
    boardapprovaldate_mean = projects_df['boardapprovaldate'].mean()
    closingdate_mean = projects_df['closingdate'].mean()
    projects_df['boardapprovaldate'].fillna(boardapprovaldate_mean, inplace=True)
    projects_df['closingdate'].fillna(closingdate_mean, inplace=True)

    columns_to_fill = ['supplementprojectflg', 'countryname', 'prodline', 'lendinginstr', 'productlinetype', 'projectstatusdisplay', 'status']
    for column in columns_to_fill:
        most_frequent_value = projects_df[column].mode()[0]
        projects_df[column].fillna(most_frequent_value, inplace=True)
    return projects_df

def clean_columns(dfs):
    """
    Remove specified columns from DataFrames.

    Args:
    - dfs (list): List of DataFrames to clean.

    Returns:
    - list: List of cleaned DataFrames.
    """
    for df in dfs:
        df.drop(columns=['indicator_name', 'indicator_code'], inplace=True)
    return dfs

def encode_categorical_columns(df):
    """
    Encode categorical columns using one-hot encoding.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - DataFrame: DataFrame with encoded categorical columns.
    """
    for column in df.columns:
        if df[column].dtype == 'object':
            if (11 <= df[column].nunique() < 50) or (column in ['countryname', 'country_name', 'country_code', 'country_or_area', 'prodline', 'lendinginstr', 'productlinetype', 'projectstatusdisplay', 'status']):
                df = pd.get_dummies(df, columns=[column], prefix=column)
            else:
                df.drop([column], axis=1, inplace=True)
    return df.astype(int)

def add_country_codes(df):
    """
    Add ISO-3 country codes to DataFrame based on country names.

    Args:
    - df (DataFrame): Input DataFrame.

    Returns:
    - DataFrame: DataFrame with added country codes.
    """
    # Manually defined country codes
    country_codes = {
        'Africa': 'AFR',
        'Andean Countries': 'AND',
        'Aral Sea': 'ARL',
        'Asia': 'ASI',
        'Caribbean': 'CAR',
        'Caucasus': 'CAU',
        'Central Africa': 'CAF',
        'Central America': 'CAM',
        'Central Asia': 'CAS',
        'Co-operative Republic of Guyana': 'GUY',
        'Commonwealth of Australia': 'AUS',
        'Democratic Republic of Sao Tome and Prin': 'STP',
        'Democratic Republic of the Congo': 'COD',
        'Democratic Socialist Republic of Sri Lan': 'LKA',
        'EU Accession Countries': 'EUA',
        'East Asia and Pacific': 'EAP',
        'Eastern Africa': 'EAF',
        'Europe and Central Asia': 'ECA',
        'Islamic  Republic of Afghanistan': 'AFG',
        'Kingdom of Swaziland': 'SWZ',
        'Latin America': 'LAT',
        'Macedonia; former Yugoslav Republic of;Macedonia; former Yugoslav Republic of': 'MKD',
        'Mekong': 'MEK',
        'Mercosur': 'MCS',
        'Middle East and North Africa': 'MEA',
        'Multi-Regional': 'MRG',
        'Organization of Eastern Caribbean States': 'ECS',
        'Oriental Republic of Uruguay': 'URY',
        'Pacific Islands': 'PCI',
        'Red Sea and Gulf of Aden': 'RED',
        'Republic of Congo': 'COG',
        "Republic of Cote d'Ivoire": 'CIV',
        'Republic of Korea': 'KOR',
        'Republic of Kosovo': 'KOS',
        'Republic of Niger': 'NER',
        'Republic of Rwanda': 'RWA',
        'Republic of Togo': 'TGO',
        'Republic of Turkey': 'TUR',
        'Republic of the Union of Myanmar': 'MMR',
        'Republica Bolivariana de Venezuela': 'VEN',
        'Sint Maarten': 'SXM',
        'Socialist Federal Republic of Yugoslavia': 'YUG',
        "Socialist People's Libyan Arab Jamahiriy": 'LBY',
        'Socialist Republic of Vietnam': 'VNM',
        'Somali Democratic Republic': 'SOM',
        'South Asia': 'SAS',
        'Southern Africa': 'SAF',
        'St. Kitts and Nevis': 'KNA',
        'St. Lucia': 'LCA',
        'St. Vincent and the Grenadines': 'VCT',
        'State of Eritrea': 'ERI',
        'Taiwan; China;Taiwan; China': 'TWN',
        'The Independent State of Papua New Guine': 'PNG',
        'West Bank and Gaza': 'WBG',
        'Western Africa': 'WAF',
        'Western Balkans': 'WBL',
        'World': 'WLD'
        }

    
    # Dictionary to store country code mappings
    project_country_abbrev_dict = defaultdict(str)
    country_not_found = []  # Stores countries not found in the pycountry library
    
    # Iterate through the country names in the DataFrame
    for country in df['countryname'].drop_duplicates().sort_values():
        try:
            # Look up the country name in the pycountry library
            # Store the country name as the dictionary key and the ISO-3 code as the value
            project_country_abbrev_dict[country] = countries.lookup(country).alpha_3
        except:
            # If the country name is not found in the pycountry library, print it out and store it in the country_not_found list
            print(country, ' not found')
            country_not_found.append(country)
    
    # Update the dictionary with manually defined country codes
    project_country_abbrev_dict.update(country_codes)
    
    # Map country codes to country names in the DataFrame
    df['country_code'] = df['countryname'].apply(lambda x: project_country_abbrev_dict[x])
    return df

def process_data():
    """
    Load data from various sources including CSV files, JSON files, SQLite database, XML file, and external API.

    Returns:
        dict: A dictionary containing DataFrames loaded from different sources.

    Notes:
        - CSV files are loaded using Pandas `read_csv` function.
        - JSON files are loaded using Pandas `read_json` function.
        - XML data is parsed using BeautifulSoup and loaded into a DataFrame.
        - SQLite database is queried using Pandas `read_sql` function.
        - Data from an external API (World Bank API) is fetched using the requests library and loaded into a DataFrame.

    Example:
        >>> data = load_data()
        >>> projects_df = data['projects_df']
        >>> population_df = data['population_df']
        >>> # Access other DataFrames similarly
    """

    # Load XML data
    def load_xml_data():
        with open('/opt/airflow/data/population_data.xml', 'r') as f:
            xml_data = f.read()

        soup = BeautifulSoup(xml_data, 'lxml')
        data = {'country_or_area': [], 'item': [], 'year': [], 'value': []}

        records = soup.find_all('record')

        for record in records:
            country_or_area = record.find('field', {'name': 'Country or Area'})
            item = record.find('field', {'name': 'Item'})
            year = record.find('field', {'name': 'Year'})
            value = record.find('field', {'name': 'Value'})

            if country_or_area and item and year and value:
                data['country_or_area'].append(country_or_area.text)
                data['item'].append(item.text)
                data['year'].append(year.text)
                data['value'].append(value.text)

        return pd.DataFrame(data)

    # Load data from different sources
    projects_df = pd.read_csv("/opt/airflow/data/projects_data.csv", dtype='str')
    population_df = pd.read_csv("/opt/airflow/data/population_data.csv", skiprows=4)
    population_json_df = pd.read_json('/opt/airflow/data/population_data.json', orient='records')
    population_xml_df = load_xml_data()
    conn = sqlite3.connect('/opt/airflow/data/population_data.db')
    population_sql_df = pd.read_sql('SELECT * FROM population_data', conn)
    rural_df = pd.read_csv('/opt/airflow/data/rural_population_percent.csv', skiprows=4)
    electricity_df = pd.read_csv('/opt/airflow/data/electricity_access_percent.csv', skiprows=4)
    gdp_df = pd.read_csv('/opt/airflow/data/gdp_data.csv', skiprows=4)
    mystery_df = pd.read_csv('/opt/airflow/data/mystery.csv', encoding='utf-16')



    # Fetching data from API
    dfs = []
    country_codes = ['ind'] #population_df['Country Code'].unique()

    for country_code in country_codes:
        api_url = f"https://api.worldbank.org/v2/countries/{country_code}/indicators/SP.POP.TOTL/?format=json"
        response = requests.get(api_url)

        if response.status_code == 200:
            json_data = response.json()
            data = json_data[1]
            df = pd.DataFrame(data)
            dfs.append(df)
        else:
            print(f"Failed to fetch data for country code: {country_code}")

    population_api_df = pd.concat(dfs, ignore_index=True)

    # Data loading
    population_api_df['indicator_id'] = population_api_df['indicator'].apply(lambda x: x['id'])
    population_api_df['indicator_name'] = population_api_df['indicator'].apply(lambda x: x['value'])
    population_api_df['country_code'] = population_api_df['countryiso3code']
    population_api_df['year'] = population_api_df['date']
    population_api_df['country_name'] = population_api_df['country'].apply(lambda x: x['value'])
    population_api_df.drop(['indicator', 'country'], axis=1, inplace=True)
    population_api_df = population_api_df[['country_name', 'country_code', 'indicator_id', 'indicator_name', 'year', 'value']]
 
    # Combining data
    main_df = pd.concat([rural_df, electricity_df])
 
    # Cleaning Data
    # Copy original DataFrames to keep them intact
    projects_cleaned_df = projects_df.copy()
    population_cleaned_df = population_df.copy()
    population_json_cleaned_df = population_json_df.copy()
    population_xml_cleaned_df = population_xml_df.copy()
    population_sql_cleaned_df = population_sql_df.copy()
    main_cleaned_df = main_df.copy()
    mystery_cleaned_df = mystery_df.copy()
    gdp_cleaned_df = gdp_df.copy()
    rural_cleaned_df = rural_df.copy()
    electricity_cleaned_df = electricity_df.copy()

    # Drop unnecessary columns
    rural_cleaned_df.drop(['Unnamed: 62'], axis=1, inplace=True)
    electricity_cleaned_df.drop(['Unnamed: 62'], axis=1, inplace=True)
    projects_cleaned_df.drop(['Unnamed: 56'], axis=1, inplace=True)
    population_sql_cleaned_df.drop(['index'], axis=1, inplace=True)
    population_cleaned_df.drop(['Unnamed: 62'], axis=1, inplace=True)
    mystery_cleaned_df.drop(['Unnamed: 0'], axis=1, inplace=True)
    gdp_cleaned_df.drop(['Unnamed: 62'], axis=1, inplace=True)
    main_cleaned_df.drop(['Unnamed: 62'], axis=1, inplace=True)

    # Define non-country values
    non_countries = [
        'World', 'High income', 'OECD members', 'Post-demographic dividend', 'IDA & IBRD total', 'Low & middle income',
        'Middle income', 'IBRD only', 'East Asia & Pacific', 'Europe & Central Asia', 'North America',
        'Upper middle income', 'Late-demographic dividend', 'European Union',
        'East Asia & Pacific (excluding high income)', 'East Asia & Pacific (IDA & IBRD countries)', 'Euro area',
        'Early-demographic dividend', 'Lower middle income', 'Latin America & Caribbean',
        'Latin America & the Caribbean (IDA & IBRD countries)', 'Latin America & Caribbean (excluding high income)',
        'Europe & Central Asia (IDA & IBRD countries)', 'Middle East & North Africa',
        'Europe & Central Asia (excluding high income)', 'South Asia (IDA & IBRD)', 'South Asia', 'Arab World',
        'IDA total', 'Sub-Saharan Africa', 'Sub-Saharan Africa (IDA & IBRD countries)',
        'Sub-Saharan Africa (excluding high income)', 'Middle East & North Africa (excluding high income)',
        'Middle East & North Africa (IDA & IBRD countries)', 'Central Europe and the Baltics',
        'Pre-demographic dividend', 'IDA only', 'Least developed countries: UN classification', 'IDA blend',
        'Fragile and conflict affected situations', 'Heavily indebted poor countries (HIPC)', 'Low income', 'Small states',
        'Other small states', 'Not classified', 'Caribbean small states', 'Pacific island small states'
    ]

    # Clean and filter DataFrames
    projects_cleaned_df = clean_column_names(projects_cleaned_df, non_countries)
    population_cleaned_df = clean_column_names(population_cleaned_df, non_countries)
    population_json_cleaned_df = clean_column_names(population_json_cleaned_df, non_countries)
    population_xml_cleaned_df = clean_column_names(population_xml_cleaned_df, non_countries)
    population_sql_cleaned_df = clean_column_names(population_sql_cleaned_df, non_countries)
    main_cleaned_df = clean_column_names(main_cleaned_df, non_countries)
    mystery_cleaned_df = clean_column_names(mystery_cleaned_df, non_countries)
    gdp_cleaned_df = clean_column_names(gdp_cleaned_df, non_countries)
    rural_cleaned_df = clean_column_names(rural_cleaned_df, non_countries)
    electricity_cleaned_df = clean_column_names(electricity_cleaned_df, non_countries)

        

    # Convert columns to datetime
    convert_to_datetime(projects_cleaned_df, ['boardapprovaldate', 'closingdate'])

    # Remove commas and convert to float
    remove_commas_and_convert_to_float(projects_cleaned_df, ['lendprojectcost', 'ibrdcommamt', 'idacommamt', 'totalamt', 'grantamt'])

    # Apply lambda function to countryname column
    projects_cleaned_df['countryname'] = projects_cleaned_df['countryname'].apply(lambda x: x.split(';')[0].strip() if (';' in x) and (x.split(';')[0].strip() == x.split(';')[1].strip()) else x)

    # Clean sector and theme columns
    clean_string_columns(projects_cleaned_df, ['sector1', 'sector2', 'sector3', 'sector4', 'sector5', 'theme1', 'theme2', 'theme3', 'theme4', 'theme5'])
    
    # Concatenate values of sector and theme columns
    projects_cleaned_df['sector'] = projects_cleaned_df[['sector1', 'sector2', 'sector3', 'sector4', 'sector5']].apply(lambda row: ';'.join(row.dropna()), axis=1)
    projects_cleaned_df['theme'] = projects_cleaned_df[['theme1', 'theme2', 'theme3', 'theme4', 'theme5']].apply(lambda row: ';'.join(row.dropna()), axis=1)

    # Clean location column
    projects_cleaned_df['country'] = projects_cleaned_df['location'].apply(lambda x: ';'.join([loc.split('!$!')[-1] for loc in x.split(';')]) if isinstance(x, str) else np.NaN)

    # Apply regex replacement to string columns
    projects_cleaned_df.replace({'^(\(Historic\))': ''}, regex=True, inplace=True)

    # Drop columns with all NaN values
    projects_cleaned_df.dropna(axis=1, how='all', inplace=True)

    # Replace NaN values with 0 in numeric columns
    replace_nan_with_zero(projects_cleaned_df, projects_cleaned_df.select_dtypes(include=['number']).columns)
    replace_nan_with_zero(mystery_cleaned_df, mystery_cleaned_df.select_dtypes(include=['number']).columns)

    # Fill missing values in GDP and population DataFrames using backward fill followed by forward fill
    columns_to_fill = gdp_cleaned_df.columns[4:62]  # Assuming columns "1960" to "2017"
    columns_to_fill_2 = electricity_cleaned_df.columns[4:62]  # Assuming columns "1960" to "2017"
    gdp_cleaned_df[columns_to_fill] = gdp_cleaned_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    population_cleaned_df[columns_to_fill] = population_cleaned_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    population_json_cleaned_df[columns_to_fill] = population_json_cleaned_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    population_sql_cleaned_df[columns_to_fill] = population_sql_cleaned_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    rural_cleaned_df[columns_to_fill] = rural_cleaned_df[columns_to_fill].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)
    electricity_cleaned_df[columns_to_fill_2] = electricity_cleaned_df[columns_to_fill_2].fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)

    # Mean Imputation for 'boardapprovaldate' and 'closingdate' columns in projects_cleaned_df
    boardapprovaldate_mean = projects_cleaned_df['boardapprovaldate'].mean()
    closingdate_mean = projects_cleaned_df['closingdate'].mean()
    projects_cleaned_df['boardapprovaldate'].fillna(boardapprovaldate_mean, inplace=True)
    projects_cleaned_df['closingdate'].fillna(closingdate_mean, inplace=True)

    # # Remove non-country values from the data
    population_cleaned_df = population_cleaned_df[~population_cleaned_df['country_name'].isin(non_countries)]
    population_json_cleaned_df = population_json_cleaned_df[~population_json_cleaned_df['country_name'].isin(non_countries)]
    population_sql_cleaned_df = population_sql_cleaned_df[~population_sql_cleaned_df['country_name'].isin(non_countries)]
    population_xml_cleaned_df = population_xml_cleaned_df[~population_xml_cleaned_df['country_or_area'].isin(non_countries)]

    # Mode Imputation for other categorical columns in projects_cleaned_df
    columns_to_fill = ['supplementprojectflg', 'countryname', 'prodline', 'lendinginstr', 'productlinetype', 'projectstatusdisplay', 'status']
    for column in columns_to_fill:
        most_frequent_value = projects_cleaned_df[column].mode()[0]
        projects_cleaned_df[column].fillna(most_frequent_value, inplace=True)
 

    population_cleaned_df = population_cleaned_df.dropna(subset=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
    population_json_cleaned_df = population_cleaned_df.dropna(subset=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
    population_sql_cleaned_df = population_cleaned_df.dropna(subset=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
    gdp_cleaned_df = gdp_cleaned_df.dropna(subset=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
    rural_cleaned_df = rural_cleaned_df.dropna(subset=['1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968', '1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977', '1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986', '1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])
    electricity_cleaned_df = electricity_cleaned_df.dropna(subset=['1990', '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017'])

    population_xml_cleaned_df['year'] = population_xml_cleaned_df['year'].astype('int')
    population_xml_cleaned_df['value'] = population_xml_cleaned_df['value'].replace('', np.nan)
    population_xml_cleaned_df['value'] = population_xml_cleaned_df.sort_values('year').groupby('country_or_area')['value'].fillna(method='ffill').fillna(method='bfill').astype(float)

    population_cleaned_df = population_cleaned_df.drop(columns=['indicator_name', 'indicator_code'])
    population_json_cleaned_df = population_json_cleaned_df.drop(columns=['indicator_name', 'indicator_code'])
    population_sql_cleaned_df = population_sql_cleaned_df.drop(columns=['indicator_name', 'indicator_code'])
    population_xml_cleaned_df = population_xml_cleaned_df.drop(columns=['item'])
    gdp_cleaned_df = gdp_cleaned_df.drop(columns=['indicator_name', 'indicator_code'])
    rural_cleaned_df = rural_cleaned_df.drop(columns=['indicator_name', 'indicator_code'])
    electricity_cleaned_df = electricity_cleaned_df.drop(columns=['indicator_name', 'indicator_code'])

    # Melt the DataFrame to stack the '1960' to '2017' columns into a single 'year' column
    population_cleaned_df = population_cleaned_df.melt(id_vars=['country_name', 'country_code'], var_name='year', value_name='value')# Melt the DataFrame to stack the '1960' to '2017' columns into a single 'year' column
    population_json_cleaned_df = population_json_cleaned_df.melt(id_vars=['country_name', 'country_code'], var_name='year', value_name='value')# Melt the DataFrame to stack the '1960' to '2017' columns into a single 'year' column
    population_sql_cleaned_df = population_sql_cleaned_df.melt(id_vars=['country_name', 'country_code'], var_name='year', value_name='value')
    gdp_cleaned_df = gdp_cleaned_df.melt(id_vars=['country_name', 'country_code'], var_name='year', value_name='gdp')
    rural_cleaned_df = rural_cleaned_df.melt(id_vars=['country_name', 'country_code'], var_name='year', value_name='ruralpopulationpercent')
    electricity_cleaned_df = electricity_cleaned_df.melt(id_vars=['country_name', 'country_code'], var_name='year', value_name='electricityaccesspercent')

    gdp_cleaned_df['year'] = gdp_cleaned_df['year'].astype('int')
    population_cleaned_df['year'] = population_cleaned_df['year'].astype('int')
    population_json_cleaned_df['year'] = population_json_cleaned_df['year'].astype('int')
    population_sql_cleaned_df['year'] = population_sql_cleaned_df['year'].astype('int')
    population_xml_cleaned_df['year'] = population_xml_cleaned_df['year'].astype('int')
    rural_cleaned_df['year'] = rural_cleaned_df['year'].astype('int')
    electricity_cleaned_df['year'] = electricity_cleaned_df['year'].astype('int')
  
    # Reset the index
    population_cleaned_df = population_cleaned_df.reset_index(drop=True)
    population_json_cleaned_df = population_json_cleaned_df.reset_index(drop=True)
    population_sql_cleaned_df = population_sql_cleaned_df.reset_index(drop=True)
    gdp_cleaned_df = gdp_cleaned_df.reset_index(drop=True)
    rural_cleaned_df = rural_cleaned_df.reset_index(drop=True)
    electricity_cleaned_df = electricity_cleaned_df.reset_index(drop=True)

    projects_cleaned_df = projects_cleaned_df.drop_duplicates()
    population_cleaned_df = population_cleaned_df.drop_duplicates()
    population_json_cleaned_df = population_json_cleaned_df.drop_duplicates()
    population_sql_cleaned_df = population_sql_cleaned_df.drop_duplicates()
    population_xml_cleaned_df = population_xml_cleaned_df.drop_duplicates()
    gdp_cleaned_df = gdp_cleaned_df.drop_duplicates()

    projects_df_capped = projects_cleaned_df.copy()
    population_sql_df_capped = population_sql_cleaned_df.copy()
    population_df_capped = population_cleaned_df.copy()
    population_json_df_capped = population_json_cleaned_df.copy()
    population_xml_df_capped = population_xml_cleaned_df.copy()
    gdp_df_capped = gdp_cleaned_df.copy()

    # Apply Winsorizer to numeric columns in projects_df
    for column in ['lendprojectcost', 'ibrdcommamt', 'idacommamt', 'totalamt']:
        winsoriser_iqr = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=[column])
        projects_df_capped[column] = winsoriser_iqr.fit_transform(projects_df_capped[[column]])

    # Apply Gaussian Winsorizer to grantamt column in projects_df
    winsoriser_gaussian = Winsorizer(capping_method='gaussian', tail='both', fold=2, variables=['grantamt'])
    projects_df_capped['grantamt'] = winsoriser_gaussian.fit_transform(projects_df_capped[['grantamt']])

    # Apply Winsorizer to numeric columns in population DataFrames
    numeric_columns = population_df_capped.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        winsoriser_iqr = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=[column])
        population_df_capped[column] = winsoriser_iqr.fit_transform(population_df_capped[[column]])
        population_json_df_capped[column] = winsoriser_iqr.fit_transform(population_json_df_capped[[column]])
        population_sql_df_capped[column] = winsoriser_iqr.fit_transform(population_sql_df_capped[[column]]) 
        population_xml_df_capped[column] = winsoriser_iqr.fit_transform(population_xml_df_capped[[column]])

    # Apply Winsorizer to gdp column in gdp_df
    winsoriser_iqr_gdp = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['gdp'])
    gdp_df_capped['gdp'] = winsoriser_iqr_gdp.fit_transform(gdp_df_capped[['gdp']])

    # Extract features from datetime columns in projects_df
    date_columns = ['boardapprovaldate', 'closingdate']
    for col in date_columns:
        projects_df_capped[col] = pd.to_datetime(projects_df_capped[col])
        projects_df_capped[col + '_year'] = projects_df_capped[col].dt.year
        projects_df_capped[col + '_month'] = projects_df_capped[col].dt.month
        projects_df_capped[col + '_day'] = projects_df_capped[col].dt.day
        projects_df_capped.drop([col], axis=1, inplace=True)

    # Map 'Y' to 1 and 'N' to 0 in projects_df
    projects_df_capped['supplementprojectflg'] = projects_df_capped['supplementprojectflg'].map({'Y': 1, 'N': 0})

    # Encode categorical columns
    projects_df_capped = encode_categorical_columns(projects_df_capped)
    population_df_capped = encode_categorical_columns(population_df_capped)
    population_sql_df_capped = encode_categorical_columns(population_sql_df_capped)
    population_xml_df_capped = encode_categorical_columns(population_xml_df_capped)
    population_json_df_capped = encode_categorical_columns(population_json_df_capped)
    gdp_df_capped = encode_categorical_columns(gdp_df_capped)

    # Define numerical columns for scaling
    num_columns_projects = ['supplementprojectflg', 'lendprojectcost', 'ibrdcommamt', 'idacommamt', 'totalamt', 'grantamt', 'boardapprovaldate_year', 'boardapprovaldate_month', 'boardapprovaldate_day', 'closingdate_year', 'closingdate_month', 'closingdate_day']
    num_columns_population = ['year', 'value']

    # Scale numerical columns using StandardScaler
    scaler = StandardScaler()
    projects_df_capped[num_columns_projects] = scaler.fit_transform(projects_df_capped[num_columns_projects]) 
    population_df_capped[num_columns_population] = scaler.fit_transform(population_df_capped[num_columns_population]) 
    population_sql_df_capped[num_columns_population] = scaler.fit_transform(population_sql_df_capped[num_columns_population]) 
    population_json_df_capped[num_columns_population] = scaler.fit_transform(population_json_df_capped[num_columns_population]) 
    population_xml_df_capped[num_columns_population] = scaler.fit_transform(population_xml_df_capped[num_columns_population]) 

    projects_cleaned_df = add_country_codes(projects_cleaned_df)
    # Calculate project cost
    projects_cleaned_df['project_cost'] = projects_cleaned_df['lendprojectcost'] + projects_cleaned_df['ibrdcommamt'] + projects_cleaned_df['idacommamt'] + projects_cleaned_df['totalamt'] + projects_cleaned_df['grantamt']
    
    # Group by 'country_code' and sum the project costs
    country_project_cost = projects_cleaned_df.groupby('country_code')['project_cost'].sum().reset_index()
    
    # Merge population_cleaned_df and gdp_cleaned_df based on 'country_name', 'country_code', and 'year'
    merged_cleaned_df = pd.merge(population_cleaned_df, gdp_cleaned_df, on=['country_name', 'country_code', 'year'], how='left', suffixes=('_pop', '_gdp'))
    merged_cleaned_df = merged_cleaned_df.dropna()
    
    # Calculate GDP per capita and round it to 2 decimal places
    merged_cleaned_df['gdp_per_capita'] = (merged_cleaned_df['gdp'] / merged_cleaned_df['value']).round(2)
    
    # Merge with country_project_cost using 'country_code' as the key
    merged_cleaned_df = pd.merge(merged_cleaned_df, country_project_cost, on='country_code', how='left')
    
    # Join with rural_cleaned_df based on 'country_code' and 'year'
    merged_cleaned_df = pd.merge(merged_cleaned_df, rural_cleaned_df[['country_code', 'year', 'ruralpopulationpercent']], on=['country_code', 'year'], how='left')
    
    # Join with electricity_cleaned_df based on 'country_code' and 'year'
    merged_cleaned_df = pd.merge(merged_cleaned_df, electricity_cleaned_df[['country_code', 'year', 'electricityaccesspercent']], on=['country_code', 'year'], how='left')
    
    # Fill NaN values in specific columns with 0
    merged_cleaned_df['project_cost'].fillna(0, inplace=True)
    merged_cleaned_df['ruralpopulationpercent'].fillna(0, inplace=True)
    merged_cleaned_df['electricityaccesspercent'].fillna(0, inplace=True)
    
    # Rename columns and add underscores
    merged_cleaned_df.rename(columns={
        'value': 'population',
        'ruralpopulationpercent': 'rural_population_percent',
        'electricityaccesspercent': 'electricity_access_percent'
    }, inplace=True)
    
    data = {
        'projects_cleaned_df': projects_cleaned_df,
        'population_cleaned_df': population_cleaned_df,
        'population_json_cleaned_df': population_json_cleaned_df,
        'population_xml_cleaned_df': population_xml_cleaned_df,
        'population_sql_cleaned_df': population_sql_cleaned_df,
        'mystery_cleaned_df': mystery_cleaned_df,
        'rural_cleaned_df': rural_cleaned_df,
        'electricity_cleaned_df': electricity_cleaned_df,
        'merged_cleaned_df': merged_cleaned_df,
        'gdp_cleaned_df': gdp_cleaned_df
            }
    
    # Define file paths
    file_paths = {
        'projects_cleaned_df': 'projects_cleaned.csv',
        'population_cleaned_df': 'population_cleaned.csv',
        'population_json_cleaned_df': 'population_json_cleaned.csv',
        'population_xml_cleaned_df': 'population_xml_cleaned.csv',
        'population_sql_cleaned_df': 'population_sql_cleaned.csv',
        'mystery_cleaned_df': 'mystery_cleaned.csv',
        'rural_cleaned_df': 'rural_cleaned.csv',
        'electricity_cleaned_df': 'electricity_cleaned.csv',
        'merged_cleaned_df': 'merged_cleaned.csv',
        'gdp_cleaned_df': 'gdp_cleaned.csv'
    }

    # Save each cleaned DataFrame to a CSV file
    for key, value in file_paths.items(): 
        file_path = '/opt/airflow/data/csv_result/' + value
        data[key].to_csv(file_path, index=False)

# Define the function to load DataFrame into SQLite
def load_to_sqlite():
    merged_cleaned_df = pd.read_csv('/opt/airflow/data/csv_result/merged_cleaned.csv')
    # Connect to SQLite database (creates a new database if it doesn't exist)
    conn = sqlite3.connect('/opt/airflow/data/db_result/merged_data.db')
    
    # Load merged_cleaned_df into the SQLite database
    merged_cleaned_df.to_sql('merged_data', conn, if_exists='replace', index=False)
    
    # Commit changes and close the connection
    conn.commit()
    conn.close()
 

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 5, 13),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}

dag = DAG(
    'data_processing_dag',
    default_args=default_args,
    description='A DAG to process data',
    schedule_interval=None,
)


# Specify your Google Cloud Storage bucket name and the source CSV file path 
source_object = '/opt/airflow/data/csv_result/merged_cleaned.csv'

# Specify the destination bucket and object name in GCS
destination_bucket = 'kmzway_dns_storage'
destination_object = 'merged_cleaned.csv'

# Specify the Google Cloud Storage connection ID
google_cloud_storage_conn_id = 'google_cloud_default'

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=process_data,
    dag=dag,
    provide_context=True
)

# Define the PythonOperator to execute the load_to_sqlite function
load_to_sqlite_task = PythonOperator(
    task_id='load_to_sqlite_task',
    python_callable=load_to_sqlite,
    dag=dag,
)
 
upload_gcs_task = LocalFilesystemToGCSOperator(
        task_id='upload_to_gcs',
        src=source_object,
        dst=destination_object,
        bucket=destination_bucket, 
        dag=dag,
    )

load_data_task >> load_to_sqlite_task >> upload_gcs_task

