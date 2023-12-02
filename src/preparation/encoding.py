from numpy import ndarray
from pandas import Series, read_csv, DataFrame, concat
from matplotlib.pyplot import savefig, show, figure, tight_layout, subplots 
from dslabs_functions import get_variable_types, plot_bar_chart, HEIGHT
from matplotlib.figure import Figure 
from sklearn.impute import SimpleImputer
import csv  

us_areas_regions = {
'Alabama': 'South', 'Alaska': 'West',
'Arizona': 'West',
'Arkansas': 'South',
'California': 'West',
'Colorado': 'West',
'Connecticut': 'Northeast',
'Delaware': 'South',
'District of Columbia': 'South',  # Adding DC to the South region
'Florida': 'South',
'Georgia': 'South',
'Guam': 'West',  # Adding Guam to the West region
'Hawaii': 'West',
'Idaho': 'West',
'Illinois': 'Midwest',
'Indiana': 'Midwest',
'Iowa': 'Midwest',
'Kansas': 'Midwest',
'Kentucky': 'South',
'Louisiana': 'South',
'Maine': 'Northeast',
'Maryland': 'South',
'Massachusetts': 'Northeast',
'Michigan': 'Midwest',
'Minnesota': 'Midwest',
'Mississippi': 'South',
'Missouri': 'Midwest',
'Montana': 'West',
'Nebraska': 'Midwest',
'Nevada': 'West',
'New Hampshire': 'Northeast',
'New Jersey': 'Northeast',
'New Mexico': 'West',
'New York': 'Northeast',
'Virgin Islands': 'South',
'North Carolina': 'South',
'North Dakota': 'Midwest', 'Ohio': 'Midwest', 'Oklahoma': 'South', 'Oregon': 'West', 'Pennsylvania': 'Northeast', 'Puerto Rico': 'South', 'Rhode Island': 'Northeast', 'South Carolina': 'South', 'South Dakota': 'Midwest', 'Tennessee': 'South', 'Texas': 'South', 'Utah': 'West', 'Vermont': 'Northeast', 'Virginia': 'South', 'Washington': 'West', 'West Virginia': 'South', 'Wisconsin': 'Midwest', 'Wyoming': 'West'
}

def replace_states(data: DataFrame, var: str) -> DataFrame:
    unique_values: list[str] = data[var].unique()
    data[var] = data[var].replace(us_areas_regions)
    unique_values: list[str] = data[var].unique()
    data[var] = data[var].replace({'Midwest': 0, 'Northeast': 1, 'South': 2, 'West': 3})
    return data

def replace_ghealt(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"Poor": 0, "Fair": 1, "Good": 2, "Very good": 3, "Excellent": 5})
    data[var] = data[var].fillna(30)
    return data

def replace_LastCheckupTime(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"Within past year (anytime less than 12 months ago)": 0, "Within past 2 years (1 year but less than 2 years ago)": 1, "Within past 5 years (2 years but less than 5 years ago)": 2, "5 or more years ago": 3})
    data[var] = data[var].fillna(30)
    return data

def replace_RemovedTeeth(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"None of them": 0, "1 to 5": 1, "6 or more, but not all": 2, "All": 3})
    data[var] = data[var].fillna(30)
    return data

def replace_Age(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"Age 18 to 24": 0, "Age 25 to 29": 1, "Age 30 to 34": 2, "Age 35 to 39": 3, "Age 40 to 44": 4, "Age 45 to 49": 5, "Age 50 to 54": 6, "Age 55 to 59": 7, "Age 60 to 64": 8, "Age 65 to 69": 9, "Age 70 to 74": 10, "Age 75 to 79": 11, "Age 80 or older": 12})
    data[var] = data[var].fillna(30)
    return data

def replace_Diabetes(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"Yes": 1, "No": 0, "No, pre-diabetes or borderline diabetes": 0, "Yes, but only during pregnancy (female)": 1})
    data[var] = data[var].fillna(30)
    return data
    
def replace_SmokerStatus(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"Former smoker": 1, "Current smoker - now smokes every day": 1, "Current smoker - now smokes some days": 1, "Never smoked": 0})
    data[var] = data[var].fillna(30)
    return data

def replace_ECigaretteUsage(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"Use them every day": 1, "Use them some days": 1, "Not at all (right now)": 1, "Never used e-cigarettes in my entire life": 0})
    data[var] = data[var].fillna(30)
    return data

def replace_RaceEthnicityCategory(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"White only, Non-Hispanic" : 0, "Black only, Non-Hispanic" : 1, "Multiracial, Non-Hispanic" : 1, "Other race only, Non-Hispanic" : 1, "Hispanic" : 1})
    data[var] = data[var].fillna(30)
    return data

def replace_TetanusLast10Tdap(data: DataFrame, var: str) -> DataFrame:
    data[var] = data[var].replace({"No, did not receive any tetanus shot in the past 10 years" : 0, "Yes, received Tdap" : 1, "Yes, received tetanus shot but not sure what type" : 1, "Yes, received tetanus shot, but not Tdap" : 1})
    data[var] = data[var].fillna(30)
    return data

def replace_binary(data: DataFrame) -> DataFrame:
    data.replace("No", 0, inplace=True)
    data.replace("Yes", 1, inplace=True)
    data.replace("Male", 0, inplace=True)
    data.replace("Female", 1, inplace=True)
    return data

def fill(data: DataFrame) -> DataFrame:
    lst_dfs = []
    variables: dict = get_variable_types(data)
    imp = SimpleImputer(strategy="most_frequent", copy=False)
    tmp_nr = DataFrame(imp.fit_transform(data[variables["numeric"]]), columns=variables["numeric"],)
    tmp_bi = DataFrame(imp.fit_transform(data[variables["binary"]]), columns=variables["binary"],)
    lst_dfs.append(tmp_nr)
    lst_dfs.append(tmp_bi)
    data = concat(lst_dfs, axis=1)
    return data

file_tag = "CovidPos"
data: DataFrame = read_csv("data/class_pos_covid.csv")
data = replace_states(data, "State")
data = replace_ghealt(data, "GeneralHealth")
data = replace_LastCheckupTime(data, "LastCheckupTime")
data = replace_RemovedTeeth(data, "RemovedTeeth")
data = replace_Age(data, "AgeCategory")
data = replace_SmokerStatus(data, "SmokerStatus")
data = replace_Diabetes(data, "HadDiabetes")
data = replace_ECigaretteUsage(data, "ECigaretteUsage")
data = replace_RaceEthnicityCategory(data, "RaceEthnicityCategory")
data = replace_TetanusLast10Tdap(data, "TetanusLast10Tdap")

data = replace_binary(data)

data = fill(data)



data.to_csv("newdata.csv")
"""
def remove_lines_with_missing_variables(input_file, output_file):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        for row in reader:
            missing_vars_count = sum(cell == '' or (isinstance(cell, str) and cell.strip() == '') for cell in row)
            row.append(missing_vars_count)
            writer.writerow(row)

input_csv_file = 'class_pos_covid.csv/class_pos_covid.csv'
output_csv_file = 'variables_missing.csv'
remove_lines_with_missing_variables(input_csv_file, output_csv_file)

'State', 'GeneralHealth', 'LastCheckupTime', 'RemovedTeeth', 'HadDiabetes', 'SmokerStatus', 'ECigaretteUsage', 'RaceEthnicityCategory', 'AgeCategory', 'TetanusLast10Tdap'
"""

