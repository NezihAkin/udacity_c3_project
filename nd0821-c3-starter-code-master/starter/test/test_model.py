importpytest
importpandasaspd
importos

@pytest.fixture(scope='session')
defdata():
current_dir=os.getcwd()
data_path=os.path.join(current_dir,'../data/census.csv')
df=pd.read_csv(data_path)
returndf

deftest_columns_and_types(data):
col_list={
'age',
'workclass',
'fnlgt',
'education',
'education-num',
'marital-status',
'occupation',
'relationship',
'race',
'sex',
'capital-gain',
'capital-loss',
'hours-per-week',
'native-country',
'salary'
}

#Checkcolumnpresence
