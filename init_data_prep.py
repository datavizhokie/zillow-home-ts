import pandas as pd


def read_data(file):

    # Force RegionName (zip) to 5 characters with zero pad
    df = pd.read_csv(file, converters={'RegionName': '{:0>5}'.format})
    print(df.dtypes)

    return df


def transpose(df, drop_fields, key_fields, cadence, series_level):

    df_slim = df.drop(drop_fields, axis=1).rename(columns={'RegionName':'zip'})

    print("Slim Data columns before transpose")
    print(df_slim.columns)

    df_transposed = df_slim.set_index(series_level).stack().reset_index(level=1, drop=True).rename('median_value').reset_index()

    df_transposed = df_slim.melt(id_vars = key_fields, 
            var_name = cadence, 
            value_name = "median_value")

    df_transposed['year_month'] =  pd.to_datetime(df_transposed['year_month'])
    df_transposed['median_value'] = df_transposed['median_value'].astype(float)

    print("Transposed data:")
    print(df_transposed.head())
    print(df_transposed.dtypes)

    return df_transposed

 
def write(df, file):

    try:
        df_transposed.to_csv(file, index=False)
        print("Tranposed data written to CSV")
        token = "cheese"
    except:
        print("Error on CSV write") 
        token = "no cheese"   

    return token



df = read_data(file='Zip_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv')
df_transposed = transpose(df=df, drop_fields=['RegionID','SizeRank','RegionType','StateName','Metro','CountyName'], key_fields=['zip', 'City','State'], cadence='year_month', series_level='zip')
token = write(df=df_transposed, file="2bdrm_by_zip_and_yearmonth_median_values.csv")