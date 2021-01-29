import pandas as pd
import sys


def read_data(file, converters=None, date_parser=None):

    # Force RegionName (zip) to 5 characters with zero pad
    df = pd.read_csv(file, converters=converters, date_parser=date_parser)
    print(f"Data types for '{file}'")
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

    df_transposed['year_month_dt'] =  pd.to_datetime(df_transposed['year_month'])
    df_transposed['median_value'] = df_transposed['median_value'].astype(float)
    df_transposed['year_month'] = df_transposed['year_month_dt'].dt.strftime('%Y-%m')
    df_transposed = df_transposed.drop('year_month_dt', axis=1)

    return df_transposed


def join_holidays(df1, df_holiday):
    df_holiday['date'] =  pd.to_datetime(df_holiday.date)
    df_holiday['year_month'] = df_holiday['date'].dt.strftime('%Y-%m')

    df_holiday_grouped = df_holiday.groupby('year_month', as_index=False)['holiday_name'].count()\
        .rename(columns={'holiday_name':'cnt_holidays'}).reset_index(drop=True)


    #join holidays to price data by year_month
    df_with_feat = df1.merge(df_holiday_grouped, how='left', on='year_month')
    df_with_feat['cnt_holidays'] = df_with_feat['cnt_holidays'].fillna(0)
    print("Transposed Data with Holidays features:")
    print(df_with_feat.tail(5))

    return df_with_feat

 
def write(df, file):

    try:
        df.to_csv(file, index=False)
        print("Tranposed data written to CSV")
        token = "cheese"
    except:
        print("Error on CSV write") 
        token = "no cheese"   

    return token


df = read_data("Zip_zhvi_bdrmcnt_2_uc_sfrcondo_tier_0.33_0.67_sm_sa_mon.csv", converters={'RegionName': '{:0>5}'.format}, date_parser=False)
df_holiday = read_data("holiday.csv", converters=None, date_parser='date')
df_transposed = transpose(df=df, drop_fields=['RegionID','SizeRank','RegionType','StateName','Metro','CountyName'], key_fields=['zip', 'City','State'], cadence='year_month', series_level='zip')
df_with_feat = join_holidays(df1=df_transposed, df_holiday=df_holiday)
token = write(df=df_with_feat, file="2bdrm_by_zip_yearmonth.csv")