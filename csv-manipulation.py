import pandas as pd
import sklearn.datasets as datasets
import pandas_ml as pdml
import numpy as np
# READ DATASETS AND CLEAR NAN COLUMNS AND ROWS
search_data = pd.read_csv(
    "c:/Users/Arya Akkus/Desktop/Arya/comp551/2020_US_weekly_symptoms_dataset.csv", sep=',',
    header=0, engine='python')

search_data.dropna(axis="columns", how="all", inplace=True)
search_data.dropna(axis="rows", how="all", thresh=12, inplace=True)
search_data.dropna(axis="columns", how="all", inplace=True)
search_data['date'] = pd.to_datetime(search_data['date'])
search_data['week'] = search_data['date'].dt.isocalendar().week
search_data.sort_values(['open_covid_region_code', 'week'],
                        ascending=[True, True], inplace=True)
print(search_data.shape)

search_data.to_csv(
    'c:/Users/Arya Akkus/Desktop/Arya/comp551/cleared_search.csv', index=0)


agg_data = pd.read_csv(
    "c:/Users/Arya Akkus/Desktop/Arya/comp551/aggregated_cc_by.csv", sep=',',
    header=0, engine='python')
# agg_data.dropna(axis="columns", how="all", inplace=True)
# EXTRACT USA DATA FROM DATASET 2
indexNames = agg_data[agg_data['open_covid_region_code'].str.find(
    'US-') < 0].index
agg_data.drop(indexNames, inplace=True)


agg_data['date'] = pd.to_datetime(agg_data['date'])
indexNames2 = agg_data[agg_data['date'] < '2020-01-06'].index
agg_data.drop(indexNames2, inplace=True)
agg_data.dropna(axis="columns", how="all", inplace=True)

# # CONVERT DAILY DATA TO WEEKLY
agg_data['week'] = agg_data['date'].dt.isocalendar().week
agg_data.fillna(0, inplace=True)
print(agg_data.shape)


logic = {
    # 'open_covid_region_code': 'first',
    # 'region_name': 'first',
    # 'cases_cumulative': 'last',
    # 'cases_new': 'sum',
    # 'cases_cumulative_per_million': 'last',
    # 'cases_new_per_million': 'sum',
    # 'deaths_cumulative': 'last',
    # 'deaths_new': 'sum',
    # 'deaths_cumulative_per_million': 'last',
    # 'deaths_new_per_million': 'sum',
    # 'tests_new': 'sum',
    # 'tests_cumulative': 'last',
    # 'tests_cumulative_per_thousand': 'last',
    # 'tests_new_per_thousand': 'sum',
    # 'test_units': 'last',
    # 'hospitalized_current': 'mean',
    'hospitalized_new': 'sum',
    'hospitalized_cumulative': 'last',
    # 'discharged_new': 'sum',
    # 'discharged_cumulative': 'last',
    # 'icu_current': 'mean',
    # 'icu_cumulative': 'last',
    # 'ventilator_current': 'mean',
    # 'school_closing': 'max',
    # 'school_closing_flag': 'max',
    # 'workplace_closing': 'max',
    # 'workplace_closing_flag': 'max',
    # 'cancel_public_events_flag': 'max',
    # 'restrictions_on_gatherings': 'max',
    # 'restrictions_on_gatherings_flag': 'max',
    # 'close_public_transit': 'max',
    # 'close_public_transit_flag': 'max',
    # 'stay_at_home_requirements': 'max',
    # 'stay_at_home_requirements_flag': 'max',
    # 'restrictions_on_internal_movement': 'max',
    # 'restrictions_on_internal_movement_flag': 'max',
    # 'international_travel_controls': 'max',
    # 'income_support': 'max',
    # 'income_support_flag': 'max',
    # 'debt_contract_relief': 'max',
    # 'fiscal_measures': 'max',
    # 'international_support': 'max',
    # 'public_information_campaigns': 'max',
    # 'public_information_campaigns_flag': 'max',
    # 'testing_policy': 'max',
    # 'contact_tracing': 'max',
    # 'emergency_investment_in_healthcare': 'max',
    # 'investment_in_vaccines': 'max',
    # 'wildcard': 'max',
    # 'confirmed_cases': 'last',
    # 'confirmed_deaths': 'last',
    # 'stringency_index': 'max',
    # 'stringency_index_for_display': 'max',
    # 'stringency_legacy_index': 'max',
    # 'stringency_legacy_index_for_display': 'max',
    # 'government_response_index': 'max',
    # 'government_response_index_for_display': 'max',
    # 'containment_health_index': 'max',
    # 'containment_health_index_for_display': 'max',
    # 'economic_support_index': 'max',
    # 'economic_support_index_for_display': 'max'
}

df1 = agg_data.groupby(
    ['open_covid_region_code', 'week'], as_index=False).agg(logic)
print(df1.shape)
df1.to_csv('c:/Users/Arya Akkus/Desktop/Arya/comp551/cleared_agg.csv')
df2 = pd.merge(left=search_data, right=df1,
               on=['open_covid_region_code', 'week'])
df2.to_csv('c:/Users/Arya Akkus/Desktop/Arya/comp551/merged_data.csv', index=0)
print(df2.shape)

# SET TARGET AND NORMALIZE DATA


# dataframe = pdml.ModelFrame(df2.to_dict(orient='list'))
