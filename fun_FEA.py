####################################################################################################################################
# Project     : Reusable Code for EDA and Feature Engineering 
#
# Coding      : Kelly Tay
#
# Date        : Since 2019-11-13
#
# Code Reviewed - 27/12/2019
# Code updated - 7/1/2020
#
# Description : General functions for feature engineering
#               01) (function) gather - wide to long dataframe 
#               02) (class - outliers) normalise - normalise data
#               03) (class - outliers) log_transform - transform data 
#               04) (class - outliers) standardized - transform data 
#               05) (class - outliers) label_outliers - labels datapoints that are outliers (using std) 
#               06) (class - outliers) outliers_study 
#               07) (class - dt_calculations) findMonth - creates a month label based on date 
#               08) (class - dt_calculations) findWeekending - returns  weekending dates
#               09) (class - dt_calculations) findDay - returns the day of the week (sunday)
#               10) (class - dt_calculations) time_duration - calculate the datetime difference of two dates 
#               11) (class - dt_calculations) countweekdays - calculate the number of weekdays between two dates
#               12) (class - dt_calculations) CalculateDuration - given a list of dates, calculate the average duration
#               13) (class - dt_calculations) fun_cal_duration - calculates the average time duration per group
#               14) (class - correlation) corr - calculates the correlation between variables of two tables 
#               15) (function) cat_nominal_stats - for feature engineering 
#               16) (class - deviations) no_deviations - calculates no of standard deviations away from the mean 
#               17) (class - deviations) deviation_study 
#               19) (class - FE_QMS_Stats) func_concat_cols - combines multiple columns to give a single columns 
#               20) (function - add_trend_feature) - returns the gradient of an array of numbers
#               21) (function - sparsity_features) - returns the percentage of zero values of a feature
#               22) (function - RFM) - returns the rfm customer profile: churn, freq_loyal and high_spend
#               23) (class - data_cleaning) clean_datatype - cleans the datatype of a dataframe using dictionary
#               24) (class - data_cleaning) clean_missing_values - remove columns or fill in missing values
#               25) (class - data_cleaning) clean_string - cleans string object 
###############################################################################################################################

print(' ++++ Import packages ----')
import sys
import pandas as pd
import numpy as np
import datetime
from datetime import date, timedelta
from dateutil import relativedelta 
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import operator
import re 
from sklearn.linear_model import LinearRegression

#######################################  Start of Functions #############################################

def gather(DF, key, value, var_name = None):
    """ function equivalent to R function - gather
    """
    if isinstance(DF, pd.DataFrame):
        id_vars =  key
        value_vars = [v for v in DF.columns if v not in [key]]
        var_name = var_name
        value_name = value
        gathered_df = pd.melt(DF, id_vars = key, value_vars= value_vars, var_name = var_name, value_name = value_name)
        return gathered_df
    else:
        raise Exception('Data passed in is not a Pandas Dataframe.')    
# End of function - gather


class outliers(object):
    @classmethod
    def normalise(cls,column):
        """ function normalises the data, converting all data points to decimals between 0 and 1 
            using min-max scaling 
            Input: 
                1. column. series 
            output:
                1. series (normalised data)

            How to use:
                > df.column_name.apply(normalise)
                > normalise(df.column_name)
        """
        upper = column.max()
        lower = column.min()
        y = (column - lower)/(upper-lower)
        return y
    
    @classmethod
    def log_transform(cls,column):
        """ function transform the data using log transform  
            When to use:
                > to make the data less skewed
            Input: 
                1. column. series 
            output:
                1. series (transformed data)

            How to use:
                > df.column_name.apply(normalise)
                > normalise(df.column_name)
        """
        # check if there are values = 0
        if column.any(0) ==  True:
            column =  column + 1 
        helpful_log = np.log(column)

        return helpful_log

    @classmethod
    def standardized(cls,column):
        """ function transform the data using standardisation 
            When to use:
                > to make the data less skewed
            Input: 
                1. column. series 
            output:
                1. series (transformed data)

            How to use:
                > df.column_name.apply(normalise)
                > normalise(df.column_name)
        """
        val_mean = np.mean(column)

        if len(column) == 1:
            return column
        else:
            std_mean = np.std(column)
            standardised = (column - val_mean)/std_mean
            return standardised
        # End of class function standardized

    @classmethod
    def label_outliers(cls, column, lower_upper_range = None):
        # code changed - code review by yuyu
        """ labels datapoints that are considered an outlier
            Input:
                1. column.series
                2. lower_upper_range. user specified lower and upper quatile
            Output:
                1. series
        """
        if lower_upper_range is None:
            lower_quantile, upper_quantile = np.percentile(column,[25,75])
        else:
            lower_quantile, upper_quantile = lower_upper_range
        
        iqr = upper_quantile-lower_quantile

        lower_whisker = lower_quatile - (1.5 * iqr)
        upper_whisker = upper_quatile + (1.5 * iqr)

        return [1 if v < lower_whisker or v > upper_whisker else 0 for v in column]
    
    @classmethod
    def outliers_study(cls, DF, grp_by_attr, nominal_attr, numeric_attr):
        """ for each grp_by_attr, function identifies numeric attributes that are outliers across nominal attributes. Output will be the Statistics (total number of outliers) for each grp_by_attr. 
            Input:
                1. DF. dataframe
                2. grp_by_attr. group by attribute (string)
                3. nominal_attr. Nominal attribute (list)
                4. numeric_attr. Numeric attribute (string)
            Output:
                > dataframe
        """
        # Subset dataframe to select only relevant columns
        df = DF.loc[:, [grp_by_attr] + nominal_attr + [numeric_attr]]
        
        df = df.groupby([grp_by_attr] + nominal_attr)[numeric_attr].agg(np.sum).reset_index()

        outlier_stats = []

        for i in list(set(df[grp_by_attr])):
            d = df[(df[grp_by_attr] == i)]
            # label outliers with function - label_outliers
            # change - use "assign" instead of assigning directly to create a new column
            d = d.assign(outliers=cls.label_outliers(d[numeric_attr]))
            out = d.groupby(grp_by_attr).outliers.agg([np.sum]).reset_index()
            outlier_stats.append(out)
        
        output = pd.concat(outlier_stats)
        output.fillna(0, inplace = True)
        # Rename the columns to include numeric attr
        output.columns = [grp_by_attr] + ['outliers'+ '__' + numeric_attr + '__' + v for v in output.columns[1:]]

        return output # frequency of outliers
# End of class - outliers


class dt_calculations(object):
    @classmethod
    def findMonth(cls, date):
        """ Function extract year and month from date 
        Input:
            1. date. date 
        Output:
            1. string
        """
        # check datetime:
        date = pd.to_datetime(date)
        month_year = date.strftime('%Y-%m')
        return month_year
    
    # Alternative code - modification: month_year=date.dt.to_period('M')

    @classmethod
    def findDay(cls, date, dt_format = None): 
        """ Function extract weekday from date 
            Input:
                1. date. date 
            Output:
                1. string
        """
        week_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday','Sunday']
        
        if dt_format == None:
            dt_format = '%Y-%m-%d'
        
        if type(date) != str:
        # if input is a datetime, convert to string
            date = date.strftime(dt_format)
        
        week_day = datetime.datetime.strptime(date, dt_format).weekday() 
    
        return (week_days[week_day]) 

    # since date is ald datetime type after layer1, week_day= date.dt.weekday_name
    
    @classmethod
    def findWeekending(cls, date, dt_format = None):
        # code changed - 27/12/2019
        """ Function extract weekending date from date 
            Input:
                1. date. date string
            Output:
                1. string
        """
        if dt_format == None:
            dt_format = '%Y-%m-%d'

        if isinstance(date, str):
            date = date.strftime(dt_format) # date convert to string
        
        weekending = date + pd.offsets.Week(weekday=6)
        week_labels = 'week' + '_' + weekending.strftime('%Y_%m_%d')
        
        return week_labels
    # end of function - findWeekending

    
    @classmethod 
    def time_duration(cls, d1, d2,dt_format=None):
        """ function calculate the time difference between d1 and d2
            Input:
                1. d1 - date string 
                2. d2 - date string 
            Output:
                1. number of days 
        """
        if dt_format is None:
            dt_format = '%Y-%m-%d'

        dt1 = datetime.datetime.strptime(d1, dt_format)
        dt2 = datetime.datetime.strptime(d2, dt_format)

        return np.abs((dt2-dt1).days)
    
    @classmethod
    def countweekdays(cls, d1, d2, dt_format = None):
        """ function to calculate the number of weekdays (exclude weekends)
            Input:
                1. d1 - date string
                2. d2 - date string 
                3. dt_format - date format as specified by user
            Output:
                > Number of weekdays 
        """
        if dt_format is None:
            dt_format = '%Y-%m-%d'
        
        diff = cls.time_duration(d1 = d1, d2 = d2, dt_format = dt_format)
        
        start = datetime.datetime.strptime(min([d1,d2]), dt_format)
        
        date_range = [(start + datetime.timedelta(days=x)).strftime(dt_format) for x in range(diff)]
        
        return len([x for x in date_range if datetime.datetime.strptime(x, dt_format).weekday() not in [5,6]])
    # End of countweekdays

    @classmethod
    def CalculateDuration(cls, date_list, week_days = False, dt_format = None):
        """ Function calculate the average duration for a list of dates
            e.g. [d1,d2,d3]
                > diff_1 = d2 - d1, diff_2 = d3 - d2
                > mean(diff_1, diff_2)
            Input:
                1. date_list. list of date 
                2. df_format - date format if date is of a special date format
                    Default date format - %Y-%m-%d
                3. week_days - to calculate weekdays or total number of days
            Output:
                1. numeric
        """

        date_list = sorted(date_list)
        
        if dt_format is None:
            dt_format = '%Y-%m-%d'
                
        if len(date_list) == 1:
            diff_days = 0 
            avg_days = 0

        else:
            diff_dates = [0]*(len(date_list)-1)

            for i in range(len(date_list)-1):
                if week_days == True:
                    diff_dates[i] = cls.countweekdays(t1 = date_list[i], t2 = date_list[i+1])
                else:
                    diff_dates[i] = cls.time_duration(t1 = date_list[i], t2 = date_list[i+1])
                # average days between product
            avg_days = round(sum(diff_dates)/len(diff_dates),2)

        return avg_days
        # end of function - CalculateDuration

    @classmethod
    def fun_cal_duration(cls, DF, grp_by_attr, date_attr, nominal_attr):

        df = DF.copy()
        
        df_short = df.loc[:, [grp_by_attr, date_attr] + nominal_attr]
            
        df_short[date_attr] = [str(v)[0:10] for v in df_short[date_attr]]
            
        # relabel the data by appending column name to values/categories
        for i in nominal_attr:
            df_short[i] = str(i) + '_' + df_short[i].astype(str)
            
        attr_list = [grp_by_attr] + nominal_attr
            
        t = df_short.groupby(attr_list)[date_attr].agg(lambda x: [v for v in x])
            
        tt = t.apply(self.CalculateDuration).reset_index()
            
        tt = tt.pivot_table(values = date_attr, index = grp_by_attr, columns =nominal_attr, fill_value = 0 ).reset_index()

        return tt
# end of class - dt_calculations
   
class correlation(object):

    @classmethod
    def corr_test(cls, dfx, dfy, test):

        if test == 'pearson':
            return pearsonr(dfx[x], dfy[y])
        elif test == 'spearman':
            return spearmanr(dfx[x], dfy[y])
        else:
            raise Exception('Invalid correlation test')


    @classmethod
    def corr(cls, dfx, dfy, cor_test = 'pearson', topn = None):
        """ function to calculate correlation of dataframes
            - Updated to include other correlations - pearson and spearman
            Input:
                1. dfx - dataframe
                2. dfy - dataframe 
                3. cor_test - 'pearson', 'spearman'
                4. topn.optional - select top n variables based on correlation score
            Output:
                dataframe - 3 columns (fe_x, fe_y, cor)

            Functionalities:
            1. One to One - correlation between dataframes with single variable
            2. One to Many - correlation between dataframes with single variable and multiple variables
            3. Many to Many - correlation between dataframes with multiple variables

            How to use:
            x1 = fe_x_df.iloc[:,0:1]
            y1 = fe_y_df.iloc[:,0:1]
            c1 = correlation.corr(dfx = x1, dfy = y1)

        """
        # Correlation for all needs
        fea_x = dfx.columns
        fea_y = dfy.columns

        rslt_all_needs = []

        for y in fea_y:
            # Compute correlation per needs
            rslt = []
            # Calculate correlation 
            for x in fea_x:   
                corr, _ = cls.corr_test(dfx[x], dfy[y], test = cor_test)
                rslt.append(corr)
            
            t = sorted(zip(fea_x, rslt), reverse=True)
            topn_df =  pd.DataFrame(list(t))
            topn_df.columns = ['fe_x',"cor"]
            topn_df['fe_y'] = y
            topn_df['cor_abs'] = np.abs(topn_df.cor)
            if topn is not None:
                # Create dataframe 
                topn_df = topn_df.nlargest(topn,'cor_abs')
                # relabel the correlation study table
                topn_df = topn_df.loc[:,['fe_y','fe_x','cor']]
            # Append to the main list
            rslt_all_needs.append(topn_df)
        
        # Combine them together
        corr_df = pd.concat(rslt_all_needs)

        return corr_df
    #End of function - corr
# End of class - correlation

def cat_nominal_stats( DF, grp_by_attr, nominal_attr, numerical_attr, stats = None):# need to add in cls
    """ function is used to derive the numerical statistics for a categorical variable
        Input:
            1. DF - Dataframe 
            2. grp_by_attr - group by attribute (string of column name)
            3. nominal_attr - categorical attribute (string of categorical attributes)
            4. numerical_attr - numeric attribute to populate the statistics
            5. stats(optional) -  the statistics to generate  
                Default - ['mean','max','min','median','std']
        Output:
            Dataframe 
        How to use:
            cat_nominal_stats(DF = data_trade_hist_win , grp_by_attr = 'lcin' , cat_attr = 'deal_curr' , nominal_attr = 'deal_amt_sgd', stats = ['mean','min','max','median','std'])
    """
    # Check for input datatype - make sure everything is a list 
    if type(grp_by_attr) is not list:
        grp_by_attr = [grp_by_attr]
    
    if type(nominal_attr) is not list:
        nominal_attr = [nominal_attr]
    
    if type(numerical_attr) is not list:
        numerical_attr = [numerical_attr]
    
    # subset dataframe 
    df = DF.loc[:,grp_by_attr + numerical_attr + nominal_attr]

    # Check is input, stats is specified by user
    if stats is None:
        stats = ['mean','max','min','median','std']

    df_pt = df.pivot_table(values = numerical_attr, index = grp_by_attr, columns = nominal_attr, aggfunc = stats, fill_value = 0)
    df_pt.reset_index(inplace=True)
    df_pt.columns = grp_by_attr + list(map("_".join,[x[::-1] for x in df_pt.columns][1:]))
    

    return df_pt
#end of Function 'cat_nominal_stats' 

class deviations(object):
    @classmethod
    def no_deviations(cls, n, avg, stds):
        """ calculate the number of standard deviations away from mean
            Input:
                1. n. number
                2. avg. mean (numeric)
                3. stds. standard deviation (numeric)
        """
        if stds == 0:
            return 0 
        else:
            return int(np.abs((n-avg)/stds))
    # end of class function - no_deviations

    @classmethod
    def deviation_study(cls, DF, grp_by_attr, numerical_attr, nominal_attr, stats = None):
        """ for each grp_by_attr, function counts the number of deviations numeric attributes across nominal attributes. Output will be the Statistics of the number of deviations for each grp_by_attr. 
            Input:
                1. DF. dataframe
                2. grp_by_attr. group by attribute (list)
                3. nominal_attr. Nominal attribute (list)
                4. numeric_attr. Numeric attribute (list)
                5. stats - list of statistics to compute for number of deviations (optional)
            Output:
                > dataframe
        """
        # Check for input datatype - make sure everything is a list 
        if type(grp_by_attr) is not list:
            grp_by_attr = [grp_by_attr]

        if type(nominal_attr) is not list:
            nominal_attr = [nominal_attr]

        if type(numerical_attr) is not list:
            numerical_attr = [numerical_attr]
        
        if stats is None:
            stats = ['mean','min','max','median']
        
        df = DF.loc[:,grp_by_attr + numerical_attr + nominal_attr]
            
        # Create the stats dataframe 
        df_pivot_table = pd.DataFrame(df.pivot_table(values = numerical_attr, index = grp_by_attr, aggfunc =['mean','std'] , fill_value = 0))
            
        df_pivot_table.reset_index(inplace = True)
            
        df_pivot_table.columns = grp_by_attr + list(map("_".join,[x[::-1] for x in df_pivot_table.columns][1:]))
            
        df_pt = df.copy().drop(columns = nominal_attr)
        
        # Merge the dataframes

        df_joined = pd.merge(df_pt, df_pivot_table, how = 'left', on = grp_by_attr).fillna(0)
            
        # This step is required as the mean and std columns are named using the numerical variable used - changes with every use
        mean_columns = [v for v in df_joined.columns if bool(re.match(r"(.*)[a-z]_mean$", v))]
        
        std_columns = [v for v in df_joined.columns if bool(re.match(r"(.*)[a-z]_std$", v))]

        # calculate the number of deviations 
        df_joined['no_devs'] = np.vectorize(cls.no_deviations)( df_joined[numerical_attr], df_joined[mean_columns], df_joined[std_columns])
            
        # calculate the min, max and average number of deviations
        out = cat_nominal_stats(DF = df_joined, grp_by_attr = grp_by_attr, nominal_attr = [] , numerical_attr = ['no_devs'], stats = stats )

        out.columns = grp_by_attr + ['_'.join(['_'.join(nominal_attr) ,v]) for v in out.columns[1:]]

        out.columns = [v[1:] if bool(re.match(r"^_(.*)", v)) else v for v in out.columns]
            
        return out, df_joined
    @classmethod
    def get_first_day(cls, dt, d_years=0, d_months=0):
        # d_years, d_months are "deltas" to apply to dt
        y, m = dt.year + d_years, dt.month + d_months
        a, m = divmod(m-1, 12)
        return date(y+a, m+1, 1)

    @classmethod
    def get_last_day(cls, dt):
        return cls.get_first_day(dt, 0, 1) + timedelta(-1)

    @classmethod
    def count_till_end_month(cls, dt):
        datetime_dt = datetime.date(int(dt.strftime("%Y")),int(dt.strftime("%m")),int(dt.strftime("%d")))
        return (cls.get_last_day(dt) - datetime_dt).days

    @classmethod
    def end_of_month(cls, dt):
        # total number of days in the month 
        total_days = (cls.get_last_day(dt) - cls.get_first_day(dt)).days
        # Half month mark
        mid_month = int(total_days/2)
        
        days_till_end = cls.count_till_end_month(dt)
        if days_till_end >= mid_month:
            return 'first_half_of_the_month'
        else:
            return 'second_half_of_the_month'


class FE_QMS_Stats(object):
    """
        # Add brief class description and list functions inside it 
    """
    @classmethod
    def func_concat_cols(cls, df, nominal_attr):
        """ function: concatenate multiple columns and clean the strings to 
                      replace non-alnumeric characters to underscore(_)
            Input:
                1. df. dataframe 
                2. nominal_attrs: list of column names to concatenate 
            Output:
                pd.series 
        """       
        # new_column = df[nominal_attr].apply(lambda x:'_'.join(x), axis=1)
        new_column = df[nominal_attr[0]] + '_' + df[nominal_attr[1]]

        new_column = new_column.apply(cls.clean_string)
        
        # returns a series 
        return new_column
    #End of Function 'func_concat_cols'


def add_trend_feature(arr, abs_values=False):
    """ function calculates the trend of a array of numbers
        Input:
            1. arrary of numbers
       
        Unit test:
        add_trend_feature(arr = np.array([5,2,3,4,5]))
    """
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]  
    # End of Function 'add_trend_feature'

def sparsity_features(series):
    # formula - count number of 0/ total number of elements 
    series = series.fillna(0)
    return np.sum(series == 0)/len(series)*100

def RFM(DF, cust_attr, recency_attr, freq_attr, monetary_attr):
    """ function is used to generate customer value to identify special groups of customer
        Inputs:
            1. DF - Dataframe 
            2. cust_attr - customer identifier
            3. recency_attr - datetime for the latest transaction/deals of a customer
            4. freq_attr - numeric - count of transactions/deals of a customer
            5. monetary_attr - numeric - value of customer's transactions/deals
        Output:
            > returns customer values labelling namely:
                1. churned
                2. freq_loyal 
                3. high_spend
    """
    # check the datatype of variables:
    if type(recency_attr) is not list:
        cust_attr = [cust_attr]
    
    if type(recency_attr) is list:
        recency_attr = ''.join(recency_attr)
        
    if type(freq_attr) is list:
        freq_attr = ''.join(freq_attr)
        
    if type(monetary_attr) is list:
        monetary_attr = ''.join(monetary_attr)
    
    # Subset dataframe:
    df = DF.loc[:, cust_attr + [recency_attr] + [freq_attr] + [monetary_attr]]
    
    # Create breaks to divide the data 
    recent_quantiles = df[recency_attr].quantile([0,0.25,0.75,1])
    freq_quantiles = df[freq_attr].quantile([0.25,0.5,0.75,1])
    amt_quantiles = df[monetary_attr].quantile([0,0.25,0.75,1])
    
    # Label customer with ranks
    df['recency_bins'] = pd.cut(df[recency_attr] , bins = recent_quantiles, include_lowest = True, labels = [3,2,1])
    df['freq_bins'] = pd.cut(df[freq_attr], bins = freq_quantiles, include_lowest = True, labels = [3,2,1])
    df['amt_bins'] = pd.cut(df[monetary_attr], bins = amt_quantiles, include_lowest = True, labels = [3,2,1])

    rfm_result = df.loc[:,['lcin','recency_bins','freq_bins','amt_bins']].drop_duplicates()

    rfm_result['rfm'] = rfm_result['recency_bins'].astype(float).fillna(0).astype(int).astype(str) + ',' + rfm_result['freq_bins'].astype(str) + ',' + rfm_result['amt_bins'].astype(str) 

    mapping = {
        # Churned Best Customers 
        ('3,1,1','3,2,1', '3,2,2', '3,1,2'): 'churned',
        # Lowest-Spending Active Loyal Customers -  frequency high
        ('1,1,3','2,1,3','1,2,3','2,2,3'): 'freq_loyal',
        # High-spending New Customers
        ('1,3,1','2,3,2','1,3,2','2,3,2'):'high_spend'
    }

    working_mapping= {}
    for k, v in mapping.items():
        for key in k:
            working_mapping[key] = v

    # Apply mapping to the dataframe 
    rfm_result['rfm_mapping'] = rfm_result.rfm.map(working_mapping)
    rfm_table = rfm_result.groupby(['lcin','rfm_mapping']).rfm_mapping.count().unstack('rfm_mapping').fillna(0).reset_index()

    # Join this table with the original table
    rfm_table_take = pd.merge(rfm_result,rfm_table, on ='lcin', how = 'outer')
    rfm_table_take = rfm_table_take.loc[:,['lcin']+[v for v in rfm_table_take.columns if 'rfm_' in v ]]
    rfm_table_take = rfm_table_take.fillna('OTH')
    
    return(rfm_table_take)
    # End of function - RFM

class data_cleaning():
    def clean_dtype(DF, dtype_dict):
        """ function converts data to the specified datatype
        
            Inputs:
                1. DF - Dataframe 
                2. dtype_dict - dictionary that specifies the columns and their corresponding datatype
                
            Output:
                > returns Dataframe with the corrected datatype
                
            Unit test:
                df = pd.DataFrame({
                    'col_a' : [1,2,3],
                    'col_b' : ['One',' TWO ','tHree** '],
                    'col_c' : ['2020-01-01 00:00:01','2020-01-02 00:00:15','2020-01-03 00:00:30']
                })
        
                # Create a dictionary for the columns you wish to clean/change:
                
                df_map = {
                    'col_a' : 'int64',
                    'col_b' : 'str',
                    'col_c' : 'datetime64'
                }
                
                # store the cleaned dataframe accordingly
                clean_df = clean_dtype(DF = df, dtype_dict =  df_map)
                
        """
        
        for item,key in dtype_dict.items():

            if key == 'str': 
                # clean strings - refer to function clean_string for more details 
                DF[item] = DF[item].astype(key).apply(FE_QMS_Stats.clean_string)
            elif key == 'datetime64':
                # For datetime variable, only preserve the date only - subscript first 10 digits
                DF[item] = pd.to_datetime([str(v)[0:10] for v in DF[item]])
            else: 
                DF[item] = DF[item].astype(key)
                
        return DF # cleaned_df


    @classmethod
    def clean_string(cls, string, remove_words = None):
        """ function: cleans string 
            Inputs:
                1. x. string 
                2. remove_words. list to words to remove
            Output - string
        """
        # convert all strings to lowercase
        string = string.lower()
        # Remove stopwords 
        if remove_words != None:
            for remove in remove_words:
                string = string.replace(remove,'')
        # Change '$' to dollar
        string = string.replace('$','dollar')
        # remove non alphabet/numeric characters
        string = re.sub(r'\W+',' ', string)
        # remove excess whitespace 
        string = re.sub(r'\s+', ' ',string).strip()
        # replace whitespace with underscore
        string = string.replace(" ","_")

        return string
    #End of Function 'clean_string'

    @classmethod
    def clean_missing_values(cls, DF, columns, threshold = 0.8, drop_cols = False, replace_missing = None):
        """ function: clean missing values
            Inputs:
                1. DF. dataframe 
                2. columns. columns to clean
                3. threshold. % of missing data to determine if a column should be removed
                4. drop_cols. Whether to drop columns
                5. replace_missing. if not None, specify a dictionary that specifies the method to use:
                    i. Numerical Variables - mean, min, max, median, value 
                    ii. Categorical Variables - max_count, value
                    WIP: datetime - as for now, missing values will not be replaced
                > example of replace_missing = {'num': 'mean', 'cat':'hello'}
            Output - string
        """
        if isinstance(DF, pd.DataFrame):
            # subset the columns to be used 
            DF = DF.loc[:, columns]

            # determine the variable type to determine what data cleaning method to be use
            num_var = [v for v in DF.columns if DF[v].dtypes == 'float64' or DF[v].dtypes == 'int64']
            cat_var = [v for v in DF.columns if DF[v].dtypes == 'object']

            # Calculate the % of missing values for each column
            check = apply(DF,2,lambda col: col.isnull().sum)

#%%
# Testing the code 

df = pd.DataFrame({
    'customer' : 'c1,c1,c1,c1,c2,c2,c2,c3,c3'.split(','),
    'product' : 'prod_01,prod_01,prod_02,prod_03,prod_03,prod_02,prod_01,prod_02,prod_05'.split(','),
    'amount' : sampling(x = [v for v in range(100)], no_samples = 9),
    'qnty' : sampling(x = list(np.repeat([np.nan],10)) + list(range(7)), no_samples = 9)
})


# %%
DF = df
num_var = [v for v in DF.columns if DF[v].dtypes == 'float64' or DF[v].dtypes == 'int64']
cat_var = [v for v in DF.columns if DF[v].dtypes == 'object']
print(num_var)
print(cat_var)


# %%

def clean_missing_values(DF, columns, threshold = 0.8, drop_cols = False, replace_missing = None):

    check = DF.apply(lambda col: col.isnull().sum()/len(col))

    # Check for columns with missing values
    if np.sum(check == 0) == DF.shape[1]:
        return DF # Return the dataframe since there is no missing values
    else:
        if drop_cols is True:
            drop_columns  = check.index[check > threshold]
            df = DF.drop(columns = drop_columns)
        else:
            if replace_missing is None:
                return df
            else:
                # Find columns
                num_var = [v for v in drop_columns if DF[v].dtypes == 'float64' or DF[v].dtypes == 'int64']

                # Switchers
                switcher = {
                    'mean': df.fillna(df.mean()[num_var]),
                    'min': df.fillna(df.min()[num_var]),
                    'max': df.fillna(df.max()[num_var]),
                    'median': df.fillna(df.median()[num_var]),
                }
                df = switcher.get(num_replace, df.fillna(replace_missing_value))

                return df


# %%
