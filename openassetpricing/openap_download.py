from . import urls
import polars as pl
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from tabulate import tabulate
import wrds
import time


class OpenAP:
    def __init__(self):
        self.url = None
        self.datasets_map = {
            'SignalDoc.csv': 'signal_doc',
            'PredictorPortsFull.csv': 'port_op',
            'PredictorAltPorts_Deciles.zip': 'port_deciles_ew',
            'PredictorAltPorts_DecilesVW.zip': 'port_deciles_vw',
            'PredictorAltPorts_LiqScreen_ME_gt_NYSE20pct.zip': 'port_ex_nyse_p20_me',
            'PredictorAltPorts_LiqScreen_NYSEonly.zip': 'port_nyse',
            'PredictorAltPorts_LiqScreen_Price_gt_5.zip': 'port_ex_price5',
            'PredictorAltPorts_Quintiles.zip': 'port_quintiles_ew',
            'PredictorAltPorts_QuintilesVW.zip': 'port_quintiles_vw',
            'signed_predictors_dl_wide.zip': 'char_predictors'
        }

    def list_datasets(self):
        table = [[key, value] for key, value in self.datasets_map.items()]
        headers = ['CZ data file', 'Name for download']
        print(tabulate(table, headers, tablefmt='simple_outline'))

    def _get_url(self, data_name):
        self.url = getattr(urls, f'{data_name}_url', None)
        return self.url

    def _zip_source(self, url):
        # Reading in chunks is 20% faster for large single file
        chunk_size = 1024 * 1024 * 10
        source = requests.get(url, stream=True)
        io_data = BytesIO()
        for chunk in source.iter_content(chunk_size=chunk_size):
            if chunk:
                io_data.write(chunk)

        io_data.seek(0)
        zip_file = ZipFile(io_data)
        return zip_file

    def _convert_to_backend(self, df, df_backend):
        if df_backend == 'polars':
            return df
        if df_backend == 'pandas':
            return df.to_pandas()

    def _dl_signal_doc(self, url, df_backend):
        df = pl.read_csv(url, infer_schema_length=300)
        return self._convert_to_backend(df, df_backend)

    def _port_indiv(self, df, predictor):
        n_input = len(predictor)
        df = df.filter(pl.col('signalname').is_in(predictor))
        n = df['signalname'].n_unique()
        if n != n_input:
            print('One or more input predictors are not available.\n')

        return df

    def _dl_port_op(self, url, df_backend, predictor=None):
        if not predictor:
            df = (
                pl.read_csv(
                    url, null_values='NA', schema_overrides={'port': pl.String})
                .with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
            )

        if predictor:
            if type(predictor) is list:
                df = (
                    pl.read_csv(
                        url, null_values='NA',
                        schema_overrides={'port': pl.String})
                    .with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
                )
                df = self._port_indiv(df, predictor)
            else:
                print('Predictor must be a list')

        df = df.sort('signalname', 'port', 'date')
        return self._convert_to_backend(df, df_backend)

    def _dl_port_alt(self, df_backend, predictor=None):
        if not predictor:
            zip_file = self._zip_source(self.url)
            df = (
                pl.read_csv(
                    zip_file.read(zip_file.filelist[0]), null_values='NA',
                    schema_overrides={'port': pl.String})
                .with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
            )

        if predictor:
            if type(predictor) is list:
                zip_file = self._zip_source(self.url)
                df = (
                    pl.read_csv(
                        zip_file.read(zip_file.filelist[0]), null_values='NA',
                        schema_overrides={'port': pl.String})
                    .with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
                )
                df = self._port_indiv(df, predictor)
            else:
                print('Predictor must be a list')

        df = df.sort('signalname', 'port', 'date')
        return self._convert_to_backend(df, df_backend)

    def _dl_char_crsp_3predictors(self):
        conn = wrds.Connection()

        df = conn.raw_sql(
            """
            select permno, date, prc, ret, shrout
            from crsp.msf
            """
            , date_cols=['date'])

        # They are signed
        df = (
            pl.from_pandas(df)
            .select(
                pl.col('permno').cast(pl.Int32),
                pl.col('date').dt.year().mul(100)
                .add(pl.col('date').dt.month())
                .cast(pl.Int32).alias('yyyymm'),
                pl.col('prc').abs().log().mul(-1).alias('Price'),
                pl.col('prc').abs().mul(pl.col('shrout')).truediv(1000)
                .log().mul(-1).alias('Size'),
                pl.col('ret').fill_null(0).mul(-1).alias('STreversal')
            )
        )
        return df

    def _dl_char_predictors(self, df_backend, predictor=None):
        if not predictor:
            temp = self._dl_char_crsp_3predictors()
            zip_file = self._zip_source(self.url)
            df = (
                pl.read_csv(
                    zip_file.read(zip_file.filelist[0]), infer_schema_length=0)
                .with_columns(
                    pl.col('permno', 'yyyymm').cast(pl.Int32),
                    pl.exclude('permno', 'yyyymm').cast(pl.Float64))
                .join(temp, how='left', on=['permno', 'yyyymm'])
            )

        if predictor:
            if type(predictor) is list:
                temp = self._dl_char_crsp_3predictors()
                zip_file = self._zip_source(self.url)
                df = (
                    pl.read_csv(
                        zip_file.read(zip_file.filelist[0]),
                        infer_schema_length=0)
                    .with_columns(
                        pl.col('permno', 'yyyymm').cast(pl.Int32),
                        pl.exclude('permno', 'yyyymm').cast(pl.Float64))
                    .join(temp, how='left', on=['permno', 'yyyymm'])
                )
                try:
                    df = df.select('permno', 'yyyymm', pl.col(predictor))
                except:
                    print('One or more input predictors are not available.\n')
            else:
                print('Predictor must be a list')

        df = df.sort('permno', 'yyyymm')
        return self._convert_to_backend(df, df_backend)

    def dl(self, data_name, df_backend, predictor=None):
        port_alt_list = [
            'port_deciles_ew', 'port_deciles_vw',
            'port_ex_nyse_p20_me', 'port_nyse', 'port_ex_price5',
            'port_quintiles_ew', 'port_quintiles_vw']

        if df_backend in ['polars', 'pandas']:
            self.url = self._get_url(data_name)
            if self.url:
                start_time = time.time()
                if data_name == 'signal_doc':
                    df = self._dl_signal_doc(self.url, df_backend)
                if data_name == 'port_op':
                    df = self._dl_port_op(self.url, df_backend, predictor)
                if data_name in port_alt_list:
                    df = self._dl_port_alt(df_backend, predictor)
                if data_name == 'char_predictors':
                    df = self._dl_char_predictors(df_backend, predictor)

                end_time = time.time()
                time_used = end_time - start_time
                if time_used <= 60:
                    print(f'Data is downloaded: {time_used:.0f}s')
                else:
                    print(f'Data is downloaded: {time_used/60:.0f} mins')

                return df
            else:
                raise ValueError('Dataset is not available.')
        else:
            raise ValueError("Unsupported backend. Choose 'polars' or 'pandas'.")
