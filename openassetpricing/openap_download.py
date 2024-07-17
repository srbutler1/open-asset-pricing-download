from . import urls
import polars as pl
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from tabulate import tabulate
import wrds


class OpenAP:
    def __init__(self):
        self.url = None
        self.datasets_map = {
            'SignalDoc.csv': 'signal_doc',
            'PredictorPortsFull.csv': 'port_op',
            'PredictorAltPorts_Deciles.zip': 'port_deciles',
            'PredictorAltPorts_DecilesVW.zip': 'port_deciles_vw',
            'signed_predictors_dl_wide.zip': 'char_all_predictors'
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
        elif df_backend == 'pandas':
            return df.to_pandas()
        else:
            raise ValueError("Unsupported backend. Choose 'polars' or 'pandas'.")

    def _dl_signal_doc(self, url, df_backend):
        df = pl.read_csv(url, infer_schema_length=300)
        return self._convert_to_backend(df, df_backend)

    def _dl_port_op(self, url, df_backend, predictor=None):
        df = (
            pl.read_csv(
                url, null_values='NA', schema_overrides={'port': pl.String})
            .sort('signalname', 'port', 'date')
        )
        if predictor:
            try:
                df = (
                    df.filter(pl.col('signalname').is_in(predictor))
                    .sort('signalname', 'port', 'date')
                )
            except:
                print('The predictor is not available')

        return self._convert_to_backend(df, df_backend)

    def _dl_port_alt(self, df_backend, predictor=None):
        zip_file = self._zip_source(self.url)
        df = (
            pl.read_csv(
                zip_file.read(zip_file.filelist[0]),
                null_values='NA',
                schema_overrides={'port': pl.String})
            .sort('signalname', 'port', 'date')
        )
        if predictor:
            try:
                df = (
                    df.filter(pl.col('signalname').is_in(predictor))
                    .sort('signalname', 'port', 'date')
                )
            except:
                print('The predictor is not available')

        return self._convert_to_backend(df, df_backend)

    def _dl_char_crsp_3predictors(self):
        conn = wrds.Connection()

        df = conn.raw_sql(
            """
            select permno, date, prc, ret, shrout
            from crsp.msf
            """
            , date_cols=['date'])

        df = (
            pl.from_pandas(df)
            .select(
                pl.col('permno').cast(pl.Int32),
                pl.col('date').dt.year().mul(100)
                .add(pl.col('date').dt.month())
                .cast(pl.Int32).alias('yyyymm'),
                pl.col('prc').abs().log().alias('Price'),
                pl.col('prc').abs().mul(pl.col('shrout')).truediv(1000)
                .log().alias('Size'),
                pl.col('ret').fill_null(0).alias('STreversal')
            )
        )
        return df

    def _dl_char_predictors(self, df_backend, predictor=None):
        temp = self._dl_char_crsp_3predictors()
        zip_file = self._zip_source(self.url)
        df = (
            pl.read_csv(
                zip_file.read(zip_file.filelist[0]), infer_schema_length=0)
            .with_columns(
                pl.col('permno', 'yyyymm').cast(pl.Int32),
                pl.exclude('permno', 'yyyymm').cast(pl.Float64))
            .join(temp, how='left', on=['permno', 'yyyymm'])
            .sort('permno', 'yyyymm')
        )
        if predictor:
            try:
                df = (
                    df.select('permno', 'yyyymm', pl.col(predictor))
                    .sort('permno', 'yyyymm')
                )
            except:
                print('The predictor is not available')

        return self._convert_to_backend(df, df_backend)

    def dl(self, data_name, df_backend, predictor=None):
        port_alt_list = ['port_deciles', 'port_deciles_vw']
        self.url = self._get_url(data_name)
        if self.url:
            if data_name == 'signal_doc':
                return self._dl_signal_doc(self.url, df_backend)
            if data_name == 'port_op':
                return self._dl_port_op(self.url, df_backend, predictor)
            if data_name in port_alt_list:
                return self._dl_port_alt(df_backend, predictor)
            if data_name == 'char_predictors':
                return self._dl_char_predictors(df_backend, predictor)
        else:
            raise ValueError('Dataset is not available.')
