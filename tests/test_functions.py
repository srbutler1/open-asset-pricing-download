import urls
from test_gdrive_parse import _get_name_id_map, _get_readable_link
import polars as pl
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from tabulate import tabulate
import wrds
import re
import time


def list_release():
    df = pl.DataFrame(
        {'release':
         [re.findall(r'release(\d+)_url', i)[0]
          for i in dir(urls) if i.startswith('release')]}
    )
    table = [i for i in df.iter_rows()]
    headers = ['Release']
    print(tabulate(table, headers, tablefmt='simple_outline'))

class OpenAP:
    def __init__(self, release_year=None):
        if not release_year:
            release_url = getattr(urls, dir(urls)[-1], None)
        if release_year:
            release_url = getattr(urls, f'release{release_year}_url', None)

        self.name_id_map, self.individual_signal_id_map = _get_name_id_map(release_url)
        self.signal_sign = (
            pl.read_csv(
                self._get_url('signal_doc'), infer_schema_length=300,
                columns=['Acronym', 'Sign'], null_values='NA')
            .rename({'Acronym': 'signal', 'Sign': 'sign'})
            .with_columns(pl.col('sign').cast(pl.Int8))
        )

    def list_port(self):
        df = (
            self.name_id_map.select('name', 'download_name')
            .filter(pl.col('name').str.starts_with('Predictor'))
        )
        table = [i for i in df.iter_rows()]
        headers = ['CZ portfolio file', 'Name for download']
        print(tabulate(table, headers, tablefmt='simple_outline'))

    def _get_url(self, data_name):
        data_header = self.name_id_map.filter(pl.col('download_name')==data_name)
        file_with_confirm = ['firm_char', 'deciles_ew', 'deciles_vw']
        if data_name in file_with_confirm:
            self.url = _get_readable_link(data_header[0, 'file_id'])
        else:
            self.url = data_header[0, 'file_id']

        return self.url

    def _get_individual_signal_url(self, signal_name):
        data_header = self.individual_signal_id_map.filter(pl.col('signal')==signal_name)
        self.url = data_header[0, 'file_id']
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

    def _port_indiv(self, df, predictor):
        n_input = len(predictor)
        df = df.filter(pl.col('signalname').is_in(predictor))
        n = df['signalname'].n_unique()
        if n != n_input:
            print('One or more input predictors are not available.')

        return df

    def _dl_port_op(self, df_backend, predictor=None):
        if not predictor:
            df = (
                pl.read_csv(
                    self.url, null_values='NA',
                    schema_overrides={'port': pl.String})
                .with_columns(pl.col('date').str.to_date('%Y-%m-%d'))
            )

        if predictor:
            if type(predictor) is list:
                df = (
                    pl.read_csv(
                        self.url, null_values='NA',
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

    def _dl_signal_crsp3(self):
        conn = wrds.Connection()

        df = conn.raw_sql(
            """
            select permno, date, prc, ret, shrout
            from crsp.msf
            """, date_cols=['date']
        )

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
            .with_columns(
                pl.when(pl.col('Size').is_finite())
                .then(pl.col('Size'))
                .alias('Size')
            )
        )
        return df

    def _dl_signal(self, df_backend, predictor=None):
        if not predictor:
            temp = self._dl_signal_crsp3()
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
            crsp3 = {'Price', 'Size', 'STreversal'}
            ex_crsp3 = [i for i in predictor if i not in crsp3]
            if type(predictor) is list:
                try:
                    if crsp3 & set(predictor):
                        temp = self._dl_signal_crsp3()

                    if len(ex_crsp3) > 0:
                        zip_file = self._zip_source(self.url)
                        df = (
                            pl.from_pandas(
                                pd.read_csv(
                                    zip_file.open(zip_file.filelist[0]),
                                    usecols=['permno', 'yyyymm']+ex_crsp3,
                                    engine='pyarrow')
                            )
                            .with_columns(
                                pl.col('permno', 'yyyymm').cast(pl.Int32))
                        )
                        if len(ex_crsp3) < len(predictor):
                            df = df.join(
                                temp, how='left', on=['permno', 'yyyymm'])
                    else:
                        df = temp

                    df = (
                        df.select('permno', 'yyyymm', pl.col(predictor))
                        .filter(
                            pl.any_horizontal(pl.col(predictor)).is_not_null())
                        .with_columns(
                            pl.exclude('permno', 'yyyymm').cast(pl.Float64))
                        .sort('permno', 'yyyymm')
                    )
                except:
                    print('One or more input predictors are not available.')
            else:
                print('Predictor must be a list')

        df = df.sort('permno', 'yyyymm')
        return self._convert_to_backend(df, df_backend)

    def _dl_individual_signal(self, df_backend, predictor, signed=False):
        crsp3 = {'Price', 'Size', 'STreversal'}
        ex_crsp3 = [i for i in predictor if i not in crsp3]
        if type(predictor) is list:
            try:
                if crsp3 & set(predictor):
                    temp = self._dl_signal_crsp3()
                    # Price, Size and STreversal from CRSP by default are signed
                    # If signed = True, do nothing
                    # If signed = False, multiply -1
                    if not signed:
                        temp = temp.with_columns(
                            pl.col('Price').mul(-1),
                            pl.col('Size').mul(-1),
                            pl.col('STreversal').mul(-1))

                if len(ex_crsp3) > 0:
                    df = pl.DataFrame(
                        schema={'permno': pl.Int32, 'yyyymm': pl.Int32})
                    for i in ex_crsp3:
                        self.url = self._get_individual_signal_url(i)
                        temp_signal = pl.read_csv(self.url)
                        if len(temp_signal) > 0:
                            temp_signal = temp_signal.with_columns(
                                pl.col('permno', 'yyyymm').cast(pl.Int32))
                        if len(temp_signal) == 0:
                            self.url = _get_readable_link(self.url)
                            temp_signal = (
                                pl.read_csv(self.url)
                                .with_columns(
                                    pl.col('permno', 'yyyymm').cast(pl.Int32))
                            )
                        if signed:
                            _sign = (
                                self.signal_sign.filter(pl.col('signal')==i)
                                .get_column('sign')[0])
                            if _sign is not None:
                                temp_signal = (
                                    temp_signal.with_columns(pl.col(i)*_sign))

                        df = df.join(
                            temp_signal, how='full', on=['permno', 'yyyymm'],
                            coalesce=True)
                    if len(ex_crsp3) < len(predictor):
                        df = df.join(
                            temp, how='full', on=['permno', 'yyyymm'],
                            coalesce=True)
                else:
                    df = temp

                df = (
                    df.select('permno', 'yyyymm', pl.col(predictor))
                    .filter(pl.any_horizontal(pl.col(predictor)).is_not_null())
                    .with_columns(
                        pl.exclude('permno', 'yyyymm').cast(pl.Float64))
                    .sort('permno', 'yyyymm')
                )
            except:
                print('One or more input predictors are not available.')
        else:
            print('Predictor must be a list')

        df = df.sort('permno', 'yyyymm')
        return self._convert_to_backend(df, df_backend)

    def _print_time(self, time_used):
        if time_used <= 60:
            print(f'\nData is downloaded: {time_used:.0f}s')
        else:
            print(f'\nData is downloaded: {time_used/60:.0f} mins')

    def dl_signal_doc(self, df_backend):
        url = self._get_url('signal_doc')
        df = pl.read_csv(url, infer_schema_length=300)
        return self._convert_to_backend(df, df_backend)

    def dl_port(self, data_name, df_backend, predictor=None):
        port_alt_list = [
            'deciles_ew', 'deciles_vw', 'ex_nyse_p20_me', 'nyse', 'ex_price5',
            'quintiles_ew', 'quintiles_vw']

        if df_backend in ['polars', 'pandas']:
            self.url = self._get_url(data_name)
            if self.url:
                start_time = time.time()
                if data_name == 'op':
                    df = self._dl_port_op(df_backend, predictor)
                if data_name in port_alt_list:
                    df = self._dl_port_alt(df_backend, predictor)

                end_time = time.time()
                time_used = end_time - start_time
                self._print_time(time_used)
                return df
            else:
                raise ValueError('Dataset is not available.')
        else:
            raise ValueError("Unsupported backend. Choose 'polars' or 'pandas'.")

    def dl_all_signals(self, df_backend, predictor=None):
        if df_backend in ['polars', 'pandas']:
            self.url = self._get_url('firm_char')
            if self.url:
                start_time = time.time()
                df = self._dl_signal(df_backend, predictor)
                end_time = time.time()
                time_used = end_time - start_time
                self._print_time(time_used)
                return df
            else:
                raise ValueError('Dataset is not available.')
        else:
            raise ValueError("Unsupported backend. Choose 'polars' or 'pandas'.")

    def dl_signal(self, df_backend, predictor, signed=False):
        if df_backend in ['polars', 'pandas']:
            start_time = time.time()
            df = self._dl_individual_signal(df_backend, predictor, signed)
            end_time = time.time()
            time_used = end_time - start_time
            self._print_time(time_used)
            return df
        else:
            raise ValueError("Unsupported backend. Choose 'polars' or 'pandas'.")

list_release()

openap = OpenAP()

openap.list_port()

df = openap.dl_signal_doc('pandas')
df = openap.dl_signal_doc('polars')

df = openap.dl_port('op', 'polars')
df = openap.dl_port('op', 'polars', ['AM'])
df = openap.dl_port('op', 'polars', ['AM', 'Mom12m'])

df = openap.dl_port('deciles_ew', 'polars')
df = openap.dl_port('deciles_vw', 'polars')
df = openap.dl_port('deciles_ew', 'polars', ['Accruals'])
df = openap.dl_port('deciles_ew', 'polars', ['BM', 'Mom6m'])

df = openap.dl_all_signals('pandas', ['BM', 'Mom12m'])
df = openap.dl_all_signals('pandas', ['AssetGrowth'])
df = openap.dl_all_signals('polars')
df = openap.dl_all_signals('pandas', ['BM', 'Mom6m', 'Size'])
df = openap.dl_all_signals('polars', ['BM', 'Mom6m', 'zerotrade6M'])

df = openap.dl_signal('polars', ['BM'])
df = openap.dl_signal('pandas', ['AssetGrowth'])
df = openap.dl_signal('pandas', ['AssetGrowth'], signed=True)
df = openap.dl_signal('polars', ['BM', 'Mom6m', 'Size'])
df = openap.dl_signal('polars', ['BM', 'Mom6m', 'Size'], signed=True)
df = openap.dl_signal('polars', ['BM', 'Mom6m', 'zerotrade6M'])

df
