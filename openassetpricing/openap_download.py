from . import urls
import polars as pl
import pandas as pd
import requests
from io import BytesIO
from zipfile import ZipFile
from pathlib import Path
import os
import shutil
from tabulate import tabulate


class OpenAP:
    def __init__(self):
        self.datasets_map = {
            'SignalDoc.csv': 'signal_doc',
            'PredictorPortsFull.csv': 'port_op',
            'PredictorAltPorts_Deciles.zip': 'port_deciles',
            'PredictorAltPorts_DecilesVW.zip': 'port_deciles_vw'
        }

    def list_datasets(self):
        table = [[key, value] for key, value in self.datasets_map.items()]
        headers = ['CZ data file', 'Name for download']
        print(tabulate(table, headers, tablefmt='simple_outline'))

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
            df = (
                df.filter(pl.col('signalname').is_in(predictor))
                .sort('signalname', 'port', 'date')
            )

        return self._convert_to_backend(df, df_backend)

    def _dl_port_alt(self, temp_dir, df_backend, predictor=None):
        df = (
            pl.read_csv(
                list(temp_dir.glob('*.csv'))[0],
                null_values='NA',
                schema_overrides={'port': pl.String})
            .sort('signalname', 'port', 'date')
        )
        shutil.rmtree(temp_dir, ignore_errors=True)
        if predictor:
            df = (
                df.filter(pl.col('signalname').is_in(predictor))
                .sort('signalname', 'port', 'date')
            )

        return self._convert_to_backend(df, df_backend)

    def dl(self, data_name, df_backend, predictor=None):
        url = getattr(urls, f'{data_name}_url', None)
        if url:
            if data_name == 'signal_doc':
                return self._dl_signal_doc(url, df_backend)
            if data_name == 'port_op':
                return self._dl_port_op(url, df_backend, predictor)
            if data_name in ['port_deciles', 'port_deciles_vw']:
                source = requests.get(url).content
                io_data = BytesIO(source)
                zip_file = ZipFile(io_data)
                temp_dir = Path.cwd()/'openap_temp_for_zip_file'
                os.makedirs(temp_dir, exist_ok=True)
                zip_file.extractall(path=temp_dir)
                return self._dl_port_alt(temp_dir, df_backend, predictor)
        else:
            raise ValueError('Dataset is not available.')
