#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataloder."""

from abc import ABC, abstractmethod
from argparse import Namespace
import multiprocessing as mp
import os
from typing import Any, Generator, List, Optional, Tuple

import dask
import numpy as np
# from pathlib import Path
import xarray as xr
import xbatcher as xb

from dctools.data.dataset import DCDataset
# import xbatcher

# from dctools.dcio.loader import FileLoader

# To install Xbatcher and PyTorch via Conda:
# conda install -c conda-forge xbatcher pytorch


class DatasetLoader(ABC):
    """Base class for data loaders."""
    def __init__(
        self,
        pred_dataset: xr.Dataset,
        ref_dataset: Optional[xr.Dataset] = None,
        batch_size: int = 1,
    ) -> None:
        """
        Initialise avec une fonction de chargement et (optionnellement) une fonction de prétraitement.
        :param load_function: Fonction qui charge les données et retourne un xarray.Dataset
        :param preprocess_function: Fonction appliquée au Dataset après chargement.
        """
        self.batch_size = batch_size
        self.pred_dataset = pred_dataset
        self.ref_dataset = ref_dataset
        # self.batch_size = conf_args.batch_size if conf_args.batch_size else -1
        # super().__init__(conf_args, batch_size)

    def __len__(self):
        return self.pred_dataset.__len__()

    def get_ref_data(self):
        for index in range(self.pred_dataset.__len__()):
            try:
                #date = self.pred_dataset.get_date(index)
                #TODO: Add batcher when getting data
                """bgen_ref = xb.BatchGenerator(
                    ds=self.ref_dataset.__getitem__(index) if self.ref_dataset else None,
                    input_dims={'time': self.batch_size},
                )
                bgen_pred = xb.BatchGenerator(
                    ds=self.pred_dataset.__getitem__(index),
                    input_dims={'time': self.batch_size},
                )"""
                return self.ref_dataset.__getitem__(index) if self.ref_dataset else None
            except IndexError:
                return None

    '''def get_date(self):
        for index in range(self.pred_dataset.__len__()):
            try:
                return self.pred_dataset.get_date(index)
            except IndexError:
                return None'''

    def get_pred_data(self):
        for index in range(self.pred_dataset.__len__()):
            try:
                #date = self.pred_dataset.get_date(index)
                """bgen_ref = xb.BatchGenerator(
                    ds=self.ref_dataset.__getitem__(index) if self.ref_dataset else None,
                    input_dims={'time': self.batch_size},
                )
                bgen_pred = xb.BatchGenerator(
                    ds=self.pred_dataset.__getitem__(index),
                    input_dims={'time': self.batch_size},
                )"""
                return self.pred_dataset.__getitem__(index)
            except IndexError:
                return None
 
    # reset list and counter 
    def reset(self):                                                                                                                                                                           
        self.__init__()

    """def load_ref(self) -> Generator[int, Any, DCDataset]:
        if self.ref_dataset:
            assert(self.ref_dataset.__len__() == self.pred_dataset.__len__())
        # assert(self.dataset.__len__() == self.dataset.__len__())
        #yield(dask.delayed(self.get_ref_data()))
        yield(self.get_ref_data())


    def load_pred(self) -> Generator[int, Any, DCDataset]:
        if self.ref_dataset:
            assert(self.ref_dataset.__len__() == self.pred_dataset.__len__())
        # assert(self.dataset.__len__() == self.dataset.__len__())
        #yield(dask.delayed(self.get_pred_data()))
        yield(self.get_pred_data())

    def load_date(self) -> Generator[int, Any, str]:
        yield(self.get_date())"""

    def load(self) -> Generator[int, Any, DCDataset]:
        if self.ref_dataset:
            assert(self.ref_dataset.__len__() == self.pred_dataset.__len__())
        for index in range(self.pred_dataset.__len__()):
            try:
                pred_sample = self.pred_dataset.__getitem__(index)
                if self.ref_dataset:
                    ref_sample = self.ref_dataset.__getitem__(index)
                else:
                    ref_sample = None
                date = self.pred_dataset.get_date(index)
            except IndexError:
                pred_sample = None
                ref_sample = None
                date = None
            yield (date, pred_sample, ref_sample)
        '''# assert(self.dataset.__len__() == self.dataset.__len__())
        #yield(dask.delayed(self.get_ref_data()))
        yield((self.get_date(), self.get_pred_data(), self.get_ref_data()))'''

    '''def load(self) -> Generator[Any, Any]:
        assert(self.ref_dataset.__len__() == self.pred_dataset.__len__())
        #yield((self.load_pred, self.load_ref))
        yield((self.load_date, self.load_pred, self.load_ref))'''

    '''def get_predict(self):
        for index in range(self.dataset_glorys.__len__()):
            return xb.BatchGenerator(
                self.dataset_glorys.__getitem__(index), {'time': self.batch_size}
            )

    def load_ref(self):
        for index in range(self.dataset_glorys.__len__()):
            return xb.BatchGenerator(
                self.dataset_glonet.__getitem__(index), {'time': self.batch_size}
            )

    def load(self) -> Generator[Any, Any]:
        assert(self.dataset_glonet.__len__() == self.dataset_glorys.__len__())
        yield((self.load_predict, self.load_ref))'''
