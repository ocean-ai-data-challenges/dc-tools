#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataloder."""

from abc import ABC
from argparse import Namespace
import multiprocessing as mp
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from loguru import logger
import numpy as np
import pandas as pd
# from pathlib import Path
import xarray as xr
import torch
# import xbatcher as xb
from xrpatcher import XRDAPatcher

from dctools.data.connection.connection_manager import BaseConnectionManager
from dctools.data.datasets.dc_catalog import DatasetCatalog
from dctools.data.transforms import CustomTransforms


class EvaluationDataloader:
    def __init__(
        self,
        pred_catalog: DatasetCatalog,
        ref_catalog: DatasetCatalog,
        pred_manager: BaseConnectionManager,
        ref_manager: BaseConnectionManager,
        pred_alias: str,
        batch_size: int = 1,
        pred_transform: Optional[CustomTransforms] = None,
        ref_transform: Optional[CustomTransforms] = None,
        eval_variables: Optional[List[str]] = None,
    ):
        """
        Initialise le dataloader pour les ensembles de données.

        Args:
            pred_catalog (DatasetCatalog): Catalogue des prédictions.
            ref_catalog (DatasetCatalog): Catalogue des références.
            batch_size (int): Taille des lots.
            pred_transform (Optional[CustomTransforms]): Transformation pour les prédictions.
            ref_transform (Optional[CustomTransforms]): Transformation pour les références.
        """
        self.pred_catalog = pred_catalog
        self.ref_catalog = ref_catalog
        self.pred_manager = pred_manager
        self.ref_manager = ref_manager
        self.pred_alias = pred_alias
        self.batch_size = batch_size
        self.pred_transform = pred_transform
        self.ref_transform = ref_transform
        self.eval_variables = eval_variables

        # Aligner les catalogues
        self._align_catalogs()

    def __len__(self):
        return len(self.pred_catalog.get_dataframe())

    def __iter__(self):
        return self._generate_batches()

    def _align_catalogs(self):
        """
        Aligne les catalogues de prédiction et de référence en fonction des dates.

        Cette méthode s'assure que les deux catalogues contiennent les mêmes
        intervalles de temps (`date_start` et `date_end`).
        """
        if not self.ref_catalog:
            logger.warning("No ref catalog. Cancel alignment.")
            return
        pred_df = self.pred_catalog.get_dataframe()
        ref_df = self.ref_catalog.get_dataframe()

        # Effectuer une jointure interne sur les colonnes `date_start` et `date_end`
        aligned_df = pd.merge(
            pred_df,
            ref_df,
            on=["date_start", "date_end"],
            suffixes=("_pred", "_ref"),
            how="inner",
        )

        # Inclure explicitement `date_start` et `date_end` dans les dictionnaires
        pred_entries = aligned_df[
            ["date_start", "date_end"] + [col for col in aligned_df.columns if col.endswith("_pred")]
        ].rename(columns=lambda x: x.replace("_pred", "")).to_dict(orient="records")

        ref_entries = aligned_df[
            ["date_start", "date_end"] + [col for col in aligned_df.columns if col.endswith("_ref")]
        ].rename(columns=lambda x: x.replace("_ref", "")).to_dict(orient="records")

        # Mettre à jour les catalogues avec les données alignées
        self.pred_catalog = DatasetCatalog(entries=pred_entries)
        self.ref_catalog = DatasetCatalog(entries=ref_entries)

    def _generate_batches(self) -> Generator[Dict[str, xr.Dataset], None, None]:
        """
        Génère des lots d'ensembles de données pour l'évaluation.

        Yields:
            Dict[str, xr.Dataset]: Dictionnaire contenant les datasets de prédiction et de référence.
        """
        batch = []
        pred_df = self.pred_catalog.get_dataframe()
        ref_df = self.ref_catalog.get_dataframe() if self.ref_catalog else None

        if ref_df is None:
            logger.warning("No reference catalog provided. Skipping reference data.")

            for _, pred_entry in pred_df.iterrows():
                entry = {
                    "date": pred_entry["date_start"],
                    "pred_data": pred_entry["path"],
                    "ref_data": None,
                }
                batch.append(entry)

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch
        else:
            for ((_, pred_entry), (_, ref_entry)) in zip(pred_df.iterrows(), ref_df.iterrows()):
                entry = {
                    "date": pred_entry["date_start"],
                    "pred_data": pred_entry["path"],
                    "ref_data": ref_entry["path"],
                }
                batch.append(entry)

                # Retourner le lot lorsque la taille est atteinte
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []

            # Retourner le dernier lot s'il reste des éléments
            if batch:
                yield batch

    def open_pred(self, pred_entry: str) -> xr.Dataset:
        pred_data = self.pred_manager.open(pred_entry)

        from datetime import datetime
        start_time_glorys = datetime.fromisoformat(str(pred_data["time"][0].values))
        return pred_data

    def open_ref(self, ref_entry: str) -> xr.Dataset:
        ref_data = self.pred_manager.open(ref_entry)
        return ref_data


    '''def _find_matching_reference(self, pred_row: pd.Series) -> Optional[xr.Dataset]:
        """
        Trouve le fichier de référence correspondant à un fichier de prédiction.

        Args:
            pred_row (pd.Series): Métadonnées du fichier de prédiction.

        Returns:
            Optional[xr.Dataset]: Dataset de référence correspondant.
        """
        ref_row = self.catalog.df[
            (self.catalog.df["type"] == "ref") &
            (self.catalog.df["date_start"] <= pred_row["date_start"]) &
            (self.catalog.df["date_end"] >= pred_row["date_end"]) &
            (self.catalog.df["lat_min"] <= pred_row["lat_min"]) &
            (self.catalog.df["lat_max"] >= pred_row["lat_max"]) &
            (self.catalog.df["lon_min"] <= pred_row["lon_min"]) &
            (self.catalog.df["lon_max"] >= pred_row["lon_max"])
        ]
        if not ref_row.empty:
            return xr.open_dataset(ref_row.iloc[0]["path"])
        return None'''


class TorchCompatibleDataloader:
    def __init__(
        self,
        dataloader: EvaluationDataloader,
        patch_size: Tuple[int, int],
        stride: Tuple[int, int],
    ):
        """
        Initialise un dataloader compatible avec PyTorch.

        Args:
            dataloader (EvaluationDataloader): Le dataloader existant.
            patch_size (Tuple[int, int]): Taille des patches (hauteur, largeur).
            stride (Tuple[int, int]): Pas de glissement pour les patches.
        """
        self.dataloader = dataloader
        self.patch_size = patch_size
        self.stride = stride

    def __len__(self):
        """
        Retourne le nombre total de lots dans le dataloader.
        """
        return len(self.dataloader)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Génère des lots de données compatibles avec PyTorch.

        Yields:
            Dict[str, torch.Tensor]: Un dictionnaire contenant les patches de données.
        """
        for batch in self.dataloader:
            for entry in batch:
                pred_data = entry["pred_data"]
                ref_data = entry["ref_data"]

                # Générer des patches pour les données de prédiction
                pred_patches = self._generate_patches(pred_data)

                # Générer des patches pour les données de référence (si disponibles)
                ref_patches = self._generate_patches(ref_data) if ref_data is not None else None

                # Retourner les patches sous forme de tensors PyTorch
                yield {
                    "date": entry["date"],
                    "pred_patches": pred_patches,
                    "ref_patches": ref_patches,
                }

    def _generate_patches(self, dataset: xr.Dataset) -> torch.Tensor:
        """
        Génère des patches à partir d'un dataset xarray.

        Args:
            dataset (xr.Dataset): Le dataset xarray.

        Returns:
            torch.Tensor: Les patches sous forme de tensor PyTorch.
        """
        patcher = XRDAPatcher(
            data=dataset,
            patch_size=self.patch_size,
            stride=self.stride,
        )
        patches = patcher.extract_patches()
        return torch.tensor(patches)