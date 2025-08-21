#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""Dataloder."""

from abc import ABC
from argparse import Namespace
import multiprocessing as mp
import os
from typing import Any, Dict, Generator, List, Optional, Tuple

from loguru import logger
from memory_profiler import profile
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
from dctools.utilities.file_utils import FileCacheManager
from dctools.utilities.xarray_utils import UnifiedObservationView



class EvaluationDataloader:
    def __init__(
        self,
        pred_connection_params: dict,
        ref_connection_params: dict,
        pred_catalog: DatasetCatalog,
        ref_catalogs: Dict[str, DatasetCatalog],
        pred_manager: BaseConnectionManager,
        ref_managers: Dict[str, BaseConnectionManager],
        pred_alias: str,
        ref_aliases: List[str],
        batch_size: int = 1,
        pred_transform: Optional[CustomTransforms] = None,
        ref_transforms: Optional[Dict[str, CustomTransforms]] = None,
        forecast_mode: bool = False,
        forecast_index: Optional[pd.DataFrame] = None,
        n_days_forecast: int = 0,
        time_tolerance: pd.Timedelta = pd.Timedelta("12h"),
        file_cache: FileCacheManager=None,
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
        self.pred_connection_params = pred_connection_params
        self.ref_connection_params = ref_connection_params
        self.pred_catalog = pred_catalog
        self.ref_catalogs = ref_catalogs or {}
        self.pred_manager = pred_manager
        self.ref_managers = ref_managers or {}
        self.pred_alias = pred_alias
        self.ref_aliases = ref_aliases or []
        self.batch_size = batch_size
        self.pred_transform = pred_transform
        self.ref_transforms = ref_transforms or {}
        self.forecast_mode = forecast_mode
        self.forecast_index = forecast_index
        self.n_days_forecast = n_days_forecast
        self.time_tolerance = time_tolerance
        self.file_cache = file_cache
        self.pred_coords = pred_catalog.get_global_metadata().get("coord_system", None)
        self.ref_coords = {ref_alias: ref_catalog.get_global_metadata().get("coord_system", None)
                           for ref_alias, ref_catalog in self.ref_catalogs.items()}
        self.aligned_entries = {}
        if not self.forecast_mode:
            self._align_catalogs()

    def __len__(self):
        if self.forecast_mode and self.forecast_index is not None:
            return len(self.forecast_index)
        return len(self.pred_catalog.get_dataframe())

    def __iter__(self):
        return self._generate_batches()


    def _align_catalogs(self):
        """
        Aligne le catalogue de prédiction avec chaque catalogue de référence séparément.
        Si le dataset de référence est un modèle (gridded), on aligne sur les dates.
        Si c'est une observation (is_observation), on ne filtre pas le catalogue de référence.
        Remplit self.aligned_entries[ref_alias] = (pred_entries, ref_entries)
        """
        self.aligned_entries = {}
        pred_df = self.pred_catalog.get_dataframe()
        for ref_alias, ref_catalog in self.ref_catalogs.items():
            if not ref_catalog:
                continue
            ref_is_obs = ref_catalog.get_global_metadata().get("is_observation", False)
            ref_df = ref_catalog.get_dataframe()
            if not ref_is_obs:
                aligned_df = pd.merge(
                    pred_df,
                    ref_df,
                    on=["date_start", "date_end"],
                    suffixes=("_pred", "_ref"),
                    how="inner",
                )
                pred_entries = aligned_df[
                    ["date_start", "date_end"] + [col for col in aligned_df.columns if col.endswith("_pred")]
                ].rename(columns=lambda x: x.replace("_pred", "")).to_dict(orient="records")
                ref_entries = aligned_df[
                    ["date_start", "date_end"] + [col for col in aligned_df.columns if col.endswith("_ref")]
                ].rename(columns=lambda x: x.replace("_ref", "")).to_dict(orient="records")
            else:
                pred_entries = pred_df.to_dict(orient="records")
                ref_entries = UnifiedObservationView(ref_df, self.open_ref, ref_alias, self.time_tolerance)
            self.aligned_entries[ref_alias] = (pred_entries, ref_entries)

    def _find_matching_ref(self, valid_time, ref_alias):
        """Trouve le fichier de référence couvrant valid_time pour ref_alias."""
        ref_df = self.ref_catalogs[ref_alias].get_dataframe()
        match = ref_df[(ref_df["date_start"] <= valid_time) & (ref_df["date_end"] >= valid_time)]
        if not match.empty:
            return match.iloc[0]["path"]
        return None

    def _generate_batches(self) -> Generator[List[Dict[str, Any]], None, None]:
        batch = []
        for _, row in self.forecast_index.iterrows():
            # Vérifier si on a suffisamment de données pour ce forecast
            forecast_reference_time = row["forecast_reference_time"]
            lead_time = row["lead_time"]
            valid_time = row["valid_time"]
            
            # Calculer la fin du forecast complet (dernier lead time)
            max_lead_time = self.n_days_forecast - 1  # 0-indexé
            if hasattr(self, 'lead_time_unit') and self.lead_time_unit == "hours":
                forecast_end_time = forecast_reference_time + pd.Timedelta(hours=max_lead_time)
            else:
                forecast_end_time = forecast_reference_time + pd.Timedelta(days=max_lead_time)
            
            # Vérifier que le forecast complet est dans la plage de données disponibles
            for ref_alias in self.ref_aliases:
                # ref_alias = self.ref_aliases[0] if self.ref_aliases else None
                if ref_alias:
                    ref_catalog = self.ref_catalogs[ref_alias]
                    ref_df = ref_catalog.get_dataframe()
                    
                    # Trouver la date de fin maximale disponible dans les données de référence
                    max_available_date = ref_df["date_end"].max()
                    
                    # Si la fin du forecast dépasse les données disponibles, ignorer cette entrée
                    if forecast_end_time > max_available_date:
                        logger.debug(f"Skipping forecast starting at {forecast_reference_time}: "
                                f"forecast ends at {forecast_end_time} but data only available until {max_available_date}")
                        continue
                
                entry = {
                    "forecast_reference_time": forecast_reference_time,
                    "lead_time": lead_time,
                    "valid_time": valid_time,
                    "pred_data": row["file"],
                    "ref_data": None,
                    "ref_alias": ref_alias,
                    "pred_coords": self.pred_coords,
                    "ref_coords": self.ref_coords[ref_alias] if ref_alias else None,
                }
                
                if ref_alias:
                    ref_catalog = self.ref_catalogs[ref_alias]
                    coord_system = ref_catalog.get_global_metadata().get("coord_system")
                    is_observation = coord_system.is_observation_dataset() if coord_system else False
                    
                    if is_observation:
                        # Logique observation : filtrer le catalogue d'observation sur l'intervalle temporel du forecast_index
                        ref_entries = UnifiedObservationView(
                            ref_catalog.get_dataframe(), self.open_ref, ref_alias, self.time_tolerance
                        )
                        obs_time_interval = (valid_time, valid_time)
                        filtered_obs = ref_entries.open_concat_in_time(obs_time_interval)
                        if filtered_obs is None:
                            continue
                        entry["ref_data"] = filtered_obs
                        entry["ref_is_observation"] = True
                    else:
                        # Logique gridded : associer le fichier de référence couvrant valid_time
                        ref_path = self._find_matching_ref(valid_time, ref_alias)
                        if ref_path is None:
                            logger.debug(f"No reference data found for valid_time {valid_time}")
                            continue
                        entry["ref_data"] = ref_path
                        entry["ref_is_observation"] = False
                
                batch.append(entry)
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch



    def open_pred(self, pred_entry: str) -> xr.Dataset:
        pred_data = self.pred_manager.open(pred_entry, self.file_cache)
        return pred_data



    def open_ref(self, ref_entry: str, ref_alias: str) -> xr.Dataset:
        ref_data = self.ref_managers[ref_alias].open(ref_entry, self.file_cache)
        return ref_data


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