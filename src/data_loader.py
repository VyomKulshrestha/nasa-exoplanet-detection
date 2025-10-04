"""
Data loader for NASA exoplanet datasets using lightkurve.
Handles downloading and initial processing of Kepler, K2, and TESS data.
"""

import numpy as np
import pandas as pd
import lightkurve as lk
from astroquery.nasa_exoplanet_archive import NasaExoplanetArchive
from astropy import units as u
import warnings
from typing import List, Tuple, Optional, Dict
import os
from tqdm import tqdm

warnings.filterwarnings('ignore')


class ExoplanetDataLoader:
    """
    Loads and manages exoplanet data from NASA archives.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory to store downloaded data
        """
        self.data_dir = data_dir
        self.confirmed_planets = None
        self.candidates = None
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(f"{data_dir}/raw", exist_ok=True)
        os.makedirs(f"{data_dir}/processed", exist_ok=True)
    
    def load_confirmed_planets(self) -> pd.DataFrame:
        """
        Load confirmed exoplanets from NASA Exoplanet Archive.
        
        Returns:
            DataFrame with confirmed exoplanet data
        """
        print("Loading confirmed exoplanets from NASA Exoplanet Archive...")
        
        try:
            # Query confirmed planets
            confirmed = NasaExoplanetArchive.query_criteria(
                table="pscomppars", 
                select="pl_name,hostname,pl_orbper,pl_rade,pl_masse,st_teff,st_rad,st_mass"
            )
            
            self.confirmed_planets = confirmed.to_pandas()
            print(f"Loaded {len(self.confirmed_planets)} confirmed exoplanets")
            
            # Save to file
            filepath = f"{self.data_dir}/confirmed_planets.csv"
            self.confirmed_planets.to_csv(filepath, index=False)
            
            return self.confirmed_planets
            
        except Exception as e:
            print(f"Error loading confirmed planets: {e}")
            return pd.DataFrame()
    
    def load_kepler_objects_of_interest(self) -> pd.DataFrame:
        """
        Load Kepler Objects of Interest (KOIs) - candidates and confirmed planets.
        
        Returns:
            DataFrame with KOI data
        """
        print("Loading Kepler Objects of Interest...")
        
        try:
            # Query KOI table
            kois = NasaExoplanetArchive.query_criteria(
                table="cumulative",
                select="kepoi_name,kepid,koi_disposition,koi_period,koi_depth,koi_duration,koi_ror,koi_prad,koi_srad,koi_teq"
            )
            
            self.candidates = kois.to_pandas()
            print(f"Loaded {len(self.candidates)} Kepler Objects of Interest")
            
            # Save to file
            filepath = f"{self.data_dir}/kepler_kois.csv"
            self.candidates.to_csv(filepath, index=False)
            
            return self.candidates
            
        except Exception as e:
            print(f"Error loading KOIs: {e}")
            return pd.DataFrame()
    
    def download_light_curves(self, target_list: List[str], mission: str = "Kepler", 
                            max_targets: int = 100) -> Dict[str, any]:
        """
        Download light curves for a list of targets.
        
        Args:
            target_list: List of target names or IDs
            mission: Mission name ("Kepler", "K2", or "TESS")
            max_targets: Maximum number of targets to download
            
        Returns:
            Dictionary mapping target names to light curve collections
        """
        print(f"Downloading {mission} light curves for {min(len(target_list), max_targets)} targets...")
        
        light_curves = {}
        failed_downloads = []
        
        # Limit the number of targets to avoid overwhelming the system
        targets_to_process = target_list[:max_targets]
        
        for target in tqdm(targets_to_process, desc="Downloading light curves"):
            try:
                # Search for light curves
                search_result = lk.search_lightcurve(target, mission=mission)
                
                if len(search_result) > 0:
                    # Download the light curve collection
                    lc_collection = search_result.download_all()
                    
                    if lc_collection is not None and len(lc_collection) > 0:
                        light_curves[target] = lc_collection
                    else:
                        failed_downloads.append(target)
                else:
                    failed_downloads.append(target)
                    
            except Exception as e:
                print(f"Failed to download {target}: {e}")
                failed_downloads.append(target)
        
        print(f"Successfully downloaded light curves for {len(light_curves)} targets")
        if failed_downloads:
            print(f"Failed to download {len(failed_downloads)} targets")
        
        return light_curves
    
    def get_training_sample(self, n_confirmed: int = 50, n_false_positives: int = 50) -> Tuple[List[str], List[int]]:
        """
        Get a balanced training sample of confirmed planets and false positives.
        
        Args:
            n_confirmed: Number of confirmed planets to include
            n_false_positives: Number of false positives to include
            
        Returns:
            Tuple of (target_list, labels) where labels are 1 for planets, 0 for false positives
        """
        if self.candidates is None:
            self.load_kepler_objects_of_interest()
        
        # Filter confirmed planets (CONFIRMED disposition)
        confirmed = self.candidates[self.candidates['koi_disposition'] == 'CONFIRMED'].copy()
        
        # Filter false positives
        false_positives = self.candidates[self.candidates['koi_disposition'] == 'FALSE POSITIVE'].copy()
        
        # Sample from each group
        confirmed_sample = confirmed.sample(n=min(n_confirmed, len(confirmed)), random_state=42)
        fp_sample = false_positives.sample(n=min(n_false_positives, len(false_positives)), random_state=42)
        
        # Create target lists and labels
        target_list = []
        labels = []
        
        # Add confirmed planets (label = 1)
        for _, row in confirmed_sample.iterrows():
            target_list.append(f"KOI-{row['kepoi_name']}")
            labels.append(1)
        
        # Add false positives (label = 0)
        for _, row in fp_sample.iterrows():
            target_list.append(f"KOI-{row['kepoi_name']}")
            labels.append(0)
        
        print(f"Created training sample: {len(confirmed_sample)} confirmed planets, {len(fp_sample)} false positives")
        
        return target_list, labels
    
    def save_light_curve_data(self, light_curves: Dict[str, any], filename: str):
        """
        Save light curve data to file for later use.
        
        Args:
            light_curves: Dictionary of light curve collections
            filename: Output filename
        """
        import pickle
        
        filepath = f"{self.data_dir}/raw/{filename}"
        with open(filepath, 'wb') as f:
            pickle.dump(light_curves, f)
        
        print(f"Saved light curve data to {filepath}")
    
    def load_light_curve_data(self, filename: str) -> Dict[str, any]:
        """
        Load previously saved light curve data.
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary of light curve collections
        """
        import pickle
        
        filepath = f"{self.data_dir}/raw/{filename}"
        with open(filepath, 'rb') as f:
            light_curves = pickle.load(f)
        
        print(f"Loaded light curve data from {filepath}")
        return light_curves


def main():
    """
    Example usage of the ExoplanetDataLoader.
    """
    # Initialize data loader
    loader = ExoplanetDataLoader()
    
    # Load metadata
    confirmed_planets = loader.load_confirmed_planets()
    kois = loader.load_kepler_objects_of_interest()
    
    # Get training sample
    target_list, labels = loader.get_training_sample(n_confirmed=10, n_false_positives=10)
    
    # Download light curves (small sample for testing)
    light_curves = loader.download_light_curves(target_list[:5], mission="Kepler")
    
    # Save the data
    loader.save_light_curve_data(light_curves, "sample_light_curves.pkl")
    
    print("Data loading complete!")


if __name__ == "__main__":
    main()


