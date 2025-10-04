"""
Data preprocessor for exoplanet light curves.
Handles cleaning, normalization, and preparation of light curve data for ML models.
"""

import numpy as np
import pandas as pd
import lightkurve as lk
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
from typing import List, Tuple, Optional, Dict, Union
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings('ignore')


class LightCurvePreprocessor:
    """
    Preprocesses light curve data for machine learning models.
    """
    
    def __init__(self, target_length: int = 2048, normalize_method: str = 'standard'):
        """
        Initialize the preprocessor.
        
        Args:
            target_length: Target length for standardized light curves
            normalize_method: Normalization method ('standard', 'minmax', or 'median')
        """
        self.target_length = target_length
        self.normalize_method = normalize_method
        self.scaler = None
        
    def clean_light_curve(self, lc: lk.LightCurve, sigma_clip: float = 5.0) -> lk.LightCurve:
        """
        Clean a single light curve by removing outliers and bad data points.
        
        Args:
            lc: Input light curve
            sigma_clip: Sigma threshold for outlier clipping
            
        Returns:
            Cleaned light curve
        """
        if lc is None or len(lc.flux) == 0:
            return None
        
        # Remove NaN values
        lc = lc.remove_nans()
        
        # Remove outliers using sigma clipping
        lc = lc.remove_outliers(sigma=sigma_clip)
        
        # Remove data points with quality flags (for Kepler/TESS data)
        if hasattr(lc, 'quality') and lc.quality is not None:
            lc = lc[lc.quality == 0]
        
        return lc
    
    def detrend_light_curve(self, lc: lk.LightCurve, method: str = 'savgol', 
                           window_length: Optional[int] = None) -> lk.LightCurve:
        """
        Remove long-term trends from light curve.
        
        Args:
            lc: Input light curve
            method: Detrending method ('savgol', 'median', or 'biweight')
            window_length: Window length for detrending (auto-calculated if None)
            
        Returns:
            Detrended light curve
        """
        if lc is None or len(lc.flux) == 0:
            return None
        
        try:
            if method == 'savgol':
                # Use Savitzky-Golay filter
                if window_length is None:
                    window_length = min(101, len(lc.flux) // 10)
                    window_length = window_length if window_length % 2 == 1 else window_length + 1
                
                lc = lc.flatten(window_length=window_length, method='savgol')
                
            elif method == 'median':
                # Use median filter
                if window_length is None:
                    window_length = 49
                lc = lc.flatten(window_length=window_length, method='median')
                
            elif method == 'biweight':
                # Use biweight location
                if window_length is None:
                    window_length = 21
                lc = lc.flatten(window_length=window_length, method='biweight')
                
        except Exception as e:
            print(f"Detrending failed: {e}")
            return lc
        
        return lc
    
    def normalize_flux(self, flux: np.ndarray, method: str = None) -> np.ndarray:
        """
        Normalize flux values.
        
        Args:
            flux: Input flux array
            method: Normalization method (uses self.normalize_method if None)
            
        Returns:
            Normalized flux array
        """
        if method is None:
            method = self.normalize_method
        
        flux = np.array(flux)
        
        if method == 'standard':
            # Standardize to zero mean, unit variance
            flux = (flux - np.mean(flux)) / np.std(flux)
            
        elif method == 'minmax':
            # Scale to [0, 1] range
            flux_min, flux_max = np.min(flux), np.max(flux)
            flux = (flux - flux_min) / (flux_max - flux_min)
            
        elif method == 'median':
            # Normalize by median
            median_flux = np.median(flux)
            flux = flux / median_flux - 1.0
            
        return flux
    
    def resample_light_curve(self, time: np.ndarray, flux: np.ndarray, 
                           target_length: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample light curve to a fixed length.
        
        Args:
            time: Time array
            flux: Flux array
            target_length: Target length (uses self.target_length if None)
            
        Returns:
            Tuple of (resampled_time, resampled_flux)
        """
        if target_length is None:
            target_length = self.target_length
        
        if len(flux) <= target_length:
            # Pad with zeros if too short
            padded_flux = np.zeros(target_length)
            padded_time = np.linspace(time[0], time[-1], target_length)
            padded_flux[:len(flux)] = flux
            return padded_time, padded_flux
        
        # Interpolate to target length
        f_interp = interp1d(time, flux, kind='linear', fill_value='extrapolate')
        new_time = np.linspace(time[0], time[-1], target_length)
        new_flux = f_interp(new_time)
        
        return new_time, new_flux
    
    def extract_features(self, flux: np.ndarray, time: np.ndarray = None) -> Dict[str, float]:
        """
        Extract statistical and astronomical features from light curve.
        
        Args:
            flux: Flux array
            time: Time array (optional)
            
        Returns:
            Dictionary of extracted features
        """
        features = {}
        
        # Basic statistical features
        features['mean'] = np.mean(flux)
        features['std'] = np.std(flux)
        features['var'] = np.var(flux)
        features['skewness'] = self._skewness(flux)
        features['kurtosis'] = self._kurtosis(flux)
        features['min'] = np.min(flux)
        features['max'] = np.max(flux)
        features['range'] = features['max'] - features['min']
        features['median'] = np.median(flux)
        features['mad'] = np.median(np.abs(flux - features['median']))
        
        # Percentile features
        percentiles = [5, 25, 75, 95]
        for p in percentiles:
            features[f'p{p}'] = np.percentile(flux, p)
        
        # Signal-to-noise ratio
        features['snr'] = features['mean'] / features['std'] if features['std'] > 0 else 0
        
        # Autocorrelation at lag 1
        if len(flux) > 1:
            features['autocorr_1'] = np.corrcoef(flux[:-1], flux[1:])[0, 1]
        else:
            features['autocorr_1'] = 0
        
        # Period-related features (if time is provided)
        if time is not None and len(time) > 10:
            try:
                # Simple periodogram
                freqs, power = signal.periodogram(flux, fs=1.0/np.median(np.diff(time)))
                peak_freq_idx = np.argmax(power[1:]) + 1  # Skip DC component
                features['dominant_period'] = 1.0 / freqs[peak_freq_idx] if freqs[peak_freq_idx] > 0 else 0
                features['power_ratio'] = power[peak_freq_idx] / np.sum(power)
            except:
                features['dominant_period'] = 0
                features['power_ratio'] = 0
        
        return features
    
    def _skewness(self, x: np.ndarray) -> float:
        """Calculate skewness of array."""
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x == 0:
            return 0
        return np.mean(((x - mean_x) / std_x) ** 3)
    
    def _kurtosis(self, x: np.ndarray) -> float:
        """Calculate kurtosis of array."""
        mean_x = np.mean(x)
        std_x = np.std(x)
        if std_x == 0:
            return 0
        return np.mean(((x - mean_x) / std_x) ** 4) - 3
    
    def process_light_curve_collection(self, lc_collection: lk.LightCurveCollection, 
                                     stitch: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a collection of light curves into a single, clean time series.
        
        Args:
            lc_collection: Collection of light curves
            stitch: Whether to stitch multiple quarters/sectors together
            
        Returns:
            Tuple of (processed_time, processed_flux)
        """
        if lc_collection is None or len(lc_collection) == 0:
            return None, None
        
        processed_lcs = []
        
        for lc in lc_collection:
            # Clean individual light curve
            clean_lc = self.clean_light_curve(lc)
            if clean_lc is not None and len(clean_lc.flux) > 0:
                # Detrend
                detrended_lc = self.detrend_light_curve(clean_lc)
                if detrended_lc is not None:
                    processed_lcs.append(detrended_lc)
        
        if not processed_lcs:
            return None, None
        
        if stitch and len(processed_lcs) > 1:
            # Stitch multiple light curves together
            try:
                stitched_lc = processed_lcs[0].stitch(processed_lcs[1:])
                time = stitched_lc.time.value
                flux = stitched_lc.flux.value
            except:
                # Fallback: concatenate manually
                times = []
                fluxes = []
                for lc in processed_lcs:
                    times.extend(lc.time.value)
                    fluxes.extend(lc.flux.value)
                time = np.array(times)
                flux = np.array(fluxes)
        else:
            # Use the first (or only) light curve
            lc = processed_lcs[0]
            time = lc.time.value
            flux = lc.flux.value
        
        # Remove any remaining NaN values
        valid_idx = np.isfinite(flux) & np.isfinite(time)
        time = time[valid_idx]
        flux = flux[valid_idx]
        
        if len(flux) == 0:
            return None, None
        
        # Normalize flux
        flux = self.normalize_flux(flux)
        
        # Resample to target length
        time, flux = self.resample_light_curve(time, flux)
        
        return time, flux
    
    def create_training_data(self, light_curves: Dict[str, any], labels: List[int]) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Create training data from light curve collections.
        
        Args:
            light_curves: Dictionary mapping target names to light curve collections
            labels: List of labels (1 for planet, 0 for false positive)
            
        Returns:
            Tuple of (flux_data, labels_array, features_list)
        """
        target_names = list(light_curves.keys())
        
        flux_data = []
        valid_labels = []
        features_list = []
        
        print("Processing light curves for training data...")
        
        for i, target in enumerate(tqdm(target_names)):
            if i >= len(labels):
                break
                
            lc_collection = light_curves[target]
            time, flux = self.process_light_curve_collection(lc_collection)
            
            if time is not None and flux is not None:
                flux_data.append(flux)
                valid_labels.append(labels[i])
                
                # Extract features
                features = self.extract_features(flux, time)
                features['target'] = target
                features_list.append(features)
        
        flux_data = np.array(flux_data)
        valid_labels = np.array(valid_labels)
        
        print(f"Created training data: {len(flux_data)} light curves")
        print(f"Class distribution: {np.sum(valid_labels)} planets, {len(valid_labels) - np.sum(valid_labels)} false positives")
        
        return flux_data, valid_labels, features_list
    
    def plot_sample_light_curves(self, flux_data: np.ndarray, labels: np.ndarray, 
                                n_samples: int = 4, save_path: str = None):
        """
        Plot sample light curves for visualization.
        
        Args:
            flux_data: Array of light curve flux data
            labels: Array of labels
            n_samples: Number of samples to plot per class
            save_path: Path to save the plot (optional)
        """
        fig, axes = plt.subplots(2, n_samples, figsize=(15, 8))
        
        # Plot confirmed planets
        planet_indices = np.where(labels == 1)[0][:n_samples]
        for i, idx in enumerate(planet_indices):
            axes[0, i].plot(flux_data[idx])
            axes[0, i].set_title(f'Confirmed Planet {idx}')
            axes[0, i].set_ylabel('Normalized Flux')
        
        # Plot false positives
        fp_indices = np.where(labels == 0)[0][:n_samples]
        for i, idx in enumerate(fp_indices):
            axes[1, i].plot(flux_data[idx])
            axes[1, i].set_title(f'False Positive {idx}')
            axes[1, i].set_ylabel('Normalized Flux')
            axes[1, i].set_xlabel('Time (samples)')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """
    Example usage of the LightCurvePreprocessor.
    """
    # This would typically be called after data loading
    print("LightCurvePreprocessor ready for use!")


if __name__ == "__main__":
    main()


