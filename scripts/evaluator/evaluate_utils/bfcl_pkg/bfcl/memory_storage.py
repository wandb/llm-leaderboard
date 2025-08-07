"""
Memory-based data storage for BFCL evaluation to avoid JSONDecodeError
"""
from typing import Dict, List, Any
import threading


class BFCLMemoryStorage:
    """
    Singleton class to store BFCL evaluation results in memory
    This avoids JSONDecodeError issues while maintaining logging functionality
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the storage"""
        self.results_by_model: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.data_lock = threading.Lock()
    
    def store_result(self, model_name: str, test_category: str, entry: Dict[str, Any]):
        """
        Store a single result entry in memory
        
        Args:
            model_name: Name of the model (e.g., "command-a-03-2025-FC")
            test_category: Test category (e.g., "simple", "parallel", etc.)
            entry: Result entry dictionary with 'id', 'result', etc.
        """
        with self.data_lock:
            if model_name not in self.results_by_model:
                self.results_by_model[model_name] = {}
            
            if test_category not in self.results_by_model[model_name]:
                self.results_by_model[model_name][test_category] = []
            
            # Update existing entry or append new one
            entries = self.results_by_model[model_name][test_category]
            entry_id = entry.get('id')
            
            # Find and update existing entry, or append new one
            updated = False
            for i, existing_entry in enumerate(entries):
                if existing_entry.get('id') == entry_id:
                    entries[i] = entry
                    updated = True
                    break
            
            if not updated:
                entries.append(entry)
    
    def store_results_batch(self, model_name: str, test_category: str, entries: List[Dict[str, Any]]):
        """
        Store multiple result entries in batch
        
        Args:
            model_name: Name of the model
            test_category: Test category
            entries: List of result entry dictionaries
        """
        for entry in entries:
            self.store_result(model_name, test_category, entry)
    
    def get_results(self, model_name: str, test_category: str) -> List[Dict[str, Any]]:
        """
        Get results for a specific model and test category
        
        Args:
            model_name: Name of the model
            test_category: Test category
            
        Returns:
            List of result entries, empty list if not found
        """
        with self.data_lock:
            return self.results_by_model.get(model_name, {}).get(test_category, []).copy()
    
    def get_all_results(self, model_name: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get all results for a specific model
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary mapping test_category to list of entries
        """
        with self.data_lock:
            model_results = self.results_by_model.get(model_name, {})
            return {category: entries.copy() for category, entries in model_results.items()}
    
    def clear_model_results(self, model_name: str):
        """
        Clear all results for a specific model
        
        Args:
            model_name: Name of the model to clear
        """
        with self.data_lock:
            if model_name in self.results_by_model:
                del self.results_by_model[model_name]
    
    def clear_all(self):
        """Clear all stored results"""
        with self.data_lock:
            self.results_by_model.clear()
    
    def get_models(self) -> List[str]:
        """Get list of models with stored results"""
        with self.data_lock:
            return list(self.results_by_model.keys())
    
    def get_test_categories(self, model_name: str) -> List[str]:
        """Get list of test categories for a specific model"""
        with self.data_lock:
            return list(self.results_by_model.get(model_name, {}).keys())
    
    def has_results(self, model_name: str, test_category: str = None) -> bool:
        """
        Check if results exist for model and optionally test category
        
        Args:
            model_name: Name of the model
            test_category: Test category (optional)
            
        Returns:
            True if results exist, False otherwise
        """
        with self.data_lock:
            if model_name not in self.results_by_model:
                return False
            
            if test_category is None:
                return len(self.results_by_model[model_name]) > 0
            
            return test_category in self.results_by_model[model_name]