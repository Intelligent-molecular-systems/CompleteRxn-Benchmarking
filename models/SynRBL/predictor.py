"""Run SynRBL on a CSV of incomplete reactions.

For the method and software, see https://github.com/TieuLongPhan/SynRBL.
"""

from synrbl import ReactionRebalancer, RebalanceConfig
import pandas as pd
import time
from typing import Union, List, Dict, Optional
import importlib

if importlib.util.find_spec("tqdm") is not None:
    tqdm = importlib.import_module("tqdm").tqdm
else:
    def tqdm(iterable, **kwargs):
        return iterable


class SynRBLPredictor:
    """Run SynRBL in batches and save its completed reactions."""
    
    def __init__(
        self,
        reaction_col: str = 'Reaction',
        id_col: str = 'Unique_identifier',
        n_jobs: int = -1,
        batch_size: int = 128,
        chunk_size: int = 1024,
        enable_logging: bool = False,
        use_default_reduction: bool = True
    ):
        """
        Initialize SynRBL predictor
        
        Args:
            reaction_col: Name of column containing reactions
            id_col: Name of column containing Unique Identifiers
            n_jobs: Number of parallel jobs
            batch_size: Number of reactions per batch
            chunk_size: Number of reactions per chunk for processing large csvs (default: 1024)
            enable_logging: Enable SynRBL logging
            use_default_reduction: Use default reduction rules
        """
        self.config = RebalanceConfig(
            reaction_col=reaction_col,
            id_col=id_col,
            n_jobs=n_jobs,
            batch_size=batch_size,
            enable_logging=enable_logging,
            use_default_reduction=use_default_reduction
        )
        self.rebalancer = ReactionRebalancer(config=self.config, user_logger=None)
        self.reaction_col = reaction_col
        self.id_col = id_col
        self.chunk_size = chunk_size
    
    def predict_single(self, reaction: str, rxn_id: int = 0) -> Dict:
        """
        Balance a single reaction
        
        Args:
            reaction: Reaction SMILES string
            rxn_id: Reaction ID
            
        Returns:
            Dictionary with balanced reaction and metadata
        """
        data = [{self.id_col: rxn_id, self.reaction_col: reaction}]
        result = self.rebalancer.rebalance(data, keep_extra=False)
        return result[0] if result else {}
    
    def predict_batch(self, reactions: Union[List[str], pd.Series, pd.DataFrame]) -> pd.DataFrame:
        """
        Balance a batch of reactions
        
        Args:
            reactions: List/Series of reaction SMILES strings or DataFrame with reaction column
            
        Returns:
            DataFrame with results
        """
        # Convert to list-of-dicts format expected by ReactionRebalancer
        if isinstance(reactions, pd.DataFrame):
            if self.id_col not in reactions.columns:
                reactions = reactions.copy()
                reactions[self.id_col] = range(len(reactions))
            data = reactions.to_dict('records')
        elif isinstance(reactions, (list, pd.Series)):
            if isinstance(reactions, pd.Series):
                reactions = reactions.values
            data = [
                {self.id_col: i, self.reaction_col: rxn}
                for i, rxn in enumerate(reactions)
            ]
        else:
            raise ValueError("reactions must be a list, Series, or DataFrame")

        results = self.rebalancer.rebalance(data, keep_extra=False)
        return pd.DataFrame(results)
    
    # Progress save function to save intermediate results
    def progress_save(self, results: List[Dict], output_file: str):
        """Save raw results from rebalancer to CSV"""
        if not results:
            return
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_file, index=False)


    def predict_from_csv(
        self,
        input_file: str,
        output_file: str,
        reaction_col: Optional[str] = None,
        keep_extra: bool = True,
    ) -> pd.DataFrame:
        """
        Process reactions from CSV file and save results using chunked processing.
        Results are saved incrementally to avoid data loss.
        
        Args:
            input_file: Path to input CSV with reactions
            output_file: Path to output CSV for results
            reaction_col: Column name containing reactions (uses self.reaction_col if None)
            keep_extra: Keep extra columns from input file in output
            chunk_size: Number of reactions to process per batch (default: 64)
        """
        reaction_col = reaction_col or self.reaction_col
        
        # Read input data
        print(f"Reading {input_file}...")
        data = pd.read_csv(input_file, low_memory=False)
        
        # Ensure ID column exists
        if self.id_col not in data.columns:
            data[self.id_col] = range(len(data))
        
        # Ensure reaction column exists
        if reaction_col not in data.columns:
            raise ValueError(f"Column '{reaction_col}' not found in input file")

        # Update config if different reaction column
        if reaction_col != self.reaction_col:
            self.config.reaction_col = reaction_col
            self.rebalancer = ReactionRebalancer(config=self.config, user_logger=None)
        
        print(f"Processing {len(data)} reactions in chunks of {self.chunk_size}...")
        start_time = time.time()

        # Build full record list then chunk records
        records = data.to_dict('records')
        chunks = [records[i:i+self.chunk_size] for i in range(0, len(records), self.chunk_size)]
        
        all_results = []
        
        # Process chunks with progress bar
        for i, chunk in enumerate(tqdm(chunks, total=len(chunks), desc="Processing chunks")):
            try:
                # Process chunk
                output = self.rebalancer.rebalance(chunk, keep_extra=keep_extra)
                if len(output) != len(chunk):
                    raise RuntimeError(
                        f"SynRBL returned {len(output)} rows for a {len(chunk)}-row "
                        "chunk; refusing to save a positionally ambiguous result."
                    )
                for source_row, result_row in zip(chunk, output):
                    expected_id = source_row.get(self.id_col)
                    observed_id = result_row.get(self.id_col)
                    if (
                        not pd.isna(observed_id)
                        and str(observed_id) != str(expected_id)
                    ):
                        raise RuntimeError(
                            f"SynRBL UID mismatch: expected {expected_id!r}, "
                            f"received {observed_id!r}"
                        )
                    # SynRBL 1.0.5 can lose the external ID while restoring
                    # records by standardized reaction string. Output order is
                    # preserved, so restore it after the checks above.
                    result_row[self.id_col] = expected_id
                chunk_df = pd.DataFrame(output)
                
                all_results.append(chunk_df)
                
                # Save progress every 10 chunks
                if (i + 1) % 10 == 0:
                    self._save_progress(all_results, output_file, data)
                    
            except Exception as e:
                print(f"\nError processing chunk {i+1}: {e}")
                if all_results:
                    self._save_progress(all_results, output_file, data)
                raise
        
        # Final save
        results_df = self._save_progress(all_results, output_file, data)
        
        time_elapsed = time.time() - start_time
        print(f"[{time_elapsed//60:.0f}m {time_elapsed%60:.0f}s] Processing complete")
        print(f"Results saved to {output_file}")
        
        return results_df
    
    def _save_progress(self, all_results: List[pd.DataFrame], output_file: str, 
                      original_data: pd.DataFrame) -> pd.DataFrame:
        """Helper to save progress and return combined DataFrame"""
        if not all_results:
            return pd.DataFrame()
        
        # Combine all results
        results_df = pd.concat(all_results, ignore_index=True)
        
        # Add columns from original data using ID column merge
        col_to_add = ["rxn_flower", "rxn_reagents_identified_flower"]
        cols_to_merge = [col for col in col_to_add if col in original_data.columns]
        
        if cols_to_merge and self.id_col in results_df.columns:
            results_df = results_df.merge(
                original_data[[self.id_col] + cols_to_merge],
                on=self.id_col,
                how='left'
            )
        
        # Drop unwanted columns
        cols_to_drop = ['R-id', 'reactants', 'products', 'total_charge_in_products', 
                       'success', 'new_reactants', 'new_products', 'standardized_reactions', 'reactions']
        results_df = results_df.drop(columns=[col for col in cols_to_drop if col in results_df.columns])
        
        # Save
        results_df.to_csv(output_file, index=False)
        
        return results_df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python predictor.py <input_file> <output_file>")
        sys.exit(1)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]


    
    predictor = SynRBLPredictor()
    predictor.predict_from_csv(input_file, output_file, keep_extra=True)
