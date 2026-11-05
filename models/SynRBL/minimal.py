"""
Run SynRBL on one example reaction.
"""

from tty import CC
from synrbl import RebalanceConfig, ReactionRebalancer
from synrbl.SynUtils import normalize_smiles, wc_similarity


def main():
    # Initialize SynRBL
    
    data = [{'id':1, 'rxn':'CCCN(CCC)C(=O)c1cccc(C(=O)NC(Cc2ccccc2)C(O)([PH2]=O)C(CC(=O)OC)C(=O)Nc2ccccc2)c1>>CCCN(CCC)C(=O)c1cccc(C(=O)NC(Cc2ccccc2)C(O)([PH2]=O)C(CC(=O)O)C(=O)Nc2ccccc2)c1'}, 
            {'id':2, 'rxn':'CC(=O)OC(C)=O.O=C(O)c1ccc(Cc2ccc(O)cc2)cn1>>CC(=O)Oc1ccc(Cc2ccc(C(=O)O)nc2)cc1'},
            {'id':3, 'rxn':'CC(=O)OC(C)=O.O=C(O)c1ccc(Cc2ccc(O)cc2)cn1.c1ccncc1.Cl>>CC(=O)Oc1ccc(Cc2ccc(C(=O)O)nc2)cc1'}]
    
    # id 3 ground truth:
    id_3_gt = "CC(=O)OC(C)=O.O=C(O)c1ccc(Cc2ccc(O)cc2)cn1.c1ccncc1.Cl>>CC(=O)Oc1ccc(Cc2ccc(C(=O)O)nc2)cc1.CC(=O)[O-].c1cc[nH+]cc1.Cl"
    config = RebalanceConfig(reaction_col="rxn", id_col="id", n_jobs=2, batch_size=500,
                            enable_logging=False, use_default_reduction=True)
    rebalancer = ReactionRebalancer(config=config, user_logger=None)
    result = rebalancer.rebalance(data, keep_extra=False)
    # Predict with SynRBL

    print("\n=== SynRBL Prediction Results ===")
    print(f"ID: {data[0]['id']}")
    print(result)
    
    #Compute similarity for id 3 
    id_3_pred = result[2]['rxn']
    similarity = wc_similarity(normalize_smiles(id_3_pred), normalize_smiles(id_3_gt))
    print(f"ID 3 similarity: {similarity}")


if __name__ == "__main__":
    main()