import numpy as np
from tqdm import tqdm
import argparse
import random
import torch
from IsoScore import IsoScore

def predict_distances(doc_embs, code_embs):
    all_distances = []
    progress_bar = tqdm(range(len(doc_embs)), desc="Compute Distances")
    for idx in progress_bar:
        # Sample Query
        query = doc_embs[idx]
        query = np.expand_dims(query, axis=0)
        
        # Sample Positive Code
        positive_code = code_embs[idx]
        positive_code = np.expand_dims(positive_code, axis=0)
        
        # Calculate Cosine distance for positive code
        positive_distance = calculate_cosine_distance(query, positive_code)
        
        # Calculate distances for all negative codes
        negative_distances = []
        for neg_idx, negative_code in enumerate(code_embs):
            if neg_idx != idx:  # Exclude the positive pair
                negative_code = np.expand_dims(negative_code, axis=0)
                distance = calculate_cosine_distance(query, negative_code)
                negative_distances.append(distance)
        
        # Combine positive and negative distances
        distances = [positive_distance] + negative_distances
        all_distances.append(distances)

    return all_distances

def calculate_mrr_from_distances(distances_lists):
    ranks = []
    for batch_idx, predictions in enumerate(distances_lists):
        correct_score = predictions[0]
        scores = np.array([prediction for prediction in predictions])
        rank = np.sum(scores <= correct_score)
        ranks.append(rank)
    mean_mrr = np.mean(1.0 / np.array(ranks))

    return mean_mrr

def calculate_cosine_distance(code_key, query):
    code_key = np.squeeze(code_key)
    query = np.squeeze(query)
    
    cosine_similarity = (np.dot(query, code_key) /
                         (np.linalg.norm(query) * np.linalg.norm(code_key)))
    # Compute cosine distance
    return 1 - cosine_similarity
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# Deprecated whitening function with different whitening methods: PCA-, ZCA- cov/cor, Cholesky    
def whitening(X, method='zca'):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension [M x N ]
        method: Whitening method. Must be one of 'zca', 'zca_cor', 'pca',
                'pca_cor', or 'cholesky'.
    """
    X = X.reshape((-1, np.prod(X.shape[1:])))
    X_centered = X - np.mean(X, axis=0)
    Sigma = np.cov(X_centered, rowvar=True) # cov matrix
    P = np.corrcoef(X_centered, rowvar=True) # corr matrix
    W = None
    
    if method in ['zca', 'pca', 'cholesky']:
        U, Lambda, _ = np.linalg.svd(Sigma) # U = eigenvectors [M x M] Lambda = eigenvalues [M x 1]
        if method == 'zca':
            W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T))
        elif method =='pca':
            W = np.dot(np.diag(1.0 / np.sqrt(Lambda + 1e-5)), U.T)
        elif method == 'cholesky':
            W = np.linalg.cholesky(np.dot(U, np.dot(np.diag(1.0 / (Lambda + 1e-5)), U.T))).T
    elif method in ['zca_cor', 'pca_cor']:
        V_sqrt = np.diag(np.std(X, axis=1))
        #P = np.dot(np.dot(np.linalg.inv(V_sqrt), Sigma), np.linalg.inv(V_sqrt))
        G, Theta, _ = np.linalg.svd(P)
        if method == 'zca_cor':
            W = np.dot(np.dot(G, np.dot(np.diag(1.0 / np.sqrt(Theta + 1e-5)), G.T)), np.linalg.inv(V_sqrt))
        elif method == 'pca_cor':
            W = np.dot(np.dot(np.diag(1.0/np.sqrt(Theta + 1e-5)), G.T), np.linalg.inv(V_sqrt))
    else:
        raise Exception('Whitening method not found.')
    print("wshape ",W.shape)
    return W, np.dot(W.T, X_centered), method
    
def zca_features_combined(X, Y, epsilon):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Combined (both modality) Input data matrix with data examples along the first dimension [M x N ] for stacked whitening
        Y:      Input data matrix (one modality) with data examples along the first dimension [M x N ]
        epsilon: Eigenvalue regularizer
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    Y_mean = np.mean(Y, axis=0)
    Y_centered = Y - Y_mean
    Sigma = np.cov(X_centered, rowvar=False) # cov matrix
    W = None
    U, Lambda, _ = np.linalg.svd(Sigma) # U = eigenvectors [M x M] Lambda = eigenvalues [M x 1]
    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + epsilon)), U.T))
    print("wshape ",W.shape)
    return np.dot(Y_centered, W.T)
    
def zca_features(X, epsilon):
    """
    Whitens the input matrix X using specified whitening method.
    Inputs:
        X:      Input data matrix with data examples along the first dimension [M x N ]
        epsilon: Eigenvalue regularizer
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    #print(X_centered.shape)
    Sigma = np.cov(X_centered, rowvar=False) # cov matrix
    W = None
    U, Lambda, _ = np.linalg.svd(Sigma) # U = eigenvectors [M x M] Lambda = eigenvalues [M x 1]
    W = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(Lambda + epsilon)), U.T))
    print("wshape ",W.shape)
    return np.dot(X_centered, W.T)
    
def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default='codebert', type=str, required=False, help="Type of model: codebert, codet5p, codellama")
    parser.add_argument("--lang", default='ruby', type=str, required=False, help="Language of the code")
    parser.add_argument("--epsilon", default=None, type=float, required=False, help="Level of epsilon used in whitening")
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument("--is_finetuned", action="store_true", help="use finetuned codebert")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    if args.is_finetuned:
        saved_code_embs = np.load(f'./embeddings/code_embs_{args.model}_{args.lang}_finetuned.npy')
    else:
        saved_code_embs = np.load(f'./embeddings/code_embs_{args.model}_{args.lang}.npy')
        
    if args.is_finetuned:    
        saved_doc_embs = np.load(f'./embeddings/doc_embs_{args.model}_{args.lang}_finetuned.npy')
    else:
        saved_doc_embs = np.load(f'./embeddings/doc_embs_{args.model}_{args.lang}.npy')
        
    #stacked_emb = np.vstack((saved_code_embs, saved_doc_embs)) # for combined
    
    if args.epsilon is not None:
        saved_code_embs = zca_features(saved_code_embs, args.epsilon)
        saved_doc_embs = zca_features(saved_doc_embs, args.epsilon)
        #saved_code_embs = zca_features_combined(stacked_emb, saved_code_embs, args.epsilon)
        #saved_doc_embs = zca_features_combined(stacked_emb, saved_doc_embs, args.epsilon)
    
    code_iso = IsoScore.IsoScore(saved_code_embs)
    doc_iso = IsoScore.IsoScore(saved_doc_embs)
    
    all_distances = predict_distances(saved_doc_embs, saved_code_embs)
    mrr = calculate_mrr_from_distances(all_distances)
    
    if args.is_finetuned:
        log_path = f'./logs/{args.model}_{args.lang}_{args.epsilon}_finetuned.npy.txt'
    else:
        log_path = f'./logs/{args.model}_{args.lang}_{args.epsilon}.npy.txt'
    with open(log_path, "w") as file:
        file.write("MRR: "+str(mrr)+"\nCodeIso: "+str(code_iso)+"\nDocIso: "+str(doc_iso))
    
    

if __name__ == '__main__':
    main()