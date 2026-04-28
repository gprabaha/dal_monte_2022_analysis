import matplotlib.pyplot as plt
import numpy as np
import pickle
import tqdm as tqdm
from sklearn.decomposition import PCA
import warnings
import itertools
warnings.filterwarnings("ignore")
from pca import save_fig


def principal_angles(combinations, num_comps=None, control=False):

    """
        Perform manifold analysis (principle angles and VAF)

        params:
            system: "neural" or "muscle"
            epoch: "delay" or "movement"
    """

    angles_list = []
    control_list = []

    num_comps = 10 if num_comps is None else num_comps
    baseline_dim = 256

    if control == True:
        # Create a random manifold as a control
        random_matrices = np.random.randn(5000, baseline_dim, num_comps)
        random_bases = np.empty(shape=(5000, num_comps, baseline_dim))
        for basis in range(5000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T

    for i, combination in enumerate(combinations):
        
        # ------------------------------------ GET PRINCIPLE ANGLES

        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1]))
        task2_data = combination[1].reshape((-1, combination[1].shape[-1]))

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:num_comps]
        pca2_comps = pca2.components_[:num_comps]

        # Get principle angles
        inner_prod_mat = pca1_comps @ pca2_comps.T # Should be m x m
        U, s, Vh = np.linalg.svd(inner_prod_mat)
        angles = np.degrees(np.arccos(s))
        angles_list.append(angles)

    if control == True:
        # Get principle angles control
        for i in range(5000):
            rand1 = np.random.randint(2, 5000-1)
            rand2 = np.random.randint(0, rand1-1)
            inner_prod_mat = random_bases[rand1] @ random_bases[rand2].T # Should be m x m
            U, s, Vh = np.linalg.svd(inner_prod_mat)
            angles = np.degrees(np.arccos(s))
            control_list.append(angles)
        control_array = np.stack(control_list, axis=0)

        return angles_list, control_array
    
    else:

        return angles_list



def vaf_ratio(combinations, num_comps=None, control=True):

    # Only use two muscle PCs for this task, but use three for the one above

    vaf_ratio_list = []
    vaf_ratio_list_control = []

    num_comps = 12 if num_comps is None else num_comps
    baseline_dim = 256
    percentile = 90

    if control == True:
        # Create a random manifold as a control
        random_matrices = np.random.randn(5000, baseline_dim, num_comps)
        random_bases = np.empty(shape=(5000, num_comps, baseline_dim))
        for basis in range(5000):
            q, _ = np.linalg.qr(random_matrices[basis])
            random_bases[basis] = q.T

    for combination in combinations:
        
        pca1 = PCA()
        pca2 = PCA()

        task1_data = combination[0].reshape((-1, combination[0].shape[-1])).numpy()
        task2_data = combination[1].reshape((-1, combination[1].shape[-1])).numpy()

        pca1.fit(task1_data)
        pca2.fit(task2_data)

        pca1_comps = pca1.components_[:num_comps]
        pca2_comps = pca2.components_[:num_comps]

        #------------------------------------- PUT TO DEVICES 

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        task1_data = torch.tensor(task1_data, dtype=torch.float32).to(device)
        task2_data = torch.tensor(task2_data, dtype=torch.float32).to(device)

        pca1_comps = torch.tensor(pca1_comps, dtype=torch.float32).to(device)
        pca2_comps = torch.tensor(pca2_comps, dtype=torch.float32).to(device)
        """

        #random_bases = torch.tensor(random_bases, dtype=torch.float32).to(device)

        # ------------------------------------ TRUE ACROSS AND WITHIN TASK VAFs

        # Get VAF
        across_task_vaf_task1 = (pca2_comps @ task1_data.T).T.var(axis=0).sum() / task1_data.var(axis=0).sum()
        within_task_vaf_task1 = (pca1_comps @ task1_data.T).T.var(axis=0).sum() / task1_data.var(axis=0).sum()
        ratio_task1 = across_task_vaf_task1 / within_task_vaf_task1
        vaf_ratio_list.append(ratio_task1)

        across_task_vaf_task2 = (pca1_comps @ task2_data.T).T.var(axis=0).sum() / task2_data.var(axis=0).sum()
        within_task_vaf_task2 = (pca2_comps @ task2_data.T).T.var(axis=0).sum() / task2_data.var(axis=0).sum()
        ratio_task2 = across_task_vaf_task2 / within_task_vaf_task2
        vaf_ratio_list.append(ratio_task2)

        if control == True:
            # ------------------------------------ CONTROL ACROSS TASK VAFs

            # Get random VAFs
            across_task_vaf = (random_bases @ task1_data.T).var(axis=2).sum(axis=1) / task1_data.var(axis=0).sum()
            vaf_ratio_list_control.append(np.percentile(across_task_vaf, percentile) / within_task_vaf_task1)

            # Get random VAFs
            across_task_vaf = (random_bases @ task2_data.T).var(axis=2).sum(axis=1) / task2_data.var(axis=0).sum()
            vaf_ratio_list_control.append(np.percentile(across_task_vaf, percentile) / within_task_vaf_task2)
    
    if control == True:
        return vaf_ratio_list, vaf_ratio_list_control
    else:
        return vaf_ratio_list


if __name__ == "__main__":
    
    with open('social_gaze/avg_psth_per_category_and_interactivity.pkl', 'rb') as f:
        ci_data = pickle.load(f)

    with open('social_gaze/avg_psth_per_category.pkl', 'rb') as f:
        c_data = pickle.load(f)
    
    region_cat_dict_ci = {}
    ci_data_groups = ci_data.groupby(["region", "category"], sort=False)
    for group, data in ci_data_groups:
        region_fr = data.to_numpy()[:, -1]
        region_fr = np.stack(region_fr, axis=1)
        region_cat_dict_ci[group] = region_fr

    region_cat_dict_c = {}
    c_data_groups = c_data.groupby(["region", "category"], sort=False)
    for group, data in c_data_groups:
        region_fr = data.to_numpy()[:, -1]
        region_fr = np.stack(region_fr, axis=1)
        region_cat_dict_c[group] = region_fr




    
    # Start with non-interactive cause easier
    regions = ["dmpfc", "ofc", "bla", "accg"]
    region_based_dict = {}
    region_based_dict_labels = {}
    for region in regions:
        region_based_dict[region] = []
        region_based_dict_labels[region] = []
    for group in region_cat_dict_c:
        region_based_dict[group[0]].append(region_cat_dict_c[group])
        region_based_dict_labels[group[0]].append(group[1])
    
    for region in region_based_dict:
        combinations = list(itertools.combinations(region_based_dict[region], 2))
        combinations_labels = list(itertools.combinations(region_based_dict_labels[region], 2))
        angles_list = principal_angles(combinations)
        angles_list = np.array(angles_list)
        for (angle, label) in zip(angles_list, combinations_labels):
            plt.plot(angle, label=label)
        plt.legend()
        save_fig(f"results/manifold/pc_angles_c_{region}.png")







    # now do harder one
    regions = ["dmpfc", "ofc", "bla", "accg"]
    region_based_dict = {}
    region_based_dict_labels = {}
    for region in regions:
        region_based_dict[region] = []
        region_based_dict_labels[region] = []
    for group in region_cat_dict_ci:
        region_based_dict[group[0]].append(region_cat_dict_ci[group])
        region_based_dict_labels[group[0]].append(group[1:])
    
    for region in region_based_dict:
        combinations = list(itertools.combinations(region_based_dict[region], 2))
        combinations_labels = list(itertools.combinations(region_based_dict_labels[region], 2))
        angles_list = principal_angles(combinations)
        angles_list = np.array(angles_list)
        for (angle, label) in zip(angles_list, combinations_labels):
            plt.plot(angle, label=label)
        plt.legend()
        save_fig(f"results/manifold/pc_angles_ci_{region}.png")

    
    