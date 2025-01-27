import seaborn as sns # 0.13.2
import matplotlib.pyplot as plt # 3.8.4
import numpy as np  # 1.23.2
import pandas as pd # 2.2.2
import os
import pickle
import shap as shap
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import  matthews_corrcoef,precision_score,recall_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from scipy.stats import loguniform, uniform
from sklearn.metrics import precision_score, recall_score, f1_score
import os
import pickle
from sklearn.metrics import auc
from matplotlib.cm import get_cmap

def map_feature_to_colors(feature, palette="Set2"):
    """
    Map feature values to colors using a specified categorical palette.
    
    Parameters:
        feature (pd.Series): Categorical feature to map.
        palette (str): Name of the Seaborn categorical palette (e.g., "Set2").
    
    Returns:
        dict: A dictionary mapping each unique value in the feature to a color.
    """
    # Get unique values of the feature
    unique_values = feature.unique()
    
    # Generate the palette with the same number of colors as unique values
    colors = sns.color_palette(palette, len(unique_values))
    
    # Create and return the mapping dictionary
    return dict(zip(unique_values, colors))

def plot_distribution(DATA):
    """
    Generate distribution plots for all columns in the dataset.

    Parameters:
        DATA (pd.DataFrame): Input DataFrame containing categorical and numeric features.

    Plots:
        - For categorical columns: Bar plots showing the count of each category.
        - For numeric columns: Violin plots showing the data distribution, with the `mean` visually represented.

    Grid Layout:
        - Plots are arranged in a grid layout with two columns.
        - Unused subplots are hidden if the number of columns does not fill the grid.

    Returns:
        None: The function displays the plots but does not return any values.
    """
    # Detect categorical and numeric columns
    categorical_cols = DATA.select_dtypes(include=["object", "category"]).columns
    numeric_cols = DATA.select_dtypes(include=["number"]).columns

    # Initialize grid layout
    n_cols = 2  # Number of columns in the grid
    n_rows = int(np.ceil(len(DATA.columns) / n_cols))  # Calculate rows based on number of columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 2 * n_rows))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Loop through each column and create the appropriate plot
    for idx, col in enumerate(DATA.columns):
        ax = axes[idx]  # Get the axis to plot on

        # Categorical columns: Bar plots
        if col in categorical_cols:
            sns.countplot(data=DATA, y=col, palette=map_feature_to_colors(DATA[col]), ax=ax, hue=col, saturation=1)
            ax.set_title(col)
            ax.set_ylabel("")
            

        # Numeric columns: Violin plots
        elif col in numeric_cols:
            sns.violinplot(data=DATA, y=col, ax=ax, color="lightgrey")
            ax.set_title(col)
            

        # Hide x-axis for numeric plots (only used for categorical)
        #if col in numeric_cols:
        ax.set_xlabel("")
        

    # Hide any unused subplots in the grid
    for idx in range(len(DATA.columns), len(axes)):
        fig.delaxes(axes[idx])

    # Adjust layout and spacing
    sns.despine()
    plt.tight_layout()
    plt.show()
    
def stratified_plots(data, stratify_col="Recurred"):
    """
    Generate stratified plots for categorical and numeric columns by a specified column.
    Percentage stacked bar plots for categorical features and violin plots for numeric features.
    
    Parameters:
        data (pd.DataFrame): Input DataFrame.
        stratify_col (str): Column name by which to stratify (e.g., "Recurred").
    """
    # Check if the stratify column exists
    if stratify_col not in data.columns:
        raise ValueError(f"{stratify_col} column not found in the dataset.")
    
    # Detect categorical and numeric columns
    categorical_cols = data.select_dtypes(include=["object", "category"]).columns
    numeric_cols = data.select_dtypes(include=["number"]).columns
    
    # Exclude the stratify column from categorical/numeric columns
    categorical_cols = [col for col in categorical_cols if col != stratify_col]
    numeric_cols = [col for col in numeric_cols if col != stratify_col]

    # Combine columns to iterate through all
    all_cols = numeric_cols + categorical_cols
    n_cols = 2  # Number of columns in the grid
    n_rows = int(np.ceil(len(all_cols) / n_cols))  # Calculate rows based on number of columns
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows))
    
    # Flatten axes for easy iteration
    axes = axes.flatten()
    
    # Loop through each column and create the appropriate plot
    for idx, col in enumerate(all_cols):
        ax = axes[idx]  # Get the axis to plot on
        
        # Categorical columns: Percentage stacked bar plots
        if col in categorical_cols:
            # Map feature values to colors
            color_mapping = map_feature_to_colors(data[col], palette="Set2")
            
            # Calculate percentages
            grouped = data.groupby([stratify_col, col], observed=False).size().unstack(fill_value=0)
            percentages = grouped.div(grouped.sum(axis=1), axis=0) * 100
            
            # Reorder colors based on column categories
            category_order = grouped.columns
            colors = [color_mapping[category] for category in category_order]
            
            # Plot percentages with specified colors
            percentages.plot(kind="barh", stacked=True, ax=ax, color=colors)
            ax.set_title(col)
            ax.set_ylabel("")
            ax.set_xlabel("")
            # Set x-ticks to 0% and 100% only
            ax.set_xticks([0, 100])
            ax.set_xticklabels(["0%", "100%"])
            ax.grid(axis="y", visible=False)  # Hides horizontal gridlines

            ax.legend(title=col, bbox_to_anchor=(1.05, 1), loc="upper left")

        # Numeric columns: Violin plots
        elif col in numeric_cols:
            if stratify_col == "Recurred":
                hue_order = ["No", "Yes"]
            else:
                hue_order = None
            sns.violinplot(
                data=data,
                x=col,
                y=stratify_col,
                order=hue_order,
                ax=ax,
                inner=None,
                linewidth=1,
                color="lightgrey",
                cut=1,
            )
            sns.swarmplot(
                data=data,
                x=col,
                y=stratify_col,
                order=hue_order,
                color="grey",
                ax=ax,
                size=2,
            )

            ax.set_title(col)
            ax.set_ylabel("")
            ax.set_xlabel("")
        
    # Hide any unused subplots in the grid
    for idx in range(len(all_cols), len(axes)):
        fig.delaxes(axes[idx])
    
    # Adjust layout and spacing
    sns.despine()
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Add space for legends
    plt.show()
    

def save_or_import_object(obj, filename, import_existing):
    """
    Saves or imports a Python object based on the specified file and flag.
    
    Parameters:
    - obj: The object to save or overwrite.
    - filename: Name of the file to save to or load from.
    - import_existing: Boolean flag. If True, imports the object if the file exists. If False, overwrites the file.
    
    Returns:
    - The imported object if `import_existing` is True and the file exists, otherwise None.
    """
    if import_existing and os.path.exists(filename):
        with open(filename, 'rb') as f:
            imported_obj = pickle.load(f)
            print(f"Imported results from '{filename}'.")
            return imported_obj
    else:
        with open(filename, 'wb') as f:
            pickle.dump(obj, f)
            if os.path.exists(filename) and not import_existing:
                print(f"File '{filename}' overwritten with new data.")
            else:
                print(f"{obj} saved to '{filename}'.")
        return obj
    

def grid_barplot_scores(data, metrics, color_dict, n_cols=2):
    """
    Plots a grid of horizontal bar plots, one for each metric.
    
    Args:
        data (pd.DataFrame): DataFrame containing model scores.
        metrics (list): List of metric names to plot.
        color_dict (dict): Dictionary mapping metrics to colors.
        n_cols (int): Number of columns in the grid layout.
    """
    n_metrics = len(metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Compute number of rows

    y_labels = data['Model']
    y = np.arange(len(y_labels))  # the label locations
    height = 0.6  # the height of the bars
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows), 
                            constrained_layout=True, )
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        mean_col = f'mean_{metric}'  # Mean column for bar width
        metric_values_col = metric  # Column containing 5-fold values

        # Plot the bar for the mean
        ax.barh(y, data[mean_col], height, label=metric.capitalize(), 
                color=color_dict[metric], alpha=1)

        # Plot dots for each fold value
        for j, model in enumerate(data['Model']):
            fold_values = data.loc[data['Model'] == model, metric_values_col].values[0]
            ax.scatter(fold_values, [y[j]] * len(fold_values), 
                    color='black',  s=10)

        # Customize subplot
        ax.set_title(f'{metric.capitalize()}')
        ax.set_xlim(0.7, 1.01)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        sns.despine()

        # Only show y-ticks on the first plot
        if i  == 0:
            ax.set_yticks(y)
            ax.set_yticklabels(y_labels)
        else:
            ax.set_yticks([])

        ax.set_xticks([0.7,0.8,0.9,1])
        ax.set_xticklabels(['0.7','0.8','0.9','1'])
        ax.grid(False,axis='y')
    # Remove unused subplots
    for i in range(len(metrics), len(axes)):
        fig.delaxes(axes[i])

    
    plt.show()
    
def plot_curves_by_fold(df, curve_type='roc'):
    """
    Plots ROC or PRC curves for each fold of cross-validation, one plot per model in a grid.

    Args:
        df: Pandas DataFrame containing the data.
        curve_type: 'roc' for ROC curves, 'prc' for PRC curves.
    """

    if curve_type not in ['roc', 'prc']:
        raise ValueError("curve_type must be 'roc' or 'prc'")

    curve_label = 'ROC' if curve_type == 'roc' else 'PRC'
    curves_col = f'{curve_type}_curves'

    num_models = len(df)
    n_cols = 3
    n_rows = (num_models + n_cols - 1) // n_cols


    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    fig.suptitle(f'{curve_label} curves (5 folds)',fontsize=22)
    axes = axes.flatten()

    for i, (index, row) in enumerate(df.iterrows()):
        ax = axes[i]
        model_name = row['Model']
        curves = row[curves_col]
        n_folds = len(curves)
        cmap = get_cmap('Set1',n_folds) # Color map for folds
        for fold_idx, curve in enumerate(curves):
            if curve_type == 'roc':
                fpr = curve['fpr']
                tpr = curve['tpr']
                curve_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=cmap(fold_idx), lw=2, label=f'Fold {fold_idx+1} (AUC = {curve_auc:.2f})',alpha=0.7)
                ax.plot([0, 1], [0, 1], linestyle='--', lw=1, color='gray', alpha=.8)

                ax.set_xlabel('FPR')
                ax.set_ylabel('TPR')

            elif curve_type == 'prc':
                precision = curve['precision']
                recall = curve['recall']
                curve_auc = auc(recall, precision)
                ax.plot(recall, precision, color=cmap(fold_idx), lw=2, label=f'Fold {fold_idx+1} (AUC = {curve_auc:.2f})',alpha=0.7)
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')

        # Set ticks to only 0 and 1
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xlim([-0.05, 1.05])
        ax.set_ylim([-0.05, 1.05])
        ax.set_title(f'{model_name}')
        # Shrink current axis by 20%
        sns.despine()
        
        

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    plt.show()
    
    
def combine_one_hot(shap_values, name, mask, return_original=False):
    """  shap_values: an Explanation object
            name: name of new feature
            mask: bool array same lenght as features

            This function assumes that shap_values[:, mask] make up a one-hot-encoded feature
    """
    mask = np.array(mask)
    mask_col_names = np.array(shap_values.feature_names, dtype='object')[mask]

    sv_name = shap.Explanation(shap_values.values[:, mask],
                                feature_names=list(mask_col_names),
                                data=shap_values.data[:, mask],
                                base_values=shap_values.base_values,
                                display_data=shap_values.display_data,
                                instance_names=shap_values.instance_names,
                                output_names=shap_values.output_names,
                                output_indexes=shap_values.output_indexes,
                                lower_bounds=shap_values.lower_bounds,
                                upper_bounds=shap_values.upper_bounds,
                                main_effects=shap_values.main_effects,
                                hierarchical_values=shap_values.hierarchical_values,
                                clustering=shap_values.clustering,
                                )

    new_data = (sv_name.data * np.arange(sum(mask))).sum(axis=1).astype(int)

    svdata = np.concatenate([
        shap_values.data[:, ~mask],
        new_data.reshape(-1, 1)
    ], axis=1)

    if shap_values.display_data is None:
        svdd = shap_values.data[:, ~mask]
    else:
        svdd = shap_values.display_data[:, ~mask]

    svdisplay_data = np.concatenate([
        svdd,
        mask_col_names[new_data].reshape(-1, 1)
    ], axis=1)

    new_values = sv_name.values.sum(axis=1)
    
        
    svvalues = np.concatenate([
        shap_values.values[:, ~mask],
        new_values.reshape(-1, 1)
    ], axis=1)
    svfeature_names = list(np.array(shap_values.feature_names)[~mask]) + [name]

    sv = shap.Explanation(svvalues,
                            base_values=shap_values.base_values,
                            data=svdata,
                            display_data=svdisplay_data,
                            instance_names=shap_values.instance_names,
                            feature_names=svfeature_names,
                            output_names=shap_values.output_names,
                            output_indexes=shap_values.output_indexes,
                            lower_bounds=shap_values.lower_bounds,
                            upper_bounds=shap_values.upper_bounds,
                            main_effects=shap_values.main_effects,
                            hierarchical_values=shap_values.hierarchical_values,
                            clustering=shap_values.clustering,
                            )
    if return_original:
        return sv, sv_name
    else:
        return sv
    
def concatenate_shap_objects(shap1, shap2):
    """
    Concatenates two SHAP Explanation objects by stacking their attributes.
    
    Parameters:
    - shap1: The first SHAP Explanation object.
    - shap2: The second SHAP Explanation object.
    
    Returns:
    - A new SHAP Explanation object with concatenated attributes.
    """
    return shap.Explanation(
        values=np.vstack([shap1.values, shap2.values]),
        base_values=np.hstack([shap1.base_values, shap2.base_values]),
        data=np.vstack([shap1.data, shap2.data]),
        display_data=(np.vstack([shap1.display_data, shap2.display_data])
                        if shap1.display_data is not None else None),
        feature_names=shap1.feature_names,
        output_names=shap1.output_names,
        instance_names=((shap1.instance_names + shap2.instance_names)
                        if shap1.instance_names is not None else None),
        output_indexes=None,  # Adjust other properties if needed
        lower_bounds=(np.vstack([shap1.lower_bounds, shap2.lower_bounds])
                        if shap1.lower_bounds is not None else None),
        upper_bounds=(np.vstack([shap1.upper_bounds, shap2.upper_bounds])
                        if shap1.upper_bounds is not None else None),
        main_effects=(np.vstack([shap1.main_effects, shap2.main_effects])
                        if shap1.main_effects is not None else None),
        hierarchical_values=(np.vstack([shap1.hierarchical_values, shap2.hierarchical_values])
                        if shap1.hierarchical_values is not None else None)
    )