import os
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import logging

logger = logging.getLogger(__name__)

def get_charts_dir():
    """Ensure a directory for interactive charts exists."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "workspace", "charts")
    os.makedirs(path, exist_ok=True)
    return path

import time

def generate_histogram(df: pd.DataFrame, column: str, color_col: str = None) -> str:
    from pathlib import Path
    try:
        fig = px.histogram(df, x=column, color=color_col, title=f"Interactive Histogram: {column}", marginal="box")
        out_path = os.path.join(get_charts_dir(), f"hist_{column}_{int(time.time())}.html")
        fig.write_html(out_path)
        return out_path
    except Exception as e:
        logger.error(f"Plotly Histogram Error: {e}")
        return None

def generate_scatter(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> str:
    from pathlib import Path
    try:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Interactive Scatter: {x_col} vs {y_col}", hover_data=df.columns[:5])
        out_path = os.path.join(get_charts_dir(), f"scatter_{x_col}_{y_col}_{int(time.time())}.html")
        fig.write_html(out_path)
        return out_path
    except Exception as e:
        logger.error(f"Plotly Scatter Error: {e}")
        return None

def generate_box(df: pd.DataFrame, columns: list, color_col: str = None) -> str:
    from pathlib import Path
    try:
        fig = px.box(df, y=columns, title="Interactive Box Plots")
        out_path = os.path.join(get_charts_dir(), f"box_plots_{int(time.time())}.html")
        fig.write_html(out_path)
        return out_path
    except Exception as e:
        logger.error(f"Plotly Box Error: {e}")
        return None

def generate_correlation(df: pd.DataFrame) -> str:
    from pathlib import Path
    try:
        corr = df.select_dtypes(include='number').corr()
        fig = px.imshow(corr, text_auto=True, title="Interactive Correlation Matrix", color_continuous_scale="RdBu_r")
        out_path = os.path.join(get_charts_dir(), f"correlation_matrix_{int(time.time())}.html")
        fig.write_html(out_path)
        return out_path
    except Exception as e:
        logger.error(f"Plotly Correlation Error: {e}")
        return None

def generate_pca(df: pd.DataFrame, target_col: str = None) -> str:
    """Performs PCA on numeric columns and generates an interactive 2D (or 3D) scatter plot."""
    try:
        # Select numeric columns, drop nulls directly
        numeric_df = df.select_dtypes(include=['number']).dropna()
        if len(numeric_df) < 5 or len(numeric_df.columns) < 2:
            return None
            
        features = numeric_df.columns.tolist()
        if target_col and target_col in features:
            features.remove(target_col)
            
        if len(features) < 2:
            return None
            
        x = numeric_df[features].values
        x = StandardScaler().fit_transform(x)
        
        # PCA
        n_components = min(3, len(features))
        pca = PCA(n_components=n_components)
        components = pca.fit_transform(x)
        
        # Plot
        total_var = pca.explained_variance_ratio_.sum() * 100
        var_1 = pca.explained_variance_ratio_[0] * 100
        var_2 = pca.explained_variance_ratio_[1] * 100
        
        if n_components >= 3:
            var_3 = pca.explained_variance_ratio_[2] * 100
            fig = px.scatter_3d(
                components, x=0, y=1, z=2, 
                color=numeric_df[target_col] if target_col in numeric_df.columns else None,
                title=f'Interactive PCA (3D) - Total Explained Variance: {total_var:.2f}%',
                labels={'0': f'PC 1 ({var_1:.1f}%)', '1': f'PC 2 ({var_2:.1f}%)', '2': f'PC 3 ({var_3:.1f}%)'}
            )
            # Add shadow and 3D effects to markers
            fig.update_traces(marker=dict(size=5, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
            # Enable lighting/shadow for 3D
            fig.update_scenes(
                xaxis_showspikes=False, yaxis_showspikes=False, zaxis_showspikes=False,
                camera=dict(projection=dict(type='orthographic'))
            )
            fig.update_traces(
                marker=dict(
                    opacity=0.9,
                    line=dict(width=0), # Remove line to look more like a sphere
                )
            )
            # Actually plot spheres with lighting 
            # (Note in plotly scatter_3d they are spheres when marker.symbol isn't changed, but we can improve lighting)
            for trace in fig.data:
                trace.marker.coloraxis = None

        else:
            fig = px.scatter(
                components, x=0, y=1, 
                color=numeric_df[target_col] if target_col in numeric_df.columns else None,
                title=f'Interactive PCA (2D) - Total Explained Variance: {total_var:.2f}%',
                labels={'0': f'PC 1 ({var_1:.1f}%)', '1': f'PC 2 ({var_2:.1f}%)'}
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')))
            
        out_path = os.path.join(get_charts_dir(), f"pca_scatter_{int(time.time())}.html")
        fig.write_html(out_path)
        return out_path
    except Exception as e:
        logger.error(f"Plotly PCA Error: {e}")
        return None
