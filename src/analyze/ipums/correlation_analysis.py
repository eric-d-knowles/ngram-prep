import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import pearsonr


def _compute_label_shifts(x_vals, y_vals, labels, inner_width=760, inner_height=530):
    """Greedy label placement in pixel space to reduce overlaps."""
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    n = len(labels)
    if n == 0:
        return []

    x_span = max(np.nanmax(x) - np.nanmin(x), 1e-12)
    y_span = max(np.nanmax(y) - np.nanmin(y), 1e-12)

    # Map data coordinates to approximate plot pixel coordinates.
    px = (x - np.nanmin(x)) / x_span * inner_width
    py = (y - np.nanmin(y)) / y_span * inner_height

    # Place denser points first (harder cases first).
    if n > 1:
        dx = (x[:, None] - x[None, :]) / x_span
        dy = (y[:, None] - y[None, :]) / y_span
        d2 = dx * dx + dy * dy
        np.fill_diagonal(d2, np.inf)
        k = min(3, n - 1)
        density = np.partition(d2, kth=k - 1, axis=1)[:, :k].sum(axis=1)
        order = np.argsort(density)
    else:
        order = np.array([0])

    candidates = [
        (0, 14), (0, -14), (14, 0), (-14, 0),
        (12, 12), (12, -12), (-12, 12), (-12, -12),
        (0, 20), (0, -20), (20, 0), (-20, 0),
    ]

    def _rect_overlap_area(a, b):
        x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
        y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
        return x_overlap * y_overlap

    shifts = [(0.0, 14.0)] * n
    placed_rects = []

    for idx in order:
        label = str(labels[idx])
        text_w = max(22.0, 6.6 * len(label))
        text_h = 13.5

        best = None
        best_score = np.inf
        base_x, base_y = px[idx], py[idx]

        for sx, sy in candidates:
            cx = base_x + sx
            cy = base_y + sy
            rect = (
                cx - text_w / 2,
                cy - text_h / 2,
                cx + text_w / 2,
                cy + text_h / 2,
            )

            overlap_penalty = 0.0
            for other in placed_rects:
                overlap_penalty += _rect_overlap_area(rect, other)

            # Prefer shorter offsets once overlap is minimized.
            distance_penalty = 0.2 * (sx * sx + sy * sy)
            score = overlap_penalty * 1000.0 + distance_penalty

            if score < best_score:
                best_score = score
                best = (sx, sy, rect)

            if overlap_penalty == 0.0 and (sx, sy) in [(0, 14), (0, -14), (14, 0), (-14, 0)]:
                break

        sx, sy, rect = best
        shifts[idx] = (float(sx), float(sy))
        placed_rects.append(rect)

    return shifts

def run_correlation_analysis(panel_df, year_of_interest=None, use_means=False):
    """
    Correlation analysis between women percentage and gender projection scores.

    Parameters
    ----------
    panel_df : DataFrame
        Long-format panel with columns ``profession``, ``year``,
        ``projection``, and ``women_prop``.
    year_of_interest : int or None
        If given (and ``use_means=False``), restrict to a single year.
    use_means : bool
        If True, correlate the over-time mean of each variable per profession.
    """
    if use_means:
        data = panel_df.groupby('profession')[['projection', 'women_prop']].mean().dropna()
        year_label = "all years"
        projection_type = "Mean"
        print(f"Using over-time means for {len(data)} professions")
    else:
        if year_of_interest is None:
            raise ValueError("year_of_interest is required when use_means=False")
        subset = panel_df.loc[panel_df['year'] == year_of_interest]
        if subset.empty:
            available = sorted(panel_df['year'].unique())
            raise ValueError(
                f"No data for year {year_of_interest}. Available: {available}"
            )
        data = subset.set_index('profession')[['projection', 'women_prop']].dropna()
        year_label = str(year_of_interest)
        projection_type = "Raw"
        print(f"Using year {year_of_interest} for {len(data)} professions")

    valid_profs = list(data.index)
    proj_vals = data['projection'].values
    demo_vals = data['women_prop'].values

    # Compute correlation
    if len(valid_profs) >= 3:
        corr, p_value = pearsonr(demo_vals, proj_vals)
        print(f"\n=== Correlation Results ({projection_type}) ===")
        print(f"Pearson r = {corr:.4f}")
        print(f"p-value = {p_value:.6f}")
        print(f"n = {len(valid_profs)} professions")

        # Create polished scatter plot with a fitted regression line.
        z = np.polyfit(demo_vals, proj_vals, 1)
        p = np.poly1d(z)
        x_line = np.linspace(demo_vals.min(), demo_vals.max(), 100)
        y_line = p(x_line)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=demo_vals,
            y=proj_vals,
            mode='markers',
            name='Professions',
            text=valid_profs,
            marker=dict(
                size=10,
                color='#3B82F6',
                line=dict(color='white', width=0.8),
                opacity=0.78,
            ),
            hovertemplate=(
                '%{text}<br>'
                'Women %: %{x:.3f}<br>'
                'Projection: %{y:.3f}<extra></extra>'
            ),
        ))
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode='lines',
            name='Linear fit',
            line=dict(color='black', width=2.5, dash='dash'),
            hovertemplate='fit  %{x:.3f}  %{y:.3f}<extra></extra>',
        ))

        label_shifts = _compute_label_shifts(demo_vals, proj_vals, valid_profs)
        for i, prof in enumerate(valid_profs):
            xshift, yshift = label_shifts[i]
            fig.add_annotation(
                x=float(demo_vals[i]),
                y=float(proj_vals[i]),
                text=str(prof),
                showarrow=False,
                xshift=xshift,
                yshift=yshift,
                font=dict(size=10, color='rgba(35, 35, 35, 0.94)'),
                align='center',
            )

        fig.update_layout(
            title=(
                'Word Embeddings vs. Real-World Gender Demographics (IPUMS)'
                f'<br><sup>Pearson r = {corr:.4f}, p = {p_value:.6f}, n = {len(valid_profs)}</sup>'
            ),
            width=900,
            height=700,
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='closest',
            legend=dict(yanchor='top', y=0.98, xanchor='left', x=0.02),
            xaxis=dict(
                title=f'IPUMS Women Percentage ({year_label})',
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False,
            ),
            yaxis=dict(
                title=f'Gender Projection Score ({year_label}) - {projection_type}',
                showgrid=True,
                gridcolor='lightgray',
                zeroline=False,
            ),
        )
        fig.show()
        if p_value < 0.05:
            print(f"\n✓ Significant correlation (p < 0.05)")
        else:
            print(f"\n✗ Not statistically significant (p >= 0.05)")
    else:
        print(f"ERROR: Not enough valid professions ({len(valid_profs)} found)")
