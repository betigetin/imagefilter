"""
app.py
SP13 - Image Smoothing and Sharpening
IIT Madras, BS Electronic Systems, Signal Processing Project
Run with: streamlit run app.py
"""
import io
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from src.core.convolution import normalize, denormalize
from src.filters.gaussian_filter import apply_gaussian_filter, build_gaussian_kernel
from src.filters.laplacian_filter import (apply_laplacian, apply_sharpening,
                                           build_laplacian_kernel)
from src.filters.mean_filter import apply_mean_filter, build_mean_kernel
from src.metrics.metrics import compute_all_metrics

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SP13 Image Filter Lab",
    page_icon=":microscope:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .filter-banner {
        padding: 10px 16px;
        border-radius: 6px;
        border-left: 4px solid;
        margin-bottom: 16px;
        font-size: 14px;
        line-height: 1.6;
    }
    .stDownloadButton > button {
        background-color: #1f4e79;
        color: white !important;
        border: none;
        font-weight: 500;
    }
    .stDownloadButton > button:hover {
        background-color: #2e75b6;
    }
</style>
""", unsafe_allow_html=True)

# ── Filter metadata ────────────────────────────────────────────────────────────
FILTER_INFO = {
    "Mean Filter": {
        "type": "Low-Pass Filter (LPF)",
        "equation": "g[x,y] = (1/K^2) * SUM h[s,t] * f[x+s, y+t]",
        "desc": ("Uniform local averaging. Attenuates high-frequency noise at the cost of "
                 "edge blurring. Larger kernels produce stronger smoothing."),
        "freq": "Sinc-shaped 2D frequency response with significant sidelobes.",
        "color": "#2e75b6",
    },
    "Gaussian Filter": {
        "type": "Low-Pass Filter (LPF)",
        "equation": "G(x,y) = (1/2*pi*s^2) * exp(-(x^2+y^2) / 2*s^2)",
        "desc": ("Gaussian-weighted averaging. Superior edge preservation vs. mean filter. "
                 "Separable implementation (O(2K) per pixel). Zero ringing artifacts."),
        "freq": "Gaussian frequency response: smooth monotonic rolloff, no sidelobes.",
        "color": "#1a7a40",
    },
    "Laplacian Sharpening": {
        "type": "High-Pass / Second-Derivative Operator",
        "equation": "g[x,y] = f[x,y] - c * (h_lap * f)[x,y]",
        "desc": ("Second-order derivative operator. Enhances edges and fine structural detail. "
                 "Sensitive to noise; enable pre-smoothing for noisy inputs."),
        "freq": "Response proportional to -(u^2+v^2): amplifies high frequencies, suppresses DC.",
        "color": "#c0392b",
    },
}

# ── Helper functions ───────────────────────────────────────────────────────────

def load_image(uploaded_file) -> np.ndarray:
    """Load uploaded image, force convert to grayscale uint8."""
    img = Image.open(uploaded_file).convert('L')
    return np.array(img, dtype=np.uint8)


def array_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode='L')


def plot_histograms(orig: np.ndarray, filt: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=orig.flatten(), nbinsx=64, name='Original',
        marker_color='rgba(55,128,191,0.75)', opacity=0.85,
    ))
    fig.add_trace(go.Histogram(
        x=filt.flatten(), nbinsx=64, name='Filtered',
        marker_color='rgba(219,64,82,0.75)', opacity=0.85,
    ))
    fig.update_layout(
        barmode='overlay',
        title='Pixel Intensity Histogram Comparison',
        xaxis_title='Pixel Intensity [0-255]',
        yaxis_title='Pixel Count',
        legend=dict(orientation='h', x=0, y=1.12),
        height=320,
        margin=dict(l=50, r=20, t=60, b=50),
    )
    return fig


def plot_diff_map(orig: np.ndarray, filt: np.ndarray) -> go.Figure:
    diff = np.abs(orig.astype(np.int16) - filt.astype(np.int16)).astype(np.float32)
    fig  = go.Figure(data=go.Heatmap(
        z=diff, colorscale='Hot',
        colorbar=dict(title='|Diff|', titlefont=dict(size=11)),
    ))
    fig.update_layout(
        title=f'Difference Map  |  Max diff: {diff.max():.1f}  |  Mean diff: {diff.mean():.2f}',
        height=380, margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_kernel_heatmap(kernel: np.ndarray, title: str) -> go.Figure:
    text = [[f'{v:.4f}' for v in row] for row in kernel.round(4)]
    fig  = go.Figure(data=go.Heatmap(
        z=kernel, colorscale='Blues',
        text=text, texttemplate='%{text}',
        textfont={'size': 13}, showscale=True,
    ))
    fig.update_layout(
        title=title, height=280,
        margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


def plot_freq_response(kernel: np.ndarray, label: str) -> go.Figure:
    padded = np.zeros((256, 256), dtype=np.float32)
    kH, kW = kernel.shape
    pH = 128 - kH // 2
    pW = 128 - kW // 2
    padded[pH:pH+kH, pW:pW+kW] = kernel
    freq    = np.abs(np.fft.fftshift(np.fft.fft2(padded)))
    freq_db = 20 * np.log10(freq + 1e-8)
    fig = go.Figure(data=go.Heatmap(
        z=freq_db, colorscale='Viridis',
        colorbar=dict(title='Magnitude (dB)'),
    ))
    fig.update_layout(
        title=f'{label}: 2D Frequency Response',
        height=320, margin=dict(l=40, r=20, t=50, b=40),
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
    )
    return fig


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.title("SP13 Image Smoothing and Sharpening Lab")
    st.caption(
        "IIT Madras, BS Electronic Systems  |  "
        "Signal Processing Project  |  Spatial Domain Filtering"
    )

    # ──────────────── SIDEBAR ────────────────────────────────────────────────
    with st.sidebar:
        st.header("Image Upload")
        uploaded_file = st.file_uploader(
            "Select a PNG or JPG image",
            type=['png', 'jpg', 'jpeg'],
            help="Color images are automatically converted to grayscale.",
        )

        st.divider()
        st.header("Filter Configuration")

        filter_type = st.radio(
            "Filter Type",
            list(FILTER_INFO.keys()),
            help="Select the spatial domain filter to apply.",
        )

        st.divider()
        params = {}

        if filter_type == "Mean Filter":
            params['size'] = st.select_slider(
                "Kernel Size", options=[3, 5, 7], value=3,
                help="Larger kernels produce stronger blurring.",
            )
            coeff = 1.0 / (params['size'] ** 2)
            st.info(
                f"Kernel: {params['size']}x{params['size']}  "
                f"|  Each coefficient = {coeff:.5f}"
            )

        elif filter_type == "Gaussian Filter":
            params['sigma'] = st.slider(
                "Sigma (sigma)", min_value=0.5, max_value=5.0,
                value=1.0, step=0.1,
                help="Standard deviation in pixels. Larger sigma = stronger blur.",
            )
            recommended = 2 * math.ceil(3 * params['sigma']) + 1
            if recommended % 2 == 0:
                recommended += 1
            params['size'] = st.select_slider(
                "Kernel Size", options=[3, 5, 7, 9, 11],
                value=min(recommended, 9),
                help=f"Recommended for sigma={params['sigma']}: {recommended}",
            )
            st.caption(
                f"Auto-recommended kernel size for sigma={params['sigma']}: {recommended}"
            )

        elif filter_type == "Laplacian Sharpening":
            params['c'] = st.slider(
                "Sharpening Coefficient (c)",
                min_value=0.1, max_value=1.5, value=0.8, step=0.05,
                help="Controls sharpening strength. c=1.0 is standard.",
            )
            params['connectivity'] = st.radio(
                "Laplacian Connectivity", options=[4, 8],
                help="4: cardinal neighbors only.  8: includes diagonals.",
            )
            params['pre_smooth'] = st.checkbox(
                "Gaussian Pre-Smoothing (sigma=0.5)",
                value=False,
                help=(
                    "Apply mild Gaussian pre-filter before Laplacian to "
                    "reduce noise amplification (unsharp-masking style)."
                ),
            )
            if params['c'] > 1.0:
                st.warning(
                    f"c = {params['c']} > 1.0 may cause visible noise "
                    "amplification. Consider enabling pre-smoothing."
                )

        st.divider()
        apply_btn = st.button(
            "Apply Filter", type="primary", use_container_width=True,
        )

    # ──────────────── MAIN CONTENT ───────────────────────────────────────────
    if uploaded_file is None:
        st.subheader("Spatial Domain Image Filters")
        cols = st.columns(3)
        for col, (fname, info) in zip(cols, FILTER_INFO.items()):
            with col:
                st.markdown(f"**{fname}**")
                st.caption(info['type'])
                st.code(info['equation'], language='text')
                st.markdown(info['desc'])
        st.info("Upload an image using the sidebar to begin processing.")
        return

    # Load image
    try:
        img = load_image(uploaded_file)
    except Exception as exc:
        st.error(f"Failed to load image: {exc}")
        return

    H, W = img.shape

    # Apply filter
    with st.spinner(f"Applying {filter_type} ..."):
        if filter_type == "Mean Filter":
            filtered  = apply_mean_filter(img, params['size'])
            kernel    = build_mean_kernel(params['size'])
            ker_title = f"Mean Kernel ({params['size']}x{params['size']})"

        elif filter_type == "Gaussian Filter":
            filtered  = apply_gaussian_filter(img, params['sigma'], params['size'])
            kernel    = build_gaussian_kernel(params['size'], params['sigma'])
            ker_title = (
                f"Gaussian Kernel ({params['size']}x{params['size']}, "
                f"sigma={params['sigma']})"
            )

        else:  # Laplacian Sharpening
            filtered  = apply_sharpening(
                img, params['c'], params['connectivity'],
                params.get('pre_smooth', False),
            )
            kernel    = build_laplacian_kernel(params['connectivity'])
            ker_title = (
                f"Laplacian Kernel ({params['connectivity']}-connectivity)"
            )

        metrics = compute_all_metrics(img, filtered)

    # Filter info banner
    info = FILTER_INFO[filter_type]
    st.markdown(
        f'<div class="filter-banner" '
        f'style="border-color:{info["color"]}; background:{info["color"]}18;">'
        f'<strong>{filter_type}</strong>  |  {info["type"]}<br>'
        f'{info["desc"]}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Comparison", "Metrics and Histograms",
        "Frequency Analysis", "Filter Kernel",
    ])

    # ── Tab 1: Comparison ────────────────────────────────────────────────────
    with tab1:
        c1, c2, c3 = st.columns(3)

        with c1:
            st.subheader("Original")
            st.image(img, use_container_width=True, clamp=True,
                     caption=f"Grayscale input  |  {W}x{H} px")

        with c2:
            st.subheader("Filtered Output")
            st.image(filtered, use_container_width=True, clamp=True,
                     caption=filter_type)

        with c3:
            st.subheader("Difference Map (x3)")
            diff_raw = np.abs(img.astype(np.int16) - filtered.astype(np.int16))
            diff_vis = np.clip(diff_raw * 3, 0, 255).astype(np.uint8)
            st.image(diff_vis, use_container_width=True, clamp=True,
                     caption=f"|Orig - Filtered| x3  |  max={diff_raw.max()}")

        buf = io.BytesIO()
        array_to_pil(filtered).save(buf, format='PNG')
        st.download_button(
            label="Download Filtered Image (PNG)",
            data=buf.getvalue(),
            file_name=f"SP13_{filter_type.replace(' ', '_')}_output.png",
            mime="image/png",
        )

    # ── Tab 2: Metrics ───────────────────────────────────────────────────────
    with tab2:
        st.subheader("Quantitative Performance Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PSNR",         f"{metrics['psnr']:.2f} dB",
                  help="Peak Signal-to-Noise Ratio. Higher = less distortion.")
        m2.metric("MSE",          f"{metrics['mse']:.5f}",
                  help="Mean Squared Error (normalized [0,1]). Lower = better.")
        m3.metric("Sharpness",    f"{metrics['sharpness']:.6f}",
                  help="Laplacian variance of output. Higher = sharper.")
        m4.metric("SSIM",         f"{metrics['ssim']:.4f}",
                  help="Structural Similarity Index. 1.0 = identical to original.")

        st.plotly_chart(
            plot_histograms(img, filtered), use_container_width=True,
        )
        st.plotly_chart(
            plot_diff_map(img, filtered), use_container_width=True,
        )

    # ── Tab 3: Frequency Analysis ─────────────────────────────────────────────
    with tab3:
        st.subheader("Frequency Domain Analysis")
        fa1, fa2 = st.columns(2)

        with fa1:
            st.plotly_chart(
                plot_freq_response(kernel, filter_type), use_container_width=True,
            )

        with fa2:
            img_f    = img.astype(np.float32) / 255.0
            fft_img  = np.abs(np.fft.fftshift(np.fft.fft2(img_f)))
            fft_db   = 20 * np.log10(fft_img + 1e-8)
            fig_fft  = go.Figure(data=go.Heatmap(
                z=fft_db, colorscale='Plasma',
                colorbar=dict(title='Magnitude (dB)'),
            ))
            fig_fft.update_layout(
                title='Input Image: 2D FFT Spectrum', height=320,
                margin=dict(l=40, r=20, t=50, b=40),
                xaxis=dict(showticklabels=False),
                yaxis=dict(showticklabels=False),
            )
            st.plotly_chart(fig_fft, use_container_width=True)

        st.info(f"Frequency interpretation: {info['freq']}")

    # ── Tab 4: Kernel ─────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Filter Kernel Visualization")
        k1, k2 = st.columns([3, 2])

        with k1:
            st.plotly_chart(
                plot_kernel_heatmap(kernel, ker_title), use_container_width=True,
            )
            st.caption(
                f"Shape: {kernel.shape[0]}x{kernel.shape[1]}  "
                f"|  Sum of coefficients: {kernel.sum():.6f}"
            )

        with k2:
            st.markdown(f"**Shape:** {kernel.shape[0]} x {kernel.shape[1]}")
            st.markdown(f"**Sum:** {kernel.sum():.6f}")
            st.markdown(f"**Min coefficient:** {kernel.min():.6f}")
            st.markdown(f"**Max coefficient:** {kernel.max():.6f}")
            dc = (
                "DC-preserving (sum approx 1)"
                if abs(kernel.sum() - 1) < 0.01
                else "DC-suppressing (sum approx 0)"
            )
            st.markdown(f"**Type:** {dc}")

            df = pd.DataFrame(kernel.round(5))
            st.dataframe(df, use_container_width=True, height=200)


if __name__ == '__main__':
    main()
