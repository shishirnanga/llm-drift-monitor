"""
dashboard/app.py

Interactive dashboard for LLM Drift Monitor.

Run with: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import src
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

from src.storage import ResultsStorage
from src.analysis import get_all_baselines, detect_drift
from dashboard.components.charts import (
    plot_performance_over_time,
    plot_model_comparison,
    plot_category_breakdown,
    plot_drift_timeline,
)
from dashboard.components.metrics import (
    display_summary_metrics,
    display_drift_alerts,
)


# Page config
st.set_page_config(
    page_title="LLM Drift Monitor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .drift-alert {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load all data from storage."""
    storage = ResultsStorage()
    runs = storage.load_all_runs()
    
    if not runs:
        st.error("No data found. Run `python main.py` to collect data first.")
        st.stop()
    
    return storage, runs


def main():
    # Header
    st.markdown('<div class="main-header">🔍 LLM Drift Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Track how AI models change over time</div>', unsafe_allow_html=True)
    
    # Load data
    storage, runs = load_data()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Date range filter
    min_date = datetime.fromisoformat(runs[0].timestamp).date()
    max_date = datetime.fromisoformat(runs[-1].timestamp).date()
    
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Model filter
    all_models = set()
    for run in runs:
        all_models.update(run.models_tested)
    
    selected_models = st.sidebar.multiselect(
        "Models",
        options=sorted(all_models),
        default=sorted(all_models)
    )
    
    # Category filter
    categories = ["math", "reasoning", "factual", "consistency", "instruction"]
    selected_categories = st.sidebar.multiselect(
        "Categories",
        options=categories,
        default=categories
    )
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Data Summary")
    st.sidebar.metric("Total Runs", len(runs))
    st.sidebar.metric("Date Range", f"{min_date} to {max_date}")
    st.sidebar.metric("Models", len(all_models))
    
    # Filter runs by date
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_runs = [
            r for r in runs
            if start_date <= datetime.fromisoformat(r.timestamp).date() <= end_date
        ]
    else:
        filtered_runs = runs
    
    if not filtered_runs:
        st.warning("No data in selected date range.")
        return
    
    # Main content
    tabs = st.tabs(["📈 Overview", "📊 Analysis", "🔬 Drift Detection", "📋 Data"])
    
    # TAB 1: Overview
    with tabs[0]:
        st.header("Performance Overview")
        
        # Summary metrics
        display_summary_metrics(filtered_runs, selected_models)
        
        st.markdown("---")
        
        # Performance over time
        st.subheader("📈 Performance Over Time")
        fig_timeline = plot_performance_over_time(
            filtered_runs, 
            selected_models,
            selected_categories
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("---")
        
        # Model comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚔️ Model Comparison")
            fig_comparison = plot_model_comparison(filtered_runs, selected_models)
            st.plotly_chart(fig_comparison, use_container_width=True)
        
        with col2:
            st.subheader("📂 By Category")
            fig_category = plot_category_breakdown(
                filtered_runs,
                selected_models,
                selected_categories
            )
            st.plotly_chart(fig_category, use_container_width=True)
    
    # TAB 2: Analysis
    with tabs[1]:
        st.header("Statistical Analysis")
        
        if len(runs) < 7:
            st.warning("Need at least 7 runs to calculate baseline. Keep collecting data!")
        else:
            st.subheader("📊 Baseline Metrics")
            
            try:
                baselines = get_all_baselines(storage, num_runs=min(7, len(runs)))
                
                for model_name in selected_models:
                    if model_name in baselines:
                        baseline = baselines[model_name]
                        
                        with st.expander(f"🎯 {model_name}", expanded=True):
                            cols = st.columns(4)
                            
                            stats = baseline.overall_stats
                            ci_lower, ci_upper = stats.confidence_interval_95
                            
                            cols[0].metric("Mean Score", f"{stats.mean:.1%}")
                            cols[1].metric("Std Dev", f"{stats.std:.3f}")
                            cols[2].metric("95% CI", f"[{ci_lower:.1%}, {ci_upper:.1%}]")
                            cols[3].metric("Runs", baseline.num_runs)
                            
                            st.markdown("**By Category:**")
                            cat_data = []
                            for cat, cat_stats in baseline.by_category.items():
                                cat_data.append({
                                    "Category": cat,
                                    "Mean": f"{cat_stats.mean:.1%}",
                                    "Std Dev": f"{cat_stats.std:.3f}"
                                })
                            st.dataframe(pd.DataFrame(cat_data), hide_index=True)
            
            except Exception as e:
                st.error(f"Error calculating baseline: {e}")
    
    # TAB 3: Drift Detection
    with tabs[2]:
        st.header("🔬 Drift Detection")
        
        if len(runs) < 10:
            st.warning(f"Need at least 10 runs for drift detection. Currently: {len(runs)}")
            st.info("Keep running `python main.py` daily. Come back when you have 10+ days of data!")
        else:
            st.subheader("Recent Drift Analysis")
            
            display_drift_alerts(storage, selected_models)
            
            st.markdown("---")
            
            st.subheader("📈 Drift Timeline")
            fig_drift = plot_drift_timeline(runs, selected_models)
            st.plotly_chart(fig_drift, use_container_width=True)
    
    # TAB 4: Raw Data
    with tabs[3]:
        st.header("📋 Raw Data")
        
        # Convert to DataFrame
        all_results = []
        for run in filtered_runs:
            for result in run.results:
                if result.model_name in selected_models:
                    if not selected_categories or result.category in selected_categories:
                        all_results.append({
                            "Date": run.timestamp[:10],
                            "Time": run.timestamp[11:19],
                            "Model": result.model_name,
                            "Test ID": result.test_id,
                            "Category": result.category,
                            "Score": result.score,
                            "Latency (ms)": result.latency_ms,
                            "Tokens": result.tokens_total,
                        })
        
        df = pd.DataFrame(all_results)
        
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download button
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"llm_drift_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
        
        # Summary stats
        st.subheader("Summary Statistics")
        st.write(df.describe())


if __name__ == "__main__":
    main()