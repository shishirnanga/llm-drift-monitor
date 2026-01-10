import streamlit as st
from typing import List
from src.analysis import detect_drift


def display_summary_metrics(runs, models: List[str]):
    """
    Display summary metrics at the top of the dashboard.
    """
    cols = st.columns(len(models))
    
    for idx, model in enumerate(models):
        with cols[idx]:
            # Get all results for this model
            all_results = []
            for run in runs:
                model_results = [r for r in run.results if r.model_name == model]
                all_results.extend(model_results)
            
            if all_results:
                # Calculate metrics
                avg_score = sum(r.score for r in all_results) / len(all_results)
                success_rate = sum(1 for r in all_results if r.success) / len(all_results)
                avg_latency = sum(r.latency_ms for r in all_results if r.success) / len([r for r in all_results if r.success])
                
                # Compare to first half
                mid_point = len(runs) // 2
                if mid_point > 0:
                    early_results = []
                    for run in runs[:mid_point]:
                        early_results.extend([r for r in run.results if r.model_name == model])
                    
                    recent_results = []
                    for run in runs[mid_point:]:
                        recent_results.extend([r for r in run.results if r.model_name == model])
                    
                    if early_results and recent_results:
                        early_avg = sum(r.score for r in early_results) / len(early_results)
                        recent_avg = sum(r.score for r in recent_results) / len(recent_results)
                        delta = recent_avg - early_avg
                    else:
                        delta = 0
                else:
                    delta = 0
                
                # Display
                st.metric(
                    label=f" {model}",
                    value=f"{avg_score:.1%}",
                    delta=f"{delta:+.1%}" if delta != 0 else None
                )
                
                st.caption(f" {avg_latency:.0f}ms avg")
                st.caption(f" {success_rate:.0%} success rate")


def display_drift_alerts(storage, models: List[str]):
    """
    Display drift detection alerts.
    """
    for model in models:
        try:
            result = detect_drift(storage, model, baseline_runs=7, current_runs=3)
            
            if result.drift_detected:
                # Show alert
                severity_colors = {
                    "minor": "🟡",
                    "moderate": "🟠", 
                    "major": "🔴"
                }
                icon = severity_colors.get(result.severity.value, "")
                
                st.warning(f"{icon} **{model}**: {result.summary}")
                
                # Show details in expander
                with st.expander(" Details"):
                    cols = st.columns(4)
                    
                    cols[0].metric("Baseline", f"{result.baseline_mean:.1%}")
                    cols[1].metric("Current", f"{result.current_mean:.1%}")
                    cols[2].metric("Change", f"{result.change_percent:+.1f}%")
                    cols[3].metric("p-value", f"{result.p_value:.4f}")
                    
                    st.markdown(f"**Cohen's d:** {result.cohens_d:.3f}")
                    st.markdown(f"**Period:** {result.test_period}")
            else:
                st.success(f" **{model}**: {result.summary}")
        
        except ValueError as e:
            if "Need at least" in str(e):
                st.info(f" **{model}**: Not enough data for drift detection yet.")
            else:
                st.error(f" Error analyzing {model}: {e}")
        except Exception as e:
            st.error(f" Error analyzing {model}: {e}")


def display_category_performance(runs, model: str):
    """
    Display performance breakdown by category.
    """
    categories = ["math", "reasoning", "factual", "consistency", "instruction"]
    
    for category in categories:
        # Get results for this category
        cat_results = []
        for run in runs:
            results = [
                r for r in run.results
                if r.model_name == model and r.category == category
            ]
            cat_results.extend(results)
        
        if cat_results:
            avg_score = sum(r.score for r in cat_results) / len(cat_results)
            
            # Create progress bar
            st.markdown(f"**{category.capitalize()}**")
            st.progress(avg_score, text=f"{avg_score:.1%}")
            st.markdown("")  # Spacing


def display_latency_stats(runs, model: str):
    """
    Display latency statistics.
    """
    all_results = []
    for run in runs:
        model_results = [r for r in run.results if r.model_name == model and r.success]
        all_results.extend(model_results)
    
    if all_results:
        latencies = [r.latency_ms for r in all_results]
        
        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        
        cols = st.columns(3)
        cols[0].metric("Average", f"{avg_latency:.0f}ms")
        cols[1].metric("Min", f"{min_latency:.0f}ms")
        cols[2].metric("Max", f"{max_latency:.0f}ms")


def display_token_stats(runs, model: str):
    """
    Display token usage statistics.
    """
    all_results = []
    for run in runs:
        model_results = [r for r in run.results if r.model_name == model]
        all_results.extend(model_results)
    
    if all_results:
        total_tokens = sum(r.tokens_total for r in all_results)
        avg_tokens = total_tokens / len(all_results)
        
        # Estimate cost (rough estimates)
        # GPT-4: $0.03 per 1K input, $0.06 per 1K output (average ~$0.045/1K)
        # Claude: $0.015 per 1K input, $0.075 per 1K output (average ~$0.045/1K)
        cost_per_1k = 0.045
        estimated_cost = (total_tokens / 1000) * cost_per_1k
        
        cols = st.columns(3)
        cols[0].metric("Total Tokens", f"{total_tokens:,}")
        cols[1].metric("Avg per Test", f"{avg_tokens:.0f}")
        cols[2].metric("Est. Cost", f"${estimated_cost:.2f}")