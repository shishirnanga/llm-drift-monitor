import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from typing import List


def plot_performance_over_time(runs, models: List[str], categories: List[str] = None):
    """
    Line chart showing performance over time.
    Args:
        runs: List of TestRun objects
        models: Which models to include
        categories: Optional category filter
    """
    data = []
    
    for run in runs:
        date = run.timestamp[:10]
        
        for model in models:
            model_results = [r for r in run.results if r.model_name == model]
            
            # Filter by category if specified
            if categories:
                model_results = [r for r in model_results if r.category in categories]
            
            if model_results:
                avg_score = sum(r.score for r in model_results) / len(model_results)
                data.append({
                    "Date": date,
                    "Model": model,
                    "Score": avg_score * 100,  # Convert to percentage
                })
    
    df = pd.DataFrame(data)
    
    fig = px.line(
        df,
        x="Date",
        y="Score",
        color="Model",
        markers=True,
        title="Performance Over Time"
    )
    
    fig.update_layout(
        yaxis_title="Average Score (%)",
        yaxis_range=[0, 100],
        hovermode="x unified",
        showlegend=True,
        height=400
    )
    
    return fig


def plot_model_comparison(runs, models: List[str]):
    """
    Bar chart comparing average performance between models.
    """
    data = []
    
    for model in models:
        all_results = []
        for run in runs:
            model_results = [r for r in run.results if r.model_name == model]
            all_results.extend(model_results)
        
        if all_results:
            avg_score = sum(r.score for r in all_results) / len(all_results)
            data.append({
                "Model": model,
                "Average Score": avg_score * 100,
                "Tests": len(all_results)
            })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x="Model",
        y="Average Score",
        text="Average Score",
        title="Model Comparison"
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Average Score (%)",
        showlegend=False,
        height=400
    )
    
    return fig


def plot_category_breakdown(runs, models: List[str], categories: List[str] = None):
    """
    Grouped bar chart showing performance by category.
    """
    data = []
    
    all_categories = categories or ["math", "reasoning", "factual", "consistency", "instruction"]
    
    for model in models:
        for category in all_categories:
            category_results = []
            
            for run in runs:
                model_results = [
                    r for r in run.results
                    if r.model_name == model and r.category == category
                ]
                category_results.extend(model_results)
            
            if category_results:
                avg_score = sum(r.score for r in category_results) / len(category_results)
                data.append({
                    "Model": model,
                    "Category": category.capitalize(),
                    "Score": avg_score * 100
                })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x="Category",
        y="Score",
        color="Model",
        barmode="group",
        title="Performance by Category"
    )
    
    fig.update_layout(
        yaxis_range=[0, 100],
        yaxis_title="Average Score (%)",
        height=400
    )
    
    return fig


def plot_drift_timeline(runs, models: List[str]):
    """
    Timeline showing when drift was detected.
    """
    # Calculate rolling average for each model
    data = []
    
    for model in models:
        for i, run in enumerate(runs):
            model_results = [r for r in run.results if r.model_name == model]
            
            if model_results:
                avg_score = sum(r.score for r in model_results) / len(model_results)
                
                # Calculate baseline (first 7 runs)
                # Calculate baseline (first 7 runs that contain THIS model)
                if i >= 7:
                    # Find first 7 runs containing this model
                    model_runs_before_current = []
                    for past_run in runs[:i]:  # Only look at runs BEFORE current run
                        if model in past_run.models_tested:
                            model_runs_before_current.append(past_run)
                            if len(model_runs_before_current) >= 7:
                                break
                    
                    if len(model_runs_before_current) >= 7:
                        # Calculate baseline from first 7 runs containing this model
                        baseline_results = []
                        for baseline_run in model_runs_before_current[:7]:
                            baseline_model_results = [
                                r for r in baseline_run.results 
                                if r.model_name == model
                            ]
                            baseline_results.extend(baseline_model_results)
                        
                        baseline_avg = sum(r.score for r in baseline_results) / len(baseline_results) if baseline_results else avg_score
                        deviation = ((avg_score - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
                    else:
                        deviation = 0
                else:
                    deviation = 0
                data.append({
                    "Date": run.timestamp[:10],
                    "Model": model,
                    "Score": avg_score * 100,
                    "Deviation": deviation
                })
    
    df = pd.DataFrame(data)
    
    fig = go.Figure()
    
    for model in models:
        model_df = df[df["Model"] == model]
        
        fig.add_trace(go.Scatter(
            x=model_df["Date"],
            y=model_df["Deviation"],
            mode='lines+markers',
            name=model,
            hovertemplate='%{x}<br>Deviation: %{y:.1f}%<extra></extra>'
        ))
    
    # Add threshold lines
    fig.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="+5% threshold")
    fig.add_hline(y=-5, line_dash="dash", line_color="orange", annotation_text="-5% threshold")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    
    fig.update_layout(
        title="Deviation from Baseline Over Time",
        xaxis_title="Date",
        yaxis_title="Deviation from Baseline (%)",
        hovermode="x unified",
        height=400
    )
    
    return fig


def plot_test_heatmap(runs, model: str):
    """
    Heatmap showing which tests pass/fail over time.
    """
    # Get all unique test IDs
    test_ids = set()
    for run in runs:
        for result in run.results:
            if result.model_name == model:
                test_ids.add(result.test_id)
    
    test_ids = sorted(test_ids)
    dates = [run.timestamp[:10] for run in runs]
    
    # Create matrix
    matrix = []
    for test_id in test_ids:
        row = []
        for run in runs:
            result = next(
                (r for r in run.results if r.test_id == test_id and r.model_name == model),
                None
            )
            row.append(result.score if result else None)
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=dates,
        y=test_ids,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        hoverongaps=False,
        hovertemplate='Date: %{x}<br>Test: %{y}<br>Score: %{z:.0%}<extra></extra>'
    ))
    
    fig.update_layout(
        title=f"Test Results Heatmap - {model}",
        xaxis_title="Date",
        yaxis_title="Test ID",
        height=800
    )
    
    return fig