#!/usr/bin/env python3
"""
recommend_app.py

Public-facing AI Model Recommendation Tool

Run with: streamlit run recommend_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.storage import ResultsStorage
from src.recommender import ModelRecommender, TaskType


# Page config
st.set_page_config(
    page_title="AI Model Advisor",
    page_icon="",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
    }
    .alternative-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    # Header
    st.markdown('<div class="big-title"> AI Model Advisor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Find the perfect AI model for your task, backed by real performance data</div>',
        unsafe_allow_html=True
    )
    
    # Load data
    try:
        storage = ResultsStorage()
        recommender = ModelRecommender(storage)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Make sure you've run `python main.py` to collect performance data first.")
        return
    
    # Main interface
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.subheader("What do you need to do?")
        
        task_descriptions = {
            TaskType.MATH: " Math & Calculations",
            TaskType.LOGICAL_REASONING: "🧩 Logic Puzzles & Reasoning",
            TaskType.CREATIVE_WRITING: "✍ Creative Writing & Stories",
            TaskType.CODE_GENERATION: "💻 Code Generation & Debugging",
            TaskType.FACTUAL_QA: "📚 Factual Questions & Knowledge",
            TaskType.INSTRUCTION_FOLLOWING: " Following Specific Instructions",
            TaskType.COMPLEX_PROBLEM: "🔬 Complex Problem Solving",
            TaskType.GENERAL: "🌐 General Purpose Tasks",
        }
        
        selected_task = st.radio(
            "Select your task type:",
            options=list(task_descriptions.keys()),
            format_func=lambda x: task_descriptions[x],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        st.subheader("What matters most?")
        
        accuracy_weight = st.slider(
            "Accuracy",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="How important is getting the right answer?"
        )
        
        speed_weight = st.slider(
            "Speed",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.1,
            help="How important is fast response time?"
        )
        
        consistency_weight = st.slider(
            "Consistency",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.1,
            help="How important is consistent behavior?"
        )
        
        # Normalize weights
        total = accuracy_weight + speed_weight + consistency_weight
        if total > 0:
            priorities = {
                "accuracy": accuracy_weight / total,
                "speed": speed_weight / total,
                "consistency": consistency_weight / total,
            }
        else:
            priorities = {"accuracy": 0.7, "speed": 0.2, "consistency": 0.1}
    
    with col2:
        st.subheader("Recommendation")
        
        try:
            recommendation = recommender.recommend(selected_task, priorities)
            
            # Main recommendation
            st.markdown(f"""
            <div class="recommendation-box">
                <h2 style="margin-top: 0;">✨ Use {recommendation.recommended_model}</h2>
                <p style="font-size: 1.2rem; margin: 1rem 0;">
                    {recommendation.reasoning}
                </p>
                <div style="display: flex; gap: 2rem; margin-top: 1.5rem;">
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Performance</div>
                        <div style="font-size: 1.8rem; font-weight: bold;">
                            {recommendation.performance_score:.1%}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Confidence</div>
                        <div style="font-size: 1.8rem; font-weight: bold;">
                            {recommendation.confidence:.1%}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.9rem; opacity: 0.8;">Avg Speed</div>
                        <div style="font-size: 1.8rem; font-weight: bold;">
                            {recommendation.avg_latency_ms}ms
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Alternatives
            if recommendation.alternatives:
                st.markdown("### Consider These Alternatives")
                
                for alt_model, alt_score, reason in recommendation.alternatives:
                    st.markdown(f"""
                    <div class="alternative-box">
                        <strong>{alt_model}</strong> ({alt_score:.1%})<br>
                        <span style="color: #666;">{reason}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Detailed comparison
            st.markdown("---")
            st.markdown("### Detailed Comparison")
            
            comparison = recommender.compare_models(selected_task)
            
            if comparison:
                # Create comparison chart
                models = list(comparison.keys())
                scores = [comparison[m]["avg_score"] * 100 for m in models]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=models,
                        y=scores,
                        text=[f"{s:.1f}%" for s in scores],
                        textposition='outside',
                        marker_color=['#667eea' if m == recommendation.recommended_model else '#cbd5e0' for m in models]
                    )
                ])
                
                fig.update_layout(
                    title=f"Performance Comparison: {selected_task.value.title()}",
                    yaxis_title="Accuracy (%)",
                    yaxis_range=[0, 100],
                    showlegend=False,
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed stats table
                with st.expander(" View Detailed Statistics"):
                    import pandas as pd
                    
                    data = []
                    for model, stats in comparison.items():
                        data.append({
                            "Model": model,
                            "Accuracy": f"{stats['avg_score']:.1%}",
                            "Consistency": f"{max(0, 1 - stats['std']*2):.1%}",
                            "Avg Speed": f"{stats['avg_latency']:.0f}ms",
                            "Tests": stats['count']
                        })
                    
                    df = pd.DataFrame(data)
                    st.dataframe(df, hide_index=True, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error generating recommendation: {e}")
            st.info("This task type might not have enough data yet. Try running more tests.")
    
    # Footer info
    st.markdown("---")
    st.markdown("###  About This Tool")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_runs = len(recommender.runs)
        st.metric("Data Points", f"{total_runs} runs")
    
    with col2:
        models = list(recommender.performance_matrix.keys())
        st.metric("Models Tested", len(models))
    
    with col3:
        total_tests = sum(
            sum(cat["count"] for cat in model_data.values())
            for model_data in recommender.performance_matrix.values()
        )
        st.metric("Total Tests", total_tests)
    
    st.markdown("""
    **How it works:** This tool analyzes real performance data from automated testing across multiple AI models.
    Recommendations are based on accuracy, speed, and consistency metrics collected over time.
    
    Data updated daily. [View source code](https://github.com/shishirnanga/llm-drift-monitor)
    """)
    
    # Model strengths section
    with st.expander(" View Model Strengths"):
        st.markdown("### What Each Model Excels At")
        
        for model in recommender.performance_matrix.keys():
            strengths = recommender.get_model_strengths(model)
            
            st.markdown(f"**{model}:**")
            for i, (category, score) in enumerate(strengths[:3], 1):
                st.markdown(f"  {i}. {category.title()}: {score:.1%}")
            st.markdown("")
    


if __name__ == "__main__":
    main()